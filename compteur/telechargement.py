r"""Installation de torch au premier lancement, via `uv` — version 2.

**Pourquoi `uv` et pas du code maison.** Une première version de ce fichier
téléchargeait les `.whl` en HTTP brut (reprise via `Range`, `zipfile`,
contrôle d'espace disque). Ça fonctionnait, mais c'était 700 lignes à
maintenir pour réinventer ce qu'un outil spécialisé fait déjà mieux : `uv`
connaît les index PyTorch, gère le cache, la reprise et les erreurs. On
l'embarque — un seul `.exe` — et on lui demande de faire le travail.

Ce que `uv` apporte, et qu'on ne répète à pas réécrire :

- il installe aussi un **Python autonome** dans notre dossier, ce qui évite
  de dépendre du Python du poste ;
- il choisit la bonne variante avec `--torch-backend cu126` (ou `cpu`),
  sans que nous ayons à connaître le nom exact du fichier `.whl` ;
- il gère ses propres reprises : interrompu puis relancé, il recommence où il
  s'était arrêté.

**La règle qui rend la v2 testable : on ne cherche torch QUE dans notre
dossier `%LOCALAPPDATA%`.** Jamais dans le Python du poste.

C'est le point le plus important du fichier. Un logiciel qui trouve un torch
« par hasard » sur la machine de développement marche sur cette machine et
échoue sur toutes les autres — et le premier test de l'utilisateur, sur un PC
vierge, est précisément le cas où il n'y a rien à trouver. Concrètement :

- `_verifier_import` lance le `python.exe` DU VENV avec `-I` (mode isolé : ni
  `PYTHONPATH`, ni `sitecustomize`) et **vérifie que `torch.__file__` est bien
  sous notre dossier**. Un torch trouvé ailleurs invalide l'installation ;
- `ajouter_au_sys_path` ne pose QUE `venv\Lib\site-packages`, en tête ;
- `torch_installe` exige cette preuve, au lieu de supposer que la présence
  d'un dossier suffit.

Emploi :

    from compteur.telechargement import installer_torch
    installer_torch("cuda", callback, dossier_cache())
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, NamedTuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

#: Version de torch visée. Épinglée : c'est celle avec laquelle les poids
#: `.pt` ont été produits, et une autre version changerait les résultats du
#: comptage — ce qu'aucun test ne détecterait.
TORCH_VERSION = "2.14.1"
TORCHVISION_VERSION = "0.29.1"

#: Variante CUDA de référence (machine de test : RTX 4070 SUPER).
VARIANTE_CUDA = "cu126"

#: Version de Python du venv. Doit correspondre à l'ABI de l'exécutable gelé
#: (construit avec Python 3.11) : les `.pyd` de torch sont tagués `cp311`, et
#: un Python 3.12 les refuserait. C'est la seule contrainte de version ici.
VERSION_PYTHON = "3.11"

#: Fichier écrit après une installation vérifiée. Sa présence EST la
#: définition de « c'est installé » : chercher un `torch/__init__.py`
#: autoriserait un `uv` interrompu entre deux étapes à prendre pour une
#: installation finie.
MARQUEUR = "installation.json"

#: Timeouts distincts : la résolution parle au réseau, l'installation écrit
#: sur le disque. 2,4 Go sur une connexion lente dépassent largement une
#: minute, et un timeout unique mettrait les deux à la même valeur — celle
#: des deux.
TIMEOUT_RESOLUTION = 900
TIMEOUT_INSTALLATION = 1800

#: Le cache de `uv` et son Python autonome sont redirigés DANS notre dossier :
#: rien ne sort de `%LOCALAPPDATA%\CompteurManifestation`, et le cache survit
#: au redémarrage comme le reste.
ENV_CACHE = "UV_CACHE_DIR"
ENV_PYTHON_INSTALL = "UV_PYTHON_INSTALL_DIR"


# ---------------------------------------------------------------------------
# Erreurs — trois messages distincts, en français
# ---------------------------------------------------------------------------


class ErreurInstallation(Exception):
    """Échec d'installation, avec un message fait pour l'utilisateur.

    Un message technique ici serait recopié tel quel dans une fenêtre : il doit
    dire ce qui s'est passé ET ce qu'on peut faire.
    """

    def __init__(self, message: str, *, details: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ErreurIntrouvable(ErreurInstallation):
    """Version ou variante absente de l'index.

    Distincte des deux autres pannes parce que le remède diffère : relancer ne
    sert à rien, il faut une autre version ou une autre variante CUDA.
    """

    def __init__(self, quoi: str) -> None:
        super().__init__(
            f"{quoi} est introuvable sur le serveur de téléchargement. "
            f"Le logiciel est peut-être trop ancien, ou le serveur a changé "
            f"ses versions.",
            details=f"introuvable : {quoi}",
        )


class ErreurReseau(ErreurInstallation):
    """Coupure réseau pendant la résolution ou le téléchargement."""

    def __init__(self, quoi: str, cause: str = "") -> None:
        super().__init__(
            f"La connexion a été interrompue pendant l'installation de "
            f"{quoi}. Relancez le logiciel : le téléchargement reprendra "
            f"où il s'est arrêté.",
            details=cause,
        )


class ErreurDisque(ErreurInstallation):
    """Pas assez de place sur le disque."""

    def __init__(self, manque: int) -> None:
        super().__init__(
            f"Espace disque insuffisant : il manque environ "
            f"{manque / (1024 ** 3):.1f} Go. Libérez de la place, ou "
            f"choisissez la version CPU, beaucoup plus légère.",
            details=f"manque {manque} octets",
        )


class Annulation(Exception):
    """Levée par le callback pour interrompre l'installation.

    Porter l'annulation dans une exception permet de la faire lever depuis
    n'importe quel point — y compris au milieu de la lecture de la sortie de
    `uv` — sans multiplier les tests `if annule: return`.
    """


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PHASE_TELECHARGEMENT = "telechargement"
PHASE_INSTALLATION = "installation"


class Progression(NamedTuple):
    """Un point d'avancement, pour la fenêtre d'installation.

    `phase` vaut `PHASE_TELECHARGEMENT` ou `PHASE_INSTALLATION`. C'est ce qui
    permet à l'interface d'afficher DEUX barres distinctes plutôt qu'une
    seule qui mouline pendant la décompression : ce sont deux attentes de
    nature différente pour l'utilisateur.
    """

    phase: str
    texte: str
    #: Unités progressions de la phase en cours. Octets en phase 1, nombre de
    #: paquets en phase 2 — voir `_installer_avec_uv` pour pourquoi.
    fait: int
    total: int
    #: Octets par seconde mesurés, 0 tant qu'on n'a pas deux mesures.
    vitesse: int
    #: Restant à faire, -1 si inconnu.
    restant: int

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.fait / self.total)

    @property
    def pourcent(self) -> int:
        return int(self.fraction * 100)

    @property
    def secondes_restantes(self) -> float:
        """Temps restant estimé, 0 si on n'a pas encore de mesure."""
        if self.vitesse <= 0 or self.restant < 0:
            return 0.0
        return self.restant / self.vitesse


#: Signature du callback. Il reçoit une `Progression` et peut lever
#: `Annulation`.
CallbackProgression = Callable[[Progression], None]


# ---------------------------------------------------------------------------
# Emplacements
# ---------------------------------------------------------------------------


def dossier_cache() -> pathlib.Path:
    r"""Racine de ce que la v2 installe : `%LOCALAPPDATA%\CompteurManifestation`.

    **Pas de `C:\Program Files`, et donc pas de droits administrateur.** Le
    premier lancement ne doit pas déclencher de UAC : un logiciel de terrain
    qui demande d'être élevé perd la moitié de ses utilisateurs, et l'UAC
    finit refusée. `%LOCALAPPDATA%` est en écriture pour l'utilisateur
    connecté, sans privilège.

    Ce n'est pas un temporaire : un téléchargement de 2,4 Go doit survivre à un
    redémarrage, sinon l'utilisateur le refait à chaque coupure de courant.
    """
    base = os.environ.get("LOCALAPPDATA")
    if not base:  # pragma: no cover — toujours défini sous Windows
        base = str(pathlib.Path.home())
    return pathlib.Path(base) / "CompteurManifestation"


def dossier_venv(dossier: pathlib.Path | None = None) -> pathlib.Path:
    """Le venv où torch est installé : `<cache>\venv`."""
    return (dossier or dossier_cache()) / "venv"


def chemin_python(dossier: pathlib.Path | None = None) -> pathlib.Path:
    """`python.exe` DU VENV.

    **Jamais `sys.executable`.** Dans un exécutable gelé, `sys.executable` est
    `CompteurManifestationV2.exe` : l'appeler en sous-processus pour tester un
    import relancerait l'application entière, fenêtre comprise. C'est un bug
    qui ne se voit pas depuis les sources, où `sys.executable` est précisément
    le bon interpréteur.
    """
    return dossier_venv(dossier) / "Scripts" / "python.exe"


def chemin_site(dossier: pathlib.Path | None = None) -> pathlib.Path:
    """Dossier `site-packages` du venv — c'est celui qu'on pose sur `sys.path`."""
    return dossier_venv(dossier) / "Lib" / "site-packages"


def uv_exe(dossier: pathlib.Path | None = None) -> pathlib.Path | None:
    """Chemin de `uv.exe`, embarqué avec l'exécutable. `None` s'il est absent.

    Cherché dans cet ordre STRICT :

    1. `sys._MEIPASS`, où PyInstaller pose les données embarquées ;
    2. à côté de l'exécutable ;
    3. à la racine du dépôt, pour `python main.py` ;
    4. dans le `PATH`, dernier recours.

    **Le `PATH` est un pis-aller, pas une solution.** Un `uv` déjà installé
    sur le poste ferait que la v2 fonctionnerait chez le développeur et
    échouerait chez l'utilisateur. D'où l'avertissementjournalisé à ce
    niveau, et l'obligation de copier `uv.exe` dans le `.spec`.
    """
    # L'ordre est STRICTEMENT « embarqué d'abord ». La copie embarquée fait
    # autorité même si un `uv` est installé sur le poste : c'est la seule
    # garantie que l'application se comporte pareil partout. Le `PATH` est un
    # secours affiché, jamais le premier choix.
    #
    # ⚠ `getattr(sys, "_MEIPASS", None)` et non `sys._MEIPASS` : cette
    # variable n'existe QUE dans le binaire gelé. Depuis les sources elle
    # manque, et un accès direct lèverait `AttributeError` — le genre
    # d'erreur qui n'apparaît que chez l'utilisateur, jamais en test.
    meipass = getattr(sys, "_MEIPASS", None)
    for candidat in (
        # 1. `sys._MEIPASS` : où PyInstaller pose les données embarquées.
        pathlib.Path(meipass) / "uv.exe" if meipass else None,
        # 2. à côté de l'exécutable (dossier `_internal` en one-dossier).
        pathlib.Path(sys.executable).parent / "uv.exe",
        # 3. dossier `_internal` explicite, pour les arborescences Improves.
        pathlib.Path(sys.executable).parent / "_internal" / "uv.exe",
        # 4. racine du dépôt, pour `python main.py` depuis les sources.
        pathlib.Path(__file__).resolve().parent.parent / "uv.exe",
    ):
        if candidat is not None and candidat.is_file():
            return candidat

    trouve = shutil.which("uv")
    if trouve:
        log.warning(
            "uv.exe absent de l'exécutable, mais trouvé dans le PATH (%s). "
            "L'application fonctionnera, mais elle dépendra de l'installation "
            "de uv sur ce poste : sur un PC vierge, elle échouera.",
            trouve,
        )
        return pathlib.Path(trouve)
    return None


# ---------------------------------------------------------------------------
# Détection du GPU, SANS torch
# ---------------------------------------------------------------------------


def nom_gpu() -> str:
    """Nom commercial de la première carte NVIDIA, ou chaîne vide.

    **On interroge `nvidia-smi`, pas torch** : à ce moment du programme, torch
    est précisément ce qu'on n'a pas encore installé. Sans cette détection, on
    proposerait un téléchargement CUDA de 2,4 Go à un portable sans carte
    graphique, qui n'aurait rien d'autre à faire que l'annuler.

    `nvidia-smi` est présent avec les pilotes NVIDIA, même sans CUDA Toolkit :
    c'est le bon outil pour cette question, et il est déjà là. Son absence
    renvoie `""` sans lever : un PC sans pilote NVIDIA est un cas normal.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ""
    try:
        fini = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001 — une carte absente n'est pas une panne
        log.debug("nvidia-smi inexécutable : %s", exc)
        return ""
    if fini.returncode != 0:
        return ""
    lignes = fini.stdout.decode("utf-8", "replace").strip().splitlines()
    return lignes[0].strip() if lignes else ""


def gpu_nvidia_present() -> bool:
    """Vrai si la machine a une carte NVIDIA utilisable par torch CUDA."""
    return bool(nom_gpu())


# ---------------------------------------------------------------------------
# Tailles attendues
# ---------------------------------------------------------------------------

#: Tailles relevées sur les index, en octets. Double rôle : afficher « 2,4 Go »
#: AVANT de commencer (pour que l'utilisateur puisse dire non tout de suite),
#: et vérifier l'espace disque avant de télécharger 2,4 Go pour rien.
TAILLES = {
    "cuda": 2602753280 + 6111881,
    "cpu": 124096268 + 1376792,
}


def taille_attendue(cuda: bool) -> int:
    """Octets attendus pour la variante demandée."""
    return TAILLES["cuda"] if cuda else TAILLES["cpu"]


# ---------------------------------------------------------------------------
# Lancement de `uv`
# ---------------------------------------------------------------------------


def _environnement(dossier: pathlib.Path) -> dict[str, str]:
    """Variables d'environnement pour `uv`.

    `UV_NO_PROGRESS` est nécessaire : `uv` écrit une barre ANSI sur son
    terminal qui, redirigée vers un pipe, produirait des caractères de contrôle
    dans notre journal et casserait notre parsing ligne à ligne. `NO_COLOR`
    supprime les codes couleur pour la même raison.
    """
    env = dict(os.environ)
    env[ENV_CACHE] = str(dossier / "uv-cache")
    env[ENV_PYTHON_INSTALL] = str(dossier / "uv-python")
    env["UV_NO_PROGRESS"] = "1"
    env["NO_COLOR"] = "1"
    return env


def _decoder(octets: bytes) -> str:
    return octets.decode("utf-8", "replace").replace("\r", "\n")


def _classer_erreur(texte: str) -> ErreurInstallation:
    """Traduit la sortie d'échec de `uv` en un message français actionnable.

    `uv` écrit ses erreurs en anglais, avec des formulations comme
    `No solution found` ou `Failed to fetch`. On ne laisse pas ça fuiter dans
    une fenêtre : on reconnaît les trois cas qui reviennent et on donne le
    message que l'utilisateur peut réellement traiter.

    L'ordre des tests est significatif : voir le commentaire sur le réseau.
    """
    bas = texte.lower()
    # L'ordre compte : une coupure réseau sur l'index PyPI se lit
    # « Failed to fetch ... 404 Not Found », et la classer comme « version
    # introuvable » enverrait l'utilisateur vérifier sa version alors que le
    # problème est transient. On teste donc le RÉSEAU d'abord, et « not
    # found » seulement sur une résolution qui a abouti.
    if any(
        mot in bas
        for mot in (
            "failed to fetch",
            "connection",
            "timeout",
            "timed out",
            "network",
            "dns",
            "unexpected eof",
            "reset by peer",
        )
    ):
        return ErreurReseau("torch", texte[-300:])
    if "no solution found" in bas or "404" in bas or "not found" in bas:
        return ErreurIntrouvable("la version demandée de torch")
    if any(mot in bas for mot in ("no space", "disk full", "not enough space")):
        return ErreurDisque(0)
    return ErreurInstallation(
        "L'installation du moteur de calcul a échoué. Relancez le logiciel "
        "pour réessayer ; si le problème persiste, le message détaillé est "
        "dans le journal.",
        details=texte[-300:],
    )


def _taille_libre(dossier: pathlib.Path) -> int:
    """Octets libres sur le disque qui PORTE le dossier.

    On remonte jusqu'au premier dossier existant : sous Windows,
    `%LOCALAPPDATA%` n'existe pas encore au tout premier lancement, et
    `disk_usage` lèverait dessus.
    """
    sonde = dossier
    while not sonde.exists() and sonde != sonde.parent:
        sonde = sonde.parent
    return shutil.disk_usage(sonde).free


def _verifier_disque(dossier: pathlib.Path, requis: int) -> None:
    """Lève `ErreurDisque` si le volume ne peut pas accueillir l'installation.

    Vérifié AVANT le téléchargement : découvrir le disque plein après 2,4 Go
    téléchargés est le pire moment possible, celui où l'utilisateur a attendu
    pour rien. On interroge le volume du dossier cible et non celui du dossier
    courant — sous Windows, les deux sont souvent sur des disques différents.
    """
    if requis <= 0:
        return
    libre = _taille_libre(dossier)
    if libre < requis:
        raise ErreurDisque(requis - libre)


def _executer(
    commande: list[str],
    dossier: pathlib.Path,
    timeout: int,
    texte_si_echec: str,
) -> str:
    """Lance `uv` en capturant sa sortie, et lève une erreur française.

    `check=True` ferait remonter l'exception brute de `subprocess`, avec la
    sortie d'`uv` que personne ne peut traduire. On capture donc pour
    traduire nous-mêmes, et on renvoie la sortie pour les cas où elle
    contient une information utile.
    """
    log.info("uv : %s", " ".join(commande))
    try:
        fini = subprocess.run(
            commande,
            capture_output=True,
            timeout=timeout,
            env=_environnement(dossier),
        )
    except subprocess.TimeoutExpired as exc:
        raise ErreurReseau(texte_si_echec, f"délai dépassé après {timeout} s") from exc
    except OSError as exc:
        raise ErreurInstallation(
            f"Impossible de lancer l'outil d'installation ({texte_si_echec}). "
            f"Les fichiers du logiciel sont peut-être incomplets.",
            details=str(exc),
        ) from exc
    sortie = _decoder(fini.stdout) + _decoder(fini.stderr)
    if fini.returncode != 0:
        raise _classer_erreur(sortie)
    return sortie


# ---------------------------------------------------------------------------
# Progression : lecture de la sortie de `uv`
# ---------------------------------------------------------------------------

#: Tailles telles que `uv` les écrit : `2.4GiB`, `12.0MiB`, `512.0KiB`.
#:
#: **Triées de la plus longue à la plus courte, volontairement.** En itérant dans
#: l'ordre d'insertion, `"b"` testerait `"2.4gib"` et matcherait — puisque
#: `endswith("b")` est vrai — donnant `"2.4gi"` comme nombre, soit 0 octets.
#: Toute la barre de progression resterait à 0 %, silencieusement, pendant les
#: 2,4 Go de torch. D'où le tri explicite et le test qui verrouille la valeur
#: réelle, pas seulement l'absence d'exception.
_UNITE = dict(
    sorted(
        {"b": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4}.items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
)


def _analyser_taille(texte: str) -> int:
    """`2.4GiB` -> octets. 0 si `uv` n'a donné qu'un nom sans taille.

    La comparaison se fait en minuscules des DEUX côtés : `uv` écrit `GiB`,
    et `endswith("gib")` sur une chaîne qui garde sa casse ne trouve jamais
    rien. Le résultat silencieux serait 0 pour TOUS les paquets, donc une
    barre de progression à 0 % pendant 2,4 Go — précisément le défaut que
    cette fonction existe pour éviter. D'où le test dédié.
    """
    texte = texte.strip()
    bas = texte.lower()
    for unite, facteur in _UNITE.items():
        if bas.endswith(unite):
            try:
                return int(float(texte[: -len(unite)]) * facteur)
            except ValueError:
                return 0
    try:
        return int(texte)
    except ValueError:
        return 0


#: Cadence du battement de cœur qui anime la barre quand `uv` se tait.
#:
#: 0,25 s : assez fin pour que la barre bouge à l'œil (quatre rafraîchissements
#: par seconde), assez large pour ne pas inonder la boucle d'événements Qt.
BATTEMENT_PROGRESSION = 0.25

#: Débit supposé tant qu'aucune mesure n'a pu être faite, en octets/seconde.
#:
#: Valeur volontairement MODESTE (5 Mo/s, une connexion grand public). Elle ne
#: sert qu'à faire bouger la barre entre le début du téléchargement de torch et
#: la première mesure réelle. La surestimer ferait racedevancer la barre vers
#: 100 % puis la figerait — exactement le défaut qu'on répare, déplacé.
DEBIT_INITIAL = 5 * 1024**2


class _SuiviTelechargement:
    """Traduit les lignes de `uv` en progression, paquet par paquet.

    `uv` écrit sur sa sortie, en texte, une ligne par paquet :

        Downloading torch (2.4GiB)
         Downloaded torch

    **Et RIEN ENTRE LES DEUX. C'est mesuré, pas supposé.** Sur une
    installation réelle de `uv` 0.7.0 (10 paquets, roue de 118 Mo), la sortie
    se réduit à : `Downloading torch` à t=0,52 s, puis `Downloaded torch` à
    t=16,2 s. Seize secondes de silence pour un seul paquet. Remettre
    `UV_NO_PROGRESS=0` ne change rien : sans terminal, `uv` n'écrit pas sa
    barre. Parser des « intervalles d'octets » est donc IMPOSSIBLE — il n'y a
    rien à parser.

    Reste la mesure physique : la taille du dossier de cache, qui croît pour de
    vraies raisons. **Écartée elle aussi, pour deux raisons mesurées** :
    `uv` décompresse la roue en des milliers de petits fichiers dans des
    `.tmpXXXX/`, donc le dossier ne mesure plus l'octet téléchargé mais l'octet
    DÉCOMPACTÉ, et le rapport entre les deux n'est pas connu ; surtout le scan
    récursif coûte 1,2 à 3,3 s pour 15 000 fichiers — un battement de cœur
    qui vole 25 % du temps à l'utilisateur pour lire un pourcentage faux.

    **Il reste donc à ESTIMER**, et c'est un choix assumé : l'utilisateur n'a
    pas besoin d'exactitude, il a besoin de voir que ça avance. Une barre qui
    avance doucement sur une estimation est meilleure qu'une barre figée sur
    une valeur exacte.

    L'estimation est ancrée sur le débit RÉELLEMENT mesuré. Dès qu'un paquet
    se termine, on connaît un nombre d'octets et un nombre de secondes : le
    débit suit. Entre deux fins de paquet, la barre avance à ce débit — donc
    elle accélère sur une bonne connexion et ralentit sur une mauvaise, ce qui
    est l'inverse d'une barre qui ne bouge pas.

    Trois invariantes tiennent, quelles que soient les mesures :

    - **monotone** : `fait` ne recule JAMAIS ;
    - **plafonnée** : `fait <= total` ;
    - **jamais 100 % avant la fin** : le total de référence mesuré à l'avance
      reste le plancher, donc tant que torch n'est pas arrivé la barre ne peut
      pas afficher « terminé ».
    """

    def __init__(self, total_attendu: int = 0) -> None:
        #: nom -> taille annoncée
        self.attendus: dict[str, int] = {}
        #: nom -> taille acquise
        self.recus: dict[str, int] = {}
        #: dernière progression émise, pour calculer la vitesse
        self._precedente: tuple[float, int] | None = None
        #: Dénominateur de référence, mesuré à l'avance (voir `TAILLES`).
        #:
        #: SANS lui, la barre est fausse dans les deux sens, et les deux
        #: ont été observés sur une installation réelle :
        #:
        #: - `uv` annonce les paquets au fur et à mesure. Tant que torch
        #:   (2,4 Go) n'est pas annoncé, le total ne couvre que les petits
        #:   paquets déjà vus, et `fait / total` vaut **100 %** alors que
        #:   2,4 Go restent à venir ;
        #: - puis torch est annoncé en dernier, le total triple, et le
        #:   pourcentage **retombe à 0 %**. Une barre qui recule paraît buggée
        #:   et fait croire à un téléchargement relancé de zéro.
        #:
        #: On part donc du total MESURÉ (`taille_attendue`) et on ne fait que
        #: le compléter par ce que `uv` annonce. Le pourcentage ne peut plus
        #: ni mentir ni reculer.
        self._total_ref = total_attendu
        #: Ce que vaut `fait` au dernier battement. Sert de plancher : c'est
        #: lui qui rend la progression monotone, quelle que soit la mesure.
        self._plancher = 0
        #: nom -> (instant d'annonce, octets ESTIMES). Un paquet reste dans ce
        #: dictionnaire tant qu'il n'est pas fini.
        #:
        #: **Plusieurs paquets a la fois, et c'est mesure** : `uv` telecharge en
        #: parallele — sur l'installation de reference, torch (118 Mo), sympy
        #: (6 Mo), networkx (2 Mo) et setuptools (1,2 Mo) sont TOUS en cours
        #: entre t=0,8 s et t=5,2 s. Ne suivre qu'UN paquet en cours perdait
        #: torch des que sympy finissait, et la barre se gelait alors pendant
        #: les 10,5 s restantes : le defecto original, reapparu alors meme
        #: qu'on le corrigeait.
        self._en_vol: dict[str, tuple[float, int]] = {}
        self._partage_total: int = 0
        #: Débit retenu, en octets/seconde. `None` tant qu'on n'a pas mesuré.
        self._debit: int | None = None

    def ligne(self, texte: str) -> Progression | None:
        """Interprète un fragment de sortie de `uv`. `None` si rien à dire.

        **Les deux marqueurs ne se ressemblent pas, et c'est mesuré.** Sur une
        installation réelle, `uv` 0.7.0 écrit :

            Downloading torch (118.2MiB)     <- début, PAS d'espace initial
             Downloading torch               <- FIN, avec un ESPACE initial

        Le second n'est PAS `Downloaded torch` — cette forme n'existe pas dans
        `uv` 0.7.0. Et `uv` télécharge plusieurs paquets en parallèle : les fins
        arrivent dans un ordre IMPRÉDICTIBLE, pas dans l'ordre des débuts.

        Les distinguer par la seule présence d'une taille serait donc faux :
        ` Downloading torch` n'a pas de taille, et le prendre pour un début
        réenregistrerait `torch` à 0 octets — ce qui remit le total et la barre
        à zéro, définitivement. L'espace initial EST l'information, et il faut
        le lire AVANT tout `strip()`. C'est exactement le bug que la
        vérification de bout en bout a fait apparaître.
        """
        # `rstrip` seulement : l'espace initial porte le sens.
        t = texte.rstrip()
        if not t:
            return None

        # -- FIN d'un paquet. Forme MESURÉE : un espace initial, et le mot
        # `Downloading` — pas `Downloaded`. C'est la seule information qui
        # distingue la fin du début, puisque la fin ne porte pas de taille.
        if t.startswith(" "):
            nom = t.strip()
            for prefixe in ("Downloaded ", "Downloading "):
                if nom.startswith(prefixe):
                    return self._paquet_fini(nom[len(prefixe):].strip())
            return None

        # `Downloaded X` reste accepté : d'autres versions de `uv` l'écrivent
        # ainsi, et refuser cette forme ferait perdre la seule mesure de débit.
        if t.startswith("Downloaded "):
            return self._paquet_fini(t[len("Downloaded "):].strip())

        if t.startswith("Downloading "):
            corps = t[len("Downloading "):].strip()
            nom, _, reste = corps.partition(" ")
            taille = _analyser_taille(reste.strip("()"))
            # On n'écrase PAS une taille déjà connue : `uv` peut répéter la
            # ligne de début, et un 0 ici effacerait le total.
            if taille > 0 or nom not in self.attendus:
                self.attendus[nom] = taille
            # Ouverture du téléchargement. On NE REMPLACE PAS l'ensemble des
            # paquets en vol : `uv` en lance plusieurs à la fois.
            if nom not in self._en_vol:
                self._en_vol[nom] = (time.monotonic(), 0)
            return self._progression(f"Téléchargement de {nom}")
        return None

    def _paquet_fini(self, nom: str) -> Progression | None:
        """Enregistre la FIN d'un paquet et en déduit le débit mesuré.

        Méthode à part parce que cette branche est la SEULE qui donne un
        nombre d'octets exact — donc la seule qui puisse calibrer
        l'estimation entre deux fins de paquet.
        """
        if nom not in self.attendus:
            # Fin d'un paquet jamais vu commencer : sans taille, on ne peut
            # rien mesurer. Plutôt que de dégrader la barre, on se tait.
            return None
        taille = self.attendus[nom]
        maintenant = time.monotonic()
        # MESURE DU DÉBIT : un paquet terminé donne un nombre d'octets ET un
        # nombre de secondes.
        debut = self._en_vol.get(nom)
        if debut is not None and maintenant - debut[0] > 0.2:
            mesure = int(taille / (maintenant - debut[0]))
            # Lissage : le débit d'un seul petit paquet est du bruit — 1,2 Mo
            # partis en 0,4 s feraient croire à 3 Mo/s, un gros à 30. On
            # moyenne les mesures.
            self._debit = (
                mesure if self._debit is None
                else (self._debit + mesure) // 2
            )
        self.recus[nom] = taille
        self._en_vol.pop(nom, None)
        return self._progression(f"{nom} téléchargé")


    def battement(self) -> Progression | None:
        """Progression ESTIMÉE, sans nouvel événement de `uv`.

        C'est ce que la boucle appelle en attendant la sortie de `uv`. Sans
        elle, la barre reste figée pendant les 2,4 Go de torch — le défaut
        qu'on répare ici.

        Renvoie `None` quand il n'y a rien à dire : aucun paquet en cours,
        donc aucune estimation à faire. Une progression sans fait ni vitesse
        ne ferait qu'embrouiller l'interface.
        """
        if not self._en_vol:
            return None
        # Le nom affiché est celui du plus gros paquet encore en vol : c'est
        # lui que l'utilisateur attend, et le minuscule se termine en 0,4 s.
        nom = max(self._en_vol, key=lambda n: self.attendus.get(n, 0))
        return self._progression(f"Téléchargement de {nom}")

    def total(self) -> int:
        return sum(self.attendus.values())

    def _part_de(self, nom: str, debut: float) -> float:
        """Fraction du débit total qui revient à ce paquet.

        `uv` télécharge plusieurs paquets en parallèle et se partage la bande
        passante entre eux. Le débit qu'on mesure est donc celui de l'ensemble,
        pas celui d'un paquet. L'appliquer tel quel au plus gros le plafonne
        dès que la phase parallèle s'arrête et que ce débit partagé reste en
        mémoire — mesuré : l'estimation de torch se figeait à 9,2 Mo pendant 8
        des 15 s de son téléchargement.

        On répartit donc au PRORATA des tailles. C'est approximatif — `uv` ne
        dit rien de sa répartition — mais deux propriétés suffisent à
        l'utilisateur : la somme des estimations reste cohérente avec le débit
        réellement mesuré, et un paquet seul récupère bien 100 % du débit.

        Seuls les paquets qui DÉBUTENT À LA MÊME INSTANT partagent
        le débit. Un paquet déjà seul ne doit pas être pénalisé par le
        découpage : torch, seul après t=3 s, doit récupérer tout le débit.
        """
        taille = self.attendus.get(nom, 0)
        if taille <= 0:
            return 0.0
        # tolerance generous : deux paquets annonces a quelques dizaines de ms
        # l'un de l'autre partagent la connexion.
        memes = [
            n for n, (d2, _) in self._en_vol.items()
            if abs(d2 - debut) < 0.25 and self.attendus.get(n, 0) > 0
        ]
        if len(memes) <= 1:
            return 1.0
        total = sum(self.attendus.get(n, 0) for n in memes)
        return taille / total if total else 0.0


    def _progression(self, texte: str) -> Progression:
        maintenant = time.monotonic()
        total = max(self.total(), self._total_ref)
        if total <= 0:
            return Progression(
                phase=PHASE_TELECHARGEMENT,
                texte=texte,
                fait=0,
                total=1,
                vitesse=0,
                restant=-1,
            )

        # Paquets terminés : nombre d'octets EXACT.
        fait = sum(self.recus.values())

        # Paquets en vol : nombre d'octets ESTIMÉS, au débit mesuré.
        #
        # Deux corrections, chacune tirée d'une mesure, chacune nécessaire.
        #
        # **1. L'estimation se calcule DEPUIS LE DÉBUT du paquet**, et non
        # depuis le dernier battement : c'est le temps écoulé qui la fait
        # croître. La recalculer à chaque battement depuis zéro la ferait
        # stagner — c'est-à-dire afficher le défaut qu'on répare.
        #
        # **2. Le débit mesuré vaut pour l'ensemble des téléchargements en
        # cours.** Quand `uv` télécharge plusieurs paquets en parallèle — ce
        # qu'il fait, mesuré : torch, sympy, networkx et setuptools sont tous
        # en vol jusqu'à t≈3 s — ce débit est PARTAGÉ. L'appliquer tel quel au
        # plus gros paquet le plafonnait à 9,2 Mo dès t=3 s, et il y restait
        # ensuite : la barre se gelait de t=3 s à t=10,7 s, soit 8 des 15 s du
        # gros téléchargement. D'où `_part_de`, qui donne à chaque paquet la
        # fraction du débit qui lui revient.
        if self._en_vol:
            debit = self._debit or DEBIT_INITIAL
            for nom, (debut, deja) in list(self._en_vol.items()):
                cible = self.attendus.get(nom, 0)
                if cible <= 0:
                    # Taille inconnue : on ne peut rien estimer pour ce
                    # paquet, et surtout pas le faire mentir avec les autres.
                    continue
                ecoule = max(0.0, maintenant - debut)
                part = debit * self._part_de(nom, debut)
                # Jamais plus que la taille du paquet, jamais moins que ce
                # qu'on avait déjà affiché pour lui.
                estime = max(deja, min(cible, int(part * ecoule)))
                if estime != deja:
                    self._en_vol[nom] = (debut, estime)
                    fait += estime

        # Monotone : le plancher est la valeur déjà affichée. C'est
        # l'invariant qui protège l'affichage d'un reflux, quelle que soit
        # la finesse de l'estimation.
        fait = max(fait, self._plancher, 0)
        fait = min(fait, total)
        self._plancher = fait

        # Vitesse : mesurée sur ce que l'interface vient de voir réellement.
        vitesse = 0
        if self._precedente is not None:
            avant_t, avant_octets = self._precedente
            dt = maintenant - avant_t
            if dt > 0.4:  # en dessous, la mesure est du bruit
                vitesse = max(0, int((fait - avant_octets) / dt))
                self._precedente = (maintenant, fait)
        else:
            self._precedente = (maintenant, fait)

        return Progression(
            phase=PHASE_TELECHARGEMENT,
            texte=texte,
            fait=fait,
            total=total,
            vitesse=vitesse,
            restant=max(0, total - fait),
        )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def torch_installe(dossier: pathlib.Path | None = None) -> bool:
    """Vrai si torch est utilisable dans NOTRE venv.

    Le test est « le marqueur existe ET le Python du venv importe torch ET
    `torch` vient de notre venv ». Le troisième point est le garde-fou : sans
    lui, un torch trouvé ailleurs validerait l'installation, et la v2 marcherait
    sur la machine du développeur en échouant partout ailleurs.
    """
    dossier = pathlib.Path(dossier or dossier_cache())
    if not (dossier / MARQUEUR).is_file():
        return False
    if not chemin_python(dossier).is_file():
        return False
    return _verifier_import(dossier)


def _verifier_import(dossier: pathlib.Path) -> bool:
    """Vrai si le Python DU VENV importe torch, et qu'il vient de notre venv.

    Testé dans un sous-processus, et non dans le processus courant : importer
    torch ici chargerait 3 Go de DLL CUDA dans le processus de l'interface, ce
    qui retarderait l'ouverture de la fenêtre de plusieurs secondes — et ce
    qu'on cherche précisément à éviter dans la version 2.

    Le sous-processus reçoit `-I` (mode isolé : ni `PYTHONPATH`, ni
    `sitecustomize`, ni le `sys.path` du poste). Sans lui, un `PYTHONPATH`
    installerait un torch_ANYWHERE ferait passer le test sans que le venv soit
    bon — exactement le mensonge que ce garde-fou doit démasquer.

    Le code de sortie 3 est réservé au cas « importé mais hors de notre
    dossier », pour que le journal dise ce qui s'est passé au lieu de « pas
    importable ».
    """
    site = chemin_site(dossier)
    code = (
        "import sys, torch;"
        "c = torch.__file__.replace('\\\\', '/').lower();"
        "a = sys.argv[1].replace('\\\\', '/').lower();"
        "sys.exit(0 if c.startswith(a) else 3)"
    )
    try:
        fini = subprocess.run(
            [str(chemin_python(dossier)), "-I", "-c", code, str(site)],
            capture_output=True,
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("test d'import de torch impossible : %s", exc)
        return False
    if fini.returncode == 3:
        log.warning(
            "torch importé mais HORS de notre venv : garde-fou déclenché, "
            "cette installation ne sera pas utilisée."
        )
        return False
    if fini.returncode != 0:
        log.warning(
            "torch présent mais non importable : %s",
            fini.stderr.decode("utf-8", "replace")[-400:],
        )
        return False
    return True


#: Message du dernier échec, lisible par l'interface.
_DERNIER_MESSAGE = ""


def dernier_message() -> str:
    """Message du dernier échec d'`installer_torch`, ou chaîne vide."""
    return _DERNIER_MESSAGE


def installer_torch(
    peripherique: str,
    callback_progression: CallbackProgression | None = None,
    dossier_cible: pathlib.Path | None = None,
) -> bool:
    """Installe torch + torchvision dans notre venv. Renvoie `True` si utilisable.

    `peripherique` vaut `cuda` ou `cpu` — jamais `auto` : le choix doit être
    fait AVANT l'installation, puisqu'il détermine la variante téléchargée
    (2,4 Go contre 120 Mo). C'est à l'appelant de résoudre `auto`, via
    `gpu_nvidia_present()`.

    **Ne lève jamais.** Toute panne est convertie en `False` + message lisible
    par `dernier_message()`. Une exception ici remonterait jusqu'à `main()` et
    tuerait le processus AVANT la fenêtre — l'inverse exact de la promesse
    faite à l'utilisateur.

    Idempotent : si torch est déjà installé et importable depuis notre venv, ne
    réinstalle rien. C'est le chemin normal du deuxième lancement.
    """
    global _DERNIER_MESSAGE
    _DERNIER_MESSAGE = ""

    dossier = pathlib.Path(dossier_cible or dossier_cache())
    cuda = peripherique == "cuda"
    callback = callback_progression or (lambda _p: None)

    # 1. Déjà là ? Le cas le plus fréquent, et le seul qui ne coûte rien.
    if torch_installe(dossier):
        log.info("torch déjà installé dans notre venv")
        return True

    uv = uv_exe()
    if uv is None:
        _DERNIER_MESSAGE = (
            "L'outil d'installation (uv.exe) est absent du logiciel. "
            "Réinstallez le logiciel depuis son dossier d'origine."
        )
        log.error("%s", _DERNIER_MESSAGE)
        return False

    # 2. Espace disque, AVANT. Marge de 2 : l'archive téléchargée ET sa
    #    décompressée coexistent, et `uv` garde en plus son propre cache dans
    #    notre dossier. 2,4 Go téléchargés donnent ~5 Go sur disque.
    try:
        _verifier_disque(dossier, taille_attendue(cuda) * 2)
    except ErreurInstallation as exc:
        _DERNIER_MESSAGE = exc.message
        log.error("%s", exc.message)
        return False

    dossier.mkdir(parents=True, exist_ok=True)
    venv = dossier_venv(dossier)

    try:
        # 3. Le Python autonome, s'il manque. ~30 Mo, et l'étape qui produit
        #    le moins de sortie sur une machine vierge : on l'annonce pour que
        #    la fenêtre ne paraisse pas figée.
        callback(
            Progression(
                phase=PHASE_TELECHARGEMENT,
                texte="Préparation de Python",
                fait=0,
                total=0,
                vitesse=0,
                restant=-1,
            )
        )
        if not chemin_python(dossier).is_file():
            _executer(
                [str(uv), "venv", "--python", VERSION_PYTHON, str(venv)],
                dossier,
                TIMEOUT_RESOLUTION,
                "Python",
            )

        # 4. torch + torchvision, avec suivi de la progression.
        _installer_avec_uv(uv, dossier, cuda, callback)
    except Annulation:
        log.info("installation annulée par l'utilisateur")
        return False
    except ErreurInstallation as exc:
        _DERNIER_MESSAGE = exc.message
        log.error("%s", exc.message)
        return False
    except Exception as exc:  # noqa: BLE001 — rien ne doit fuir vers `main()`
        _DERNIER_MESSAGE = (
            "L'installation du moteur de calcul a échoué pour une raison "
            "inattendue. Relancez le logiciel ; le détail est dans le journal."
        )
        log.exception("installation terminée sur une exception : %s", exc)
        return False

    # 5. Vérification AVANT d'écrire le marqueur. Un marqueur écrit pour une
    #    installation cassée ferait confiance à l'aveugle au prochain
    #    lancement.
    if not _verifier_import(dossier):
        _DERNIER_MESSAGE = (
            "Le moteur de calcul a été téléchargé mais ne peut pas être "
            "chargé. C'est généralement un pilote graphique NVIDIA "
            "incompatible, ou une version de Windows trop ancienne."
        )
        log.error("%s", _DERNIER_MESSAGE)
        return False

    try:
        (dossier / MARQUEUR).write_text(
            json.dumps(
                {
                    "torch": TORCH_VERSION,
                    "torchvision": TORCHVISION_VERSION,
                    "cuda": cuda,
                    "venv": str(venv),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        # L'installation EST complète ; seul le marqueur manque, ce qui fera
        # réinstaller au prochain lancement. Mieux vaut un avertissement que
        # faire échouer une installation qui marche.
        log.warning("marqueur d'installation non écrit : %s", exc)

    _nettoyer_cache_uv(dossier, uv)

    log.info("torch installé dans %s (cuda=%s)", venv, cuda)
    return True


def _nettoyer_cache_uv(dossier: pathlib.Path, uv: pathlib.Path) -> None:
    """Supprime les `.whl` téléchargées une fois l'installation terminée.

    Mesuré après une installation CUDA réelle : le cache d'`uv` pèse **4,1 Go**,
    exactement comme le venv — les deux fois les mêmes octets. Les roues n'ont
    plus d'usage après l'installation : c'est du gaspillage de 4 Go sur le
    disque de l'utilisateur, pour rien.

    On passe par `uv cache clean --cache-dir` plutôt que par un
    `shutil.rmtree` : c'est l'outil lui-même qui connaît son format de cache
    (index, verrous, `CACHEDIR.TAG`), et le supprimer à la main laisserait des
    fichiers verrouillés sous Windows si une autre instance tourne.

    **Un échec ici n'est pas grave et ne doit pas faire échouer
    l'installation** : torch est déjà en place et fonctionne. Le cache
   bordé est un problème d'hygiène disk, pas de fonctionnement.
    """
    cache = dossier / "uv-cache"
    if not cache.is_dir():
        return
    try:
        fini = subprocess.run(
            [str(uv), "cache", "clean", "--cache-dir", str(cache)],
            capture_output=True,
            timeout=600,
            env=_environnement(dossier),
        )
    except Exception as exc:  # noqa: BLE001 — du gaspillage, pas une panne
        log.warning("nettoyage du cache impossible : %s", exc)
        return
    if fini.returncode != 0:
        log.warning(
            "nettoyage du cache renvoyé %s : %s",
            fini.returncode,
            _decoder(fini.stderr)[-200:],
        )
        return
    log.info("cache uv nettoyé (~4 Go libérés)")


def _installer_avec_uv(
    uv: pathlib.Path,
    dossier: pathlib.Path,
    cuda: bool,
    callback: CallbackProgression,
) -> None:
    """Appelle `uv pip install` en suivant sa sortie, phase par phase.

    `uv` ne donne pas de progression à l'octet près, mais il écrit deux
    marqueurs de phase distincts : `Downloading …` puis `Prepared N packages`
    (la décompression) et `Installed N packages` (la copie dans le venv). La
    coupure entre les deux vient donc de `uv` lui-même, pas d'une heuristique
    de notre côté — c'est ce qui permet à l'interface d'afficher deux barres
    qui advancement au bon moment.

    **La phase 2 compte des PAQUETS, pas des octets**, et le dit. `uv` ne
    communique pas le nombre d'octets qu'il écrit dans le venv ; annoncer un
    pourcentage d'octets ici serait une progression inventée, et une barre qui
    ment est pire qu'une barre honnête en packages.
    """
    python = chemin_python(dossier)
    variante = VARIANTE_CUDA if cuda else "cpu"
    commande = [
        str(uv),
        "pip",
        "install",
        "--python",
        str(python),
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        "--torch-backend",
        variante,
        # `--link-mode=copy` : `uv` utilise par défaut un lien dur vers son
        # cache. Le cache étant redirigé dans notre dossier, le lien serait
        # valide — mais sur un disque sans liens durs (certains partages
        # réseau, certains RAID), `uv` échoue. Copier est plus lent et marche
        # partout.
        "--link-mode=copy",
    ]

    log.info("uv : %s", " ".join(commande))
    try:
        proc = subprocess.Popen(
            commande,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_environnement(dossier),
            # `bufsize=0` : aucune tamponisation côté Python, donc `read()` rend
            # ce que `uv` a DÉJÀ écrit, sans attendre de remplir un tampon.
            # C'est la condition pour que le battement de cœur ait quelque
            # chose à lire en temps réel.
            bufsize=0,
        )
    except OSError as exc:
        raise ErreurInstallation(
            "Impossible de lancer l'outil d'installation.",
            details=str(exc),
        ) from exc

    suivi = _SuiviTelechargement(taille_attendue(cuda))
    sortie = b""

    # -- Lecture dans un fil séparé, fragments dans une file.
    #
    # Le problème : `uv` écrit `Downloading torch`, puis plus RIEN pendant les
    # 2,4 Go (mesuré, voir `_SuiviTelechargement`). Une boucle qui fait
    # `for ligne in proc.stdout` reste donc BLOQUÉE sur `read()` pendant toute
    # la durée du téléchargement, et aucun battement de cœur ne peut être
    # émis — la barre est figée par construction, indépendamment de ce qu'on
    # calcule. Séparer la lecture du traitement est donc ce qui rend la barre
    # vivante.
    fragments: queue.Queue = queue.Queue()

    def _lire() -> None:
        """Lit le flux et dépose des FRAGMENTS, pas des lignes.

        Fil démon : s'il survit à la fin du processus, il n'empêche rien de
        se terminer.
        """
        flux = proc.stdout
        if flux is None:  # pragma: no cover — PIPE est toujours là
            fragments.put(None)
            return
        tampon = b""
        try:
            while True:
                # Par MORCEAUX de 4096, pas octet par octet. Mesuré sur
                # 10 Mo de texte : 0,31 s en 4096 octets, 4,07 s octet par
                # octet. Le volume réel de `uv` est de l'ordre de 10 ko —
                # les deux marcheraient — mais 4096 voit les retours chariot
                # arriver en temps réel sans coût notable.
                morceau = flux.read(4096)
                if not morceau:
                    break
                tampon += morceau
                # Découpe sur le retour chariot ET le saut de ligne. Itérer
                # sur `proc.stdout` directement n'attendrait que le saut de
                # ligne final, c'est-à-dire la fin du téléchargement.
                while True:
                    i_nl = tampon.find(b"\n")
                    i_cr = tampon.find(b"\r")
                    if i_nl == -1 and i_cr == -1:
                        break
                    if i_nl == -1:
                        i = i_cr
                    elif i_cr == -1:
                        i = i_nl
                    else:
                        i = min(i_nl, i_cr)
                    fragment = tampon[:i]
                    tampon = tampon[i + 1:]
                    if fragment.strip():
                        fragments.put(fragment)
        except (OSError, ValueError) as exc:
            # Fin de lecture (processus tué) : sortie propre, l'erreur réelle
            # remonte par le code de retour.
            log.debug("lecture de la sortie de uv interrompue : %s", exc)
        finally:
            fragments.put(None)  # sentinelle : plus rien à lire

    lecteur = threading.Thread(target=_lire, name="uv-lecture", daemon=True)
    lecteur.start()

    def _traiter(fragment: bytes) -> None:
        """Applique un fragment de sortie de `uv` aux deux phases.

        Le fragment est passé à `suivi.ligne()` **sans `strip()`** : l'espace
        initial de ` Downloading torch` est ce qui distingue la fin d'un
        paquet de son début. Le retirer ici ferait réenregistrer le paquet à
        0 octet — mesuré : la barre se remettait à zéro au moment précis où
        chaque paquet se terminait.
        """
        texte = _decoder(fragment)

        # -- Phase 1 : téléchargement.
        progression = suivi.ligne(texte)
        if progression is not None:
            callback(progression)
            return

        # Ici le `strip()` ne gêne plus : on a affaire à des phases, et non
        # à un paquet dont la taille compte.
        bas = texte.strip().lower()

        # -- Bascule en phase 2, à l'instant où `uv` décompresse.
        if bas.startswith(("prepared ", "preparing ", "unpack")):
            total_telecharge = suivi.total()
            callback(
                Progression(
                    phase=PHASE_TELECHARGEMENT,
                    texte="Téléchargement terminé",
                    fait=total_telecharge,
                    total=total_telecharge or 1,
                    vitesse=0,
                    restant=0,
                )
            )
            callback(
                Progression(
                    phase=PHASE_INSTALLATION,
                    texte="Décompression des fichiers",
                    fait=0,
                    total=max(1, len(suivi.attendus)),
                    vitesse=0,
                    restant=-1,
                )
            )
            return

        if bas.startswith("installed "):
            total = suivi.total() or 1
            callback(
                Progression(
                    phase=PHASE_INSTALLATION,
                    texte="Installation terminée",
                    fait=total,
                    total=total,
                    vitesse=0,
                    restant=0,
                )
            )


    try:
        while True:
            try:
                fragment = fragments.get(timeout=BATTEMENT_PROGRESSION)
            except queue.Empty:
                # Aucun fragment, mais le téléchargement continue : c'est
                # exactement le cas de torch, 94 s d'affilée. Sans ce
                # battement, la barre est figée — le défaut qu'on répare.
                battement = suivi.battement()
                if battement is not None:
                    callback(battement)
                continue
            if fragment is None:
                break
            sortie += fragment
            _traiter(fragment)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

    try:
        proc.wait(timeout=TIMEOUT_INSTALLATION)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise ErreurReseau("torch", "délai dépassé pendant l'installation") from exc
    if proc.returncode != 0:
        raise _classer_erreur(_decoder(sortie))


# ---------------------------------------------------------------------------
# Intégration au processus
# ---------------------------------------------------------------------------


def ajouter_au_sys_path(dossier: pathlib.Path | None = None) -> bool:
    """Pose NOTRE `site-packages` en tête du `sys.path` du processus.

    À appeler AVANT tout `import torch`, donc au tout début du programme.
    Renvoie `False` si rien n'est installé — ce n'est pas une erreur, c'est
    l'état « téléchargement nécessaire ».

    **On ne pose que le dossier de notre venv** : ni sa racine, ni le dossier
    de l'exécutable, ni le dossier courant. Ajouter la racine ferait remonter
    des `.pyd` du build par accident. Le garde-fou « torch vient d'ici, et
    nulle part ailleurs » commence ici.
    """
    dossier = pathlib.Path(dossier or dossier_cache())
    if not (dossier / MARQUEUR).is_file():
        return False
    site = chemin_site(dossier)
    if site.is_dir() and str(site) not in sys.path:
        sys.path.insert(0, str(site))
        log.info("site-packages du venv posé sur sys.path : %s", site)

    # Le `site-packages` seul ne suffit PAS, et c'est un défaut qui n'apparaît
    # QUE dans l'exécutable gelé — jamais depuis les sources, où `sys.path`
    # contient déjà la stdlib du Python qui tourne.
    #
    # Mesuré sur l'exécutable : avec le seul `site-packages` ajouté,
    # `import torch` échoue sur `ModuleNotFoundError: No module named
    # 'timeit'`. `timeit` est dans la bibliothèque STANDARD, pas dans les
    # paquets. Or PyInstaller ne sert la stdlib que depuis sa
    # `base_library.zip` embarquée : un module absent de cette archive n'est
    # résolu par AUCUN finder, même présent sur le `sys.path`.
    #
    # Il faut donc ajouter la stdlib du Python du VENV. Elle est résolue en
    # INTERROGANT le venv lui-même (`python -c "import sysconfig; …"`) plutôt
    # qu'en supposant un chemin : selon la façon dont `uv` a construit le venv,
    # la stdlib est soit dans `<venv>\Lib`, soit dans le Python autonome qu'il
    # a téléchargé ailleurs. Deviner le chemin donnerait un `sys.path` qui
    # ment sur son contenu.
    #
    # Ajoutée EN FIN de liste : la stdlib embarquée garde la priorité, et le
    # venv ne sert qu'à combler ce qui manque.
    stdlib = _stdlib_du_venv(dossier)
    for chemin in stdlib:
        if str(chemin) not in sys.path:
            sys.path.append(str(chemin))
            log.info("stdlib du venv ajoutée en fin de sys.path : %s", chemin)
    if stdlib:
        _installer_finder_stdlib(stdlib)
    return True


#: Paquets stdlib dont l'archive gelée est INCOMPLÈTE : ils y sont présents,
#: mais sans leurs sous-modules. Mesuré sur l'installation de référence, en
#: lisant les imports réels de `torchvision` :
#:
#: - `xml.etree.ElementTree` (via `torchvision.datasets.voc`)
#: - `html.parser`        (via `torchvision.datasets.flickr`)
#:
#: `logging` en est VOLONTAIREMENT absent : il a un état global réel
#: (handlers, niveau, filtres) que purger brutalement casserait. Les autres
#: n'ont pas d'état significatif, et être réimportés est sans conséquence.
PAQUETS_STDLIB = ("xml", "html", "email", "http", "json", "unittest", "urllib")


def _construire_finder(dossiers: list[pathlib.Path]):
    """Construit le finder qui sert `PAQUETS_STDLIB` depuis le venv.

    Séparé de `_installer_finder_stdlib` pour être testable directement : c'est
    le TROIT test du fichier, parce que c'est le défaut qui a demandé trois
    essais ratés avant d'être trouvé.

    **La RACINE du nom, jamais le nom entier.** Mesuré : un finder qui
    ignorait les noms contenant un point laissait `xml.etree` partir vers
    l'archive gelée — donc reproduisait exactement l'échec qu'il corrigeait.
    """
    import importlib.machinery

    recherche = [str(d) for d in dossiers]

    class _FinderVenvStdlib:
        """Sert les paquets de la liste blanche depuis le venv, sous-modules
        compris."""

        _compteur_venv = True

        def find_spec(self, nom_complet, chemin=None, cible=None):
            if nom_complet.split(".")[0] not in PAQUETS_STDLIB:
                return None
            for dossier in recherche:
                spec = importlib.machinery.PathFinder.find_spec(
                    nom_complet, [dossier]
                )
                if spec is not None:
                    return spec
            return None

    return _FinderVenvStdlib()


def _installer_finder_stdlib(dossiers: list[pathlib.Path]) -> None:
    """Force la bibliothèque standard à venir du VENV, pour de vrai.

    **Le problème, mesuré trois fois de trois façons différentes.** Dans
    l'exécutable gelé, `xml` est déjà présent dans l'archive PyInstaller —
    mais sans son sous-module `xml.etree`. Résultat : `import torchvision`
    échoue sur `No module named 'xml.etree'`, alors que le venv a un
    `xml.etree` parfait. Même chose pour `html.parser`.

    Deux solutions ont été essayées et n'ont PAS marché :

    - **ajouter la stdlib au `sys.path`** : sans effet. PyInstaller place
      `PyiFrozenImporter` dans `sys.meta_path`, consulté AVANT le `sys.path`.
      Le module de l'archive gagne, et la recherche de ses sous-modules
      reste confinée à l'archive ;
    - **un finder en tête de `meta_path` limitant la liste blanche** : sans
      effet elle aussi, parce que `xml` EST DÉJÀ IMPORTÉ quand le finder
      s'installe (PySide6 l'a chargé au démarrage de la fenêtre). Un module
      dans `sys.modules` n'est plus jamais recherché, quel que soit le
      nombre de finders installés après coup. Le remède arrive donc trop
      tard, systématiquement.

    **La solution qui marche : purger `sys.modules` des paquets stdlib
    concernés, puis réinstaller le finder.** Après `del sys.modules[...]`,
    le下一次 `import xml.etree.ElementTree` repasse par la mécanique
    normale, et notre finder sert la version du venv. On purge au moment de
    l'installation, donc bien avant que torchvision n'en ait besoin.

    Purge limitée à la liste blanche, et réimportée immédiatement : c'est le
    seul effet de bord possible est de reconstruire ces objets, ce qui est
    sans danger pour des modules sans état global significatif (`xml`,
    `html`, `json`…). On ÉVITE `logging`, qui a un état réel — il est
    retiré de la liste.
    """
    if any(getattr(f, "_compteur_venv", False) for f in sys.meta_path):
        return

    import importlib.machinery

    recherche = [str(d) for d in dossiers]

    #: Paquets stdlib dont l'archive gelée est INCOMPLÈTE (présents, mais
    #: sans leurs sous-modules). Relevé sur l'installation de référence en
    #: lisant les imports réels de `torchvision`.
    PAQUETS = PAQUETS_STDLIB

    #: Modules ET sous-modules currently en cache sous ces racines.
    purges = [
        nom
        for nom in list(sys.modules)
        if nom.split(".")[0] in PAQUETS
    ]
    for nom in purges:
        del sys.modules[nom]
    if purges:
        log.info("%d module(s) stdlib purgé(s) pour re-routage vers le venv", len(purges))

    class _FinderVenvStdlib:
        """Sert les paquets de la liste blanche depuis le venv, sous-modules
        compris.

        Le test porte sur la RACINE du nom : mesuré, refuser les noms
        contenant un point laissait `xml.etree` partir vers l'archive, donc
        reproduisait l'échec qu'on corrigeait.
        """

        _compteur_venv = True

        def find_spec(self, nom_complet, chemin=None, cible=None):
            if nom_complet.split(".")[0] not in PAQUETS:
                return None
            for dossier in recherche:
                spec = importlib.machinery.PathFinder.find_spec(
                    nom_complet, [dossier]
                )
                if spec is not None:
                    return spec
            return None

    sys.meta_path.insert(0, _FinderVenvStdlib())
    log.info("finder stdlib venv installé sur %d paquets", len(PAQUETS))


def _stdlib_du_venv(dossier: pathlib.Path) -> list[pathlib.Path]:
    """Dossiers de la bibliothèque standard du Python du venv.

    Demandés au venv lui-même : c'est la seule source qui ne suppose rien.
    Retourne une liste éventuellement VIDE — sans stdlib, `torch` ne
    s'importera pas, mais l'appelant n'a pas à le savoir ici.
    """
    python = chemin_python(dossier)
    if not python.is_file():
        return []
    code = (
        "import sysconfig, json;"
        "print(json.dumps({k: sysconfig.get_path(k) for k in "
        "('stdlib', 'platstdlib', 'platlib', 'purelib')}))"
    )
    try:
        fini = subprocess.run(
            [str(python), "-I", "-c", code], capture_output=True, timeout=60
        )
        if fini.returncode != 0:
            return []
        chemins = json.loads(fini.stdout.decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        log.debug("stdlib du venv introuvable : %s", exc)
        return []
    return [
        pathlib.Path(chemins[k])
        for k in ("stdlib", "platstdlib")
        if chemins.get(k) and pathlib.Path(chemins[k]).is_dir()
    ]


def version_torch() -> str:
    """Version de torch importée, ou chaîne vide si absent.

    Import réel, donc coûteux : à n'appeler qu'une fois, pour l'indicateur de
    la barre de statut, quand la fenêtre est déjà là.
    """
    try:
        import torch
    except ImportError:
        return ""
    return str(getattr(torch, "__version__", ""))