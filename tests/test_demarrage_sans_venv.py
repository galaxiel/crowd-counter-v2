r"""Le premier écran ne doit dépendre d'ABSOLUMENT RIEN du venv.

**Le bug que ce fichier verrouille.** La v2 exclut torch du paquet et le
télécharge au premier lancement. Le commit `b24d8cc` a exclu `numpy` et
`PIL` au même moment, sur le raisonnement que « le venv les réinstallera ».
C'est vrai pour l'**inférence** — et faux pour le **démarrage** :

    File "main.py", line 55, in main
    File "interface\app.py", line 35, in <module>
    File "...\_internal\cv2\__init__.py", line 11, in <module>
        import numpy
    ModuleNotFoundError: No module named 'numpy'

`main.py` importait `interface.app` — donc `cv2`, donc `numpy` — AVANT que
`interface.demarrage.preparer()` ait jamais eu l'occasion de poser le
`sys.path` du venv. Le paquet est alors un paquet qui ne peut pas démarrer
sans avoir d'abord réussi à démarrer.

**Le principe que ces tests énoncent.** L'écran d'installation doit être
atteignable avec PySide6 et rien d'autre. Il n'a aucune raison de toucher au
moteur : il ne fait que piloter `uv`, qui est un exécutable externe. Donc
`interface.demarrage`, `interface.style` et `compteur.telechargement` ne
doivent importer aucun des paquets que le venv est censé fournir.

Ces tests sont volontairement **statiques** (AST) et **d'exécution**
(sous-processus avec des modules bloqués) : le premier dit quelle est la
règle, le second prouve qu'elle tient pour de vrai, dans un interpréteur neuf où
rien n'est déjà importé par hasard.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent

#: Les trois modules du chemin critique AVANT que le venv soit interrogeable.
#: `interface.app` n'y est PAS : c'est justement le fichier qu'on repousse
#: après `preparer`, parce qu'il tire `cv2` et `numpy`.
CHEMIN_CRITIQUE = [
    RACINE / "interface" / "demarrage.py",
    RACINE / "interface" / "style.py",
    RACINE / "compteur" / "telechargement.py",
]

#: Paquets que le VENV fournit. Les embarquer est un doublon ; ne pas les
#: embarquer ET les importer avant le venv = crash au démarrage. C'est
#: exactement la paire de fautes que `b24d8cc` a commise.
FOURNIS_PAR_LE_VENV = {"numpy", "PIL", "pillow", "torch", "torchvision", "scipy"}

#: `cv2` n'est PAS dans la liste ci-dessus : le venv ne l'installe pas
#: (`requirements.txt` ne le demande pas), donc l'embarquer est nécessaire.
#: C'est pour ça qu'il est importé — mais seulement APRÈS `preparer`.


def _modules_importes(fichier: pathlib.Path) -> set[str]:
    """Racines des modules importés AU CHARGEMENT du module.

    Seuls les `import` de premier niveau comptent, pas ceux enfouis dans une
    fonction : une importation paresseuse s'exécute quand on l'appelle, donc
    elle ne peut pas tuer le processus au chargement. `compteur.telechargement`
    fait `import torch` dans `version_torch()`, sous `try/except ImportError`
    et documenté « à n'appeler qu'une fois » — c'est correct, et un test qui
    l'interdirait ауrait raison de tout casser.
    """
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    racines: set[str] = set()
    fonctions = {
        noeud
        for parent in ast.walk(arbre)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
        for noeud in ast.walk(parent)
    }
    for noeud in ast.walk(arbre):
        if noeud in fonctions:
            continue
        noms: list[str] = []
        if isinstance(noeud, ast.Import):
            noms = [a.name for a in noeud.names]
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            noms = [noeud.module]
        for nom in noms:
            racines.add(nom.split(".")[0])
    return racines


def _imports_paresseux_apres_echec(fichier: pathlib.Path) -> list[str]:
    """Imports de paquets du venv faits DANS une fonction, sans garde-fou.

    Un `import torch` paresseux est acceptable s'il est protégé : au pire il
    renvoie « absent », ce que la fenêtre affiche. Il est dangereux s'il
    n'est pas protégé, parce qu'il transforme une panne du venv en mort du
    processus.
    """
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    coupables: list[str] = []
    for fonc in ast.walk(arbre):
        if not isinstance(fonc, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        gardes = {
            n
            for parent in ast.walk(fonc)
            if isinstance(parent, ast.Try)
            for n in ast.walk(parent)
        }
        for noeud in ast.walk(fonc):
            noms: list[str] = []
            if isinstance(noeud, ast.Import):
                noms = [a.name for a in noeud.names]
            elif isinstance(noeud, ast.ImportFrom) and noeud.module:
                noms = [noeud.module]
            for nom in noms:
                racine = nom.split(".")[0]
                if racine not in FOURNIS_PAR_LE_VENV or noeud in gardes:
                    continue
                ligne = getattr(noeud, "lineno", "?")
                coupables.append(f"{fonc.name}:{ligne} {nom}")
    return coupables


# ---------------------------------------------------------------------------
# 1. La règle, énoncée
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chemin", CHEMIN_CRITIQUE, ids=lambda p: p.name)
def test_le_premier_ecran_nimporte_rien_du_venv(chemin):
    """Aucun module du chemin critique ne touche un paquet du venv."""
    touches = _modules_importes(chemin) & FOURNIS_PAR_LE_VENV
    assert not touches, (
        f"{chemin.name} importe {touches}. Au premier lancement le venv "
        f"n'existe pas encore : l'écran d'installation doit s'ouvrir avec "
        f"PySide6 et rien d'autre, sinon l'exécutable meurt avant lui."
    )


@pytest.mark.parametrize("chemin", CHEMIN_CRITIQUE, ids=lambda p: p.name)
def test_le_premier_ecran_nimporte_pas_cv2(chemin):
    """`cv2` tire `numpy` à son propre import — le début exact du traceback.

    Séparé du test précédent parce que `cv2` n'est PAS fourni par le venv :
    l'embarquer est correct, l'importer trop tôt ne l'est pas. La confusion
    entre les deux est exactement ce qui a produit le bug.
    """
    assert "cv2" not in _modules_importes(chemin), (
        f"{chemin.name} importe cv2. cv2 importe numpy au chargement du "
        f"module, et numpy n'est pas embarqué : le binaire meurt avant le "
        f"premier pixel."
    )


@pytest.mark.parametrize("chemin", CHEMIN_CRITIQUE, ids=lambda p: p.name)
def test_aucun_import_paresseux_sans_garde(chemin):
    """Un `import torch` paresseux doit être protégé par un `try/except`.

    C'est ce qui distingue `version_torch()` (correct : au pire « absent »,
    la fenêtre s'ouvre quand même) d'un `import cv2` nu dans une fonction
    (fatal : le processus meurt, et la règle « l'application démarre
    TOUJOURS » devient une règle « l'application démarre si le réseau
    coopère »).
    """
    coupables = _imports_paresseux_apres_echec(chemin)
    assert not coupables, (
        f"{chemin.name} importe un paquet du venv sans garde-fou : {coupables}. "
        f"Si le venv est absent ou cassé, l'exception tue le processus au "
        f"lieu de renvoyer un état « absent » affiché dans la barre d'état."
    )


def test_main_nimporte_interface_app_qu_apres_preparer():
    """`main.py` doit repousser `interface.app` APRÈS `preparer()`.

    Le traceback du binaire gelé commence exactement par là : `main.py`
    ligne 55, `in main`, qui ouvre `interface.app`. On vérifie l'ORDRE des
    deux lignes, pas seulement leur présence : un `import` déplacé en haut du
    module resterait un bug.
    """
    source = (RACINE / "main.py").read_text(encoding="utf-8")
    arbre = ast.parse(source)

    lignes_app: list[int] = []
    lignes_preparer: list[int] = []
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.ImportFrom) and noeud.module:
            if noeud.module.startswith("interface.app"):
                lignes_app.append(noeud.lineno)
        if isinstance(noeud, ast.Call):
            fn = noeud.func
            nom = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if nom == "preparer":
                lignes_preparer.append(noeud.lineno)

    assert lignes_app, "main.py doit toujours construire la fenêtre principale"
    assert lignes_preparer, "main.py doit toujours appeler preparer()"
    assert max(lignes_app) > min(lignes_preparer), (
        f"main.py importe interface.app ligne {min(lignes_app)} et appelle "
        f"preparer ligne {min(lignes_preparer)} : l'import précède le "
        f"démarrage, donc cv2/numpy sont chargés avant que le venv ait été "
        f"posé sur sys.path."
    )


# ---------------------------------------------------------------------------
# 2. La règle, prouvée dans un interpréteur neuf
# ---------------------------------------------------------------------------

#: Trappe posée sur `sys.meta_path` : elle DEVANCE le `PathFinder` normal,
#: donc aucun `sys.modules` pré-rempli ne peut la contourner. On lance un
#: interpréteur virgin — `python -c` n'a importé que `sys` — pour que le
#: résultat soit le vrai comportement du gelé, pas un artefact de l'ordre
#: d'importation du lanceur de tests.
_TRAPPE = textwrap.dedent(
    """
    import sys

    class Bloque:
        def __init__(self, interdits):
            self.interdits = set(interdits)
        def find_module(self, nom, chemin=None):
            return self.find_spec(nom, chemin)
        def find_spec(self, nom, chemin=None, cible=None):
            if nom.split(".")[0] in self.interdits:
                raise ImportError(
                    "BLOQUE VOLONTAIREMENT: " + nom + " ne doit pas etre "
                    "importe avant que le venv soit pose sur sys.path"
                )
            return None

    sys.meta_path.insert(0, Bloque(__import__("os").environ["INTERDITS"].split(",")))
    """
)


def test_le_premier_ecran_demarre_avec_pyside6_seul():
    """Importer et ouvrir l'écran d'installation, sans numpy ni cv2.

    C'est le test qui reproduit la situation du binaire gelé au premier
    lancement, dans les conditions exactes du traceback : venv absent,
    `numpy` et `cv2` inatteignables. Si l'écran s'ouvre, alors le chemin
    critique ne dépend plus du venv, et l'exécutable peut démarrer.
    """
    interdits = ",".join(sorted(FOURNIS_PAR_LE_VENV | {"cv2"}))
    programme = _TRAPPE + textwrap.dedent(
        """
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        from interface.demarrage import preparer, DialogueInstallation
        from interface.style import appliquer_style
        from PySide6.QtWidgets import QApplication

        app = QApplication([])
        appliquer_style(app)

        # L'écran doit se CONSTRUIRE, pas seulement s'importer : une
        # importation paresseuse qui n'est jamais déclenchée ne prouve rien.
        d = DialogueInstallation("NVIDIA GeForce RTX 4070 SUPER", 2_600_000_000)
        assert d.label_titre.text().startswith("Installation"), d.label_titre.text()
        assert d.phase_telechargement is not None
        assert d.btn_cpu.text().startswith("Continuer sur CPU")
        assert preparer is not None
        d.close()
        print("OK premier ecran sans venv")
        """
    )

    env = dict(os.environ)
    env["INTERDITS"] = interdits
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(RACINE)

    resultat = subprocess.run(
        [sys.executable, "-c", programme],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(RACINE),
    )
    assert resultat.returncode == 0, (
        "le premier écran n'a pas pu s'ouvrir sans numpy/cv2 — c'est "
        f"exactement le crash du binaire gelé.\n"
        f"--- stdout ---\n{resultat.stdout}\n--- stderr ---\n{resultat.stderr}"
    )
    assert "OK premier ecran sans venv" in resultat.stdout


# ---------------------------------------------------------------------------
# 3. Le paquet : ce qui est embarqué doit suffire au chemin critique
# ---------------------------------------------------------------------------


def test_le_spec_embarque_numpy():
    """`numpy` doit être dans le paquet, même si le venv l'installera aussi.

    Le doublon coûte de la place, mais l'alternative mesurée est un
    exécutable qui ne démarre pas du tout. Un binaire cassé n'a pas de taille
    acceptable : 322 Mo de fichiers que personne ne peut ouvrir valent moins
    que 380 Mo qui s'ouvrent.
    """
    source = (RACINE / "crowd-counter-v2.spec").read_text(encoding="utf-8")
    arbre = ast.parse(source)

    exclus: set[str] = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.keyword) and noeud.arg == "excludes":
            for elt in getattr(noeud.value, "elts", []):
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    exclus.add(elt.value)

    assert "numpy" not in exclus, (
        "numpy est exclu du paquet alors que cv2 l'importe au chargement. "
        "L'exécutable meurt sur ModuleNotFoundError avant l'écran "
        "d'installation. C'est le bug exact de b24d8cc."
    )
    assert "numpy.libs" not in exclus, (
        "sans `numpy.libs`, numpy importe mais ne trouve pas ses DLL OpenBLAS : "
        "l'import échoue quand même, avec une erreur moins parlante."
    )
