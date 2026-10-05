r"""Tests de l'installation de torch via `uv` (version 2).

C'est le seul endroit du logiciel où le réseau et le disque entrent en jeu,
donc le seul où un bug ne se voit pas avant un premier lancement sur la
machine d'un opérateur. Trois choses sont verrouillées ici.

**1. Le garde-fou : torch vient de NOTRE dossier, et nowhere else.** C'est le
test le plus important du fichier, et il est mesurable : on fabrique un venv
dont `site-packages` contient un faux `torch`, et on vérifie que
`ajouter_au_sys_path` ne le valide pas. Sans ce garde-fou, la v2 marche sur la
machine du développeur (où torch est installé globalement) et échoue sur toutes
les autres — et l'utilisateur ne voit la différence qu'au premier essai sur un
PC vierge.

**2. La traduction des erreurs de `uv`.** `uv` parle anglais et écrit des
codes comme `No solution found`. On ne veut pas ça dans une fenêtre française
destinée à un opérateur sur le terrain.

**3. La dégradation.** `installer_torch` ne lève jamais : une exception
remonterait jusqu'à `main()` et tuerait le processus AVANT la fenêtre, ce qui
est l'inverse exact de la promesse faite à l'utilisateur.

Aucun test ne télécharge réellement : un test qui installe 2,4 Go n'est pas un
test.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from compteur import telechargement as tl


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. Le garde-fou : torch doit venir de NOTRE dossier
# ---------------------------------------------------------------------------


def test_ajouter_au_sys_path_ne_pose_que_le_site_de_notre_venv(tmp_path):
    """Un seul chemin doit être ajouté, et ce doit être NOTRE `site-packages`.

    Ni la racine du venv, ni le dossier de l'exécutable, ni le dossier courant.
    Ajouter la racine ferait remonter des `.pyd` du build par accident ;
    ajouter le dossier courant ferait trouver n'importe quel torch posé à côté
    de l'exécutable. Le garde-fou commence ici : si on ajoute trop de chemins,
    on peut donc retomber sur n'importe quoi.
    """
    site = tl.chemin_site(tmp_path)
    site.mkdir(parents=True)
    (tmp_path / tl.MARQUEUR).write_text("{}", encoding="utf-8")

    avant = list(sys.path)
    try:
        assert tl.ajouter_au_sys_path(tmp_path) is True
        assert sys.path[0] == str(site)
        # Exactement UN chemin ajouté : le reste du `sys.path` est intact.
        assert len(sys.path) == len(avant) + 1
    finally:
        sys.path[:] = avant


def test_un_marqueur_absent_ne_ajoute_rien(tmp_path):
    """Pas de marqueur = False et `sys.path` intact.

    C'est le chemin normal du premier lancement : il ne doit rien planter, et
    surtout ne pas ajouter un chemin qui n'existe pas — sous PyInstaller, ça
    ferait un `sys.path` mentant sur le contenu réel.
    """
    avant = list(sys.path)
    assert tl.ajouter_au_sys_path(tmp_path) is False
    assert sys.path == avant


def test_torch_hors_de_notre_dossier_est_rejete(tmp_path, monkeypatch):
    """LE test du garde-fou : un torch ailleurs ne vaut PAS installation.

    On simule le pire cas : le venv est complet, mais le Python du venv importe
    un torch installé ailleurs. Un logiciel qui valide ça marcherait sur la
    machine de développement et échouerait chez l'utilisateur — le défaut
    exact que ce module doit empêcher.
    """
    # Le faux Python du venv dit « torch vient de /ailleurs/ ».
    faux = tmp_path / "faux_python.py"
    faux.write_text(
        "import sys\n"
        "print('torch de /autre/partie', file=sys.stderr)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )

    def _faux_popen(cmd, **kwargs):
        class _Fini:
            returncode = 3
            stdout = b""
            stderr = b""

        return _Fini()

    monkeypatch.setattr(tl.subprocess, "run", _faux_popen)

    assert tl._verifier_import(tmp_path) is False, (
        "un torch importé hors de notre venv doit être rejeté"
    )


def test_un_python_absent_ne_compte_pas_comme_installe(tmp_path):
    """Un marqueur sans `python.exe` dans le venv n'est pas une installation.

    Le marqueur peut survivre à un nettoyage partiel du dossier : Windows
    vide `%LOCALAPPDATA%` à l'installation d'une nouvelle version de Windows,
    et un antivirus peut bloquer la création d'un exécutable. Dans les deux
    cas, il faut réinstaller.
    """
    (tmp_path / tl.MARQUEUR).write_text("{}", encoding="utf-8")
    assert tl.torch_installe(tmp_path) is False


# ---------------------------------------------------------------------------
# 2. Les erreurs de `uv` traduites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sortie_uv, type_attendu, fragment",
    [
        # 404 / version absente : relancer ne sert à rien.
        ("error: No solution found when resolving dependencies",
         tl.ErreurIntrouvable, "introuvable"),
        # Coupure réseau : relancer sert à quelque chose.
        ("error: Failed to fetch `https://pypi.org/simple/torch/`\n  Caused by: connection reset",
         tl.ErreurReseau, "relancez"),
        # Disque plein.
        ("error: No space left on device", tl.ErreurDisque, "disque"),
    ],
)
def test_les_trois_pannes_ont_trois_messages_differents(
    sortie_uv, type_attendu, fragment, monkeypatch, tmp_path
):
    """404, coupure réseau et disque plein doivent être distingués.

    Pas une question de style : les remèdes sont opposés. Sur un 404,
    relancer ne fera rien et l'utilisateur perdrait sa foi dans le logiciel ; il
    faut lui dire que le serveur a changé. Sur une coupure, il faut au
    contraire l'inciter à réessayer, parce que ça marchera.
    """
    # On court-circuite `uv` lui-même : ce qu'on teste ici est la
    # CLASSIFICATION de sa sortie d'échec, pas `uv`. Un vrai téléchargement de
    # 2,4 Go pour obtenir un 404 serait un test, lui, inutilisable.
    monkeypatch.setattr(tl, "uv_exe", lambda *a, **k: pathlib.Path("uv"))
    monkeypatch.setattr(tl, "_verifier_disque", lambda *a, **k: None)
    monkeypatch.setattr(tl, "_verifier_import", lambda dossier: False)

    def _echec(commande, dossier, timeout, texte):
        raise tl._classer_erreur(sortie_uv)

    monkeypatch.setattr(tl, "_executer", _echec)

    assert tl.installer_torch("cpu", None, tmp_path) is False
    message = tl.dernier_message()
    assert message, "un échec doit toujours laisser un message"
    assert fragment in message.lower()


# ---------------------------------------------------------------------------
# 3. La règle absolue : installer_torch ne lève jamais
# ---------------------------------------------------------------------------


def test_uv_absent_rend_false_avec_un_message(tmp_path, monkeypatch):
    """Sans `uv.exe` embarqué, on doit échouer proprement, pas lever.

    C'est le symptôme d'une livraison incomplète : le `.spec` a oublié de
    copier `uv.exe`. Le dire précisément permet de réparer, là où une
    exception `FileNotFoundError` laisserait l'utilisateur devant un crash.
    """
    monkeypatch.setattr(tl, "uv_exe", lambda *a, **k: None)

    assert tl.installer_torch("cuda", None, tmp_path) is False
    assert "uv.exe" in tl.dernier_message()


def test_le_disque_insuffisant_est_detecte_avant_de_lancer_uv(tmp_path, monkeypatch):
    """Le contrôle d'espace précède TOUT : rien ne doit être téléchargé.

    Un test qui vérifierait après le téléchargement verrait `uv` s'exécuter
    avant l'échec — et 2,4 Mo de CPU déjà brûlés pour rien.
    """
    monkeypatch.setattr(tl, "uv_exe", lambda *a, **k: pathlib.Path("uv"))
    monkeypatch.setattr(tl.shutil, "disk_usage", lambda _d: type("U", (), {"free": 5})())
    appele = {"n": 0}

    def _interdit(*a, **k):  # pragma: no cover — ne doit JAMAIS être atteint
        appele["n"] += 1
        raise AssertionError("uv ne doit pas être lancé sur un disque plein")

    monkeypatch.setattr(tl, "_executer", _interdit)
    monkeypatch.setattr(tl, "_installer_avec_uv", _interdit)

    assert tl.installer_torch("cuda", None, tmp_path) is False
    assert appele["n"] == 0
    assert "disque" in tl.dernier_message().lower()


def test_une_exception_inattendue_ne_fuit_pas(tmp_path, monkeypatch):
    """Une exception imprévue doit devenir `False` + message.

    C'est la garantie centrale : `preparer()` appelle `installer_torch` avant
    la construction de la fenêtre. Une exception ici tue le processus, et
    l'utilisateur se retrouve devant un double-clic sans effet.
    """
    monkeypatch.setattr(tl, "uv_exe", lambda *a, **k: pathlib.Path("uv"))
    monkeypatch.setattr(tl, "_verifier_disque", lambda *a, **k: None)

    def _boom(*a, **k):
        raise RuntimeError("catastrophe imprévue")

    monkeypatch.setattr(tl, "_executer", _boom)
    monkeypatch.setattr(tl, "_installer_avec_uv", _boom)

    assert tl.installer_torch("cpu", None, tmp_path) is False
    assert tl.dernier_message()


def test_deja_installe_ne_relance_pas_uv(tmp_path, monkeypatch):
    """Torch déjà là : AUCUNE commande, aucun téléchargement.

    C'est le chemin normal du deuxième lancement, et le test le plus visible
    par l'utilisateur s'il échoue : il verrait 2,4 Go se retélécharger.
    """
    monkeypatch.setattr(tl, "torch_installe", lambda dossier=None: True)
    appele = {"n": 0}

    def _interdit(*a, **k):  # pragma: no cover — ne doit jamais être atteint
        appele["n"] += 1

    monkeypatch.setattr(tl, "_executer", _interdit)
    monkeypatch.setattr(tl, "_installer_avec_uv", _interdit)
    monkeypatch.setattr(tl, "ajouter_au_sys_path", lambda *a, **k: True)

    assert tl.installer_torch("cuda", None, tmp_path) is True
    assert appele["n"] == 0


# ---------------------------------------------------------------------------
# 4. Le suivi de progression : deux phases
# ---------------------------------------------------------------------------


def test_la_sortie_de_uv_produit_deux_phases_distinctes():
    """Les lignes de `uv` doivent alimenter la phase 1 PUIS la phase 2.

    C'est le contrat que l'interface consomme pour afficher deux barres. Si
    `Prepared N packages` ne déclenchait pas la phase 2, la seconde barre
    n'apparaîtrait jamais et l'utilisateur reverrait à une barre unique qui
    mouille pendant la décompression.
    """
    suivi = tl._SuiviTelechargement()

    lignes = [
        "Downloading numpy (12.0MiB)",
        "Downloaded numpy",
        "Downloading torch (2.4GiB)",
        "Downloaded torch",
    ]
    phases = []
    for ligne in lignes:
        p = suivi.ligne(ligne)
        assert p is not None, f"ligne non reconnue : {ligne}"
        phases.append(p.phase)

    assert set(phases) == {tl.PHASE_TELECHARGEMENT}
    assert suivi.total() == int(12.0 * 1024**2) + int(2.4 * 1024**3)


def test_un_paquet_en_cours_compte_pour_la_moitie():
    """Un téléchargement en cours doit faire AVANCER la barre.

    Sans cela, la barre reste à 0 % pendant les 2,4 Go de torch et paraît
    gelée — l'utilisateur en déduit un blocage et tue le processus.
    """
    suivi = tl._SuiviTelechargement()
    avant = suivi.ligne("Downloading torch (2.4GiB)")
    assert avant is not None
    assert avant.fait > 0, "un paquet commencé doit déjà compter"
    assert avant.fait < avant.total


def test_la_barre_ne_recule_jamais():
    """Le pourcentage doit être MONOTONE, même si `uv` annonce tard un paquet.

    Observé sur une installation réelle : `uv` annonce
    `Downloading torchvision` alors que torch est déjà dans son cache et sur
    le point d'arriver. Le total bondit de 6 Mo à 2,58 Go et le pourcentage
    retombe de 50 % à 0 %. Une barre qui recule paraît buggée et fait croire
    à un téléchargement relancé depuis zéro.
    """
    # `total_attendu` = la taille MESURÉE des roues, comme en production.
    suivi = tl._SuiviTelechargement(int(2.4 * 1024**3) + int(12.0 * 1024**2))
    pcts = []
    for ligne in (
        "Downloading torchvision (6.0MiB)",
        "Downloaded torchvision",
        # torch n'était PAS annoncé avant : le total triple d'un coup.
        "Downloading torch (2.4GiB)",
        "Downloaded torch",
        # Et torch arrive en dernier, donc 100 % seulement à la fin.
        "Downloading numpy (12.0MiB)",
        "Downloaded numpy",
    ):
        p = suivi.ligne(ligne)
        assert p is not None
        pcts.append(p.pourcent)

    assert pcts == sorted(pcts), f"la barre recule : {pcts}"
    assert pcts[-1] == 100


def test_la_barne_ne_dit_pas_termine_tant_que_torch_est_inconnu():
    """100 % avant que torch ne soit annoncé = « terminé » alors qu'il reste 2,4 Go.

    Sans le total mesuré en amont, `uv` annonçant les paquets au fur et à
    mesure, le ratio des petits paquets seuls vaut 100 % — et la barre
    annonce la fin du téléchargement au tout début.
    """
    suivi = tl._SuiviTelechargement(int(2.4 * 1024**3))
    p = suivi.ligne("Downloading numpy (12.0MiB)")
    assert p is not None
    assert p.pourcent < 100, "la barre ne doit pas se remplir à 100 % si tôt"


def test_fait_ne_depasse_pas_total():
    """`fait <= total` doit toujours tenir : un ratio > 1 casserait l'affichage."""
    suivi = tl._SuiviTelechargement()
    for ligne in ("Downloading torch (2.4GiB)", "Downloaded torch"):
        p = suivi.ligne(ligne)
        assert p is not None
        assert p.fait <= p.total, f"{p.fait} > {p.total}"
        assert 0.0 <= p.fraction <= 1.0


def test_la_barne_ne_dit_pas_termine_tant_que_torch_est_inconnu():
    """100 % avant que torch ne soit annoncé = « terminé » alors qu'il reste 2,4 Go.

    Sans le total mesuré en amont, `uv` annonçant les paquets au fur et à
    mesure, le ratio des petits paquets seuls vaut 100 % — et la barre
    annonce la fin du téléchargement au tout début.
    """
    suivi = tl._SuiviTelechargement(int(2.4 * 1024**3))
    p = suivi.ligne("Downloading numpy (12.0MiB)")
    assert p is not None
    assert p.pourcent < 100, "la barre ne doit pas se remplir à 100 % si tôt"


def test_fait_ne_depasse_pas_total():
    """`fait <= total` doit toujours tenir : un ratio > 1 casserait l'affichage."""
    suivi = tl._SuiviTelechargement()
    for ligne in ("Downloading torch (2.4GiB)", "Downloaded torch"):
        p = suivi.ligne(ligne)
        assert p is not None
        assert p.fait <= p.total, f"{p.fait} > {p.total}"
        assert 0.0 <= p.fraction <= 1.0


def test_le_finder_stdlib_sert_les_sous_modules(tmp_path, monkeypatch):
    """`xml` seul ne suffit PAS : il faut aussi `xml.etree`.

    C'est le défaut mesuré dans l'exécutable gelé : `xml` est présent dans
    l'archive PyInstaller mais pas `xml.etree`, donc `import torchvision`
    échoue sur `No module named 'xml.etree'`. Un finder qui refuserait les
    noms contenant un point reproduirait exactement l'échec qu'il corrige.

    On IMPORTE réellement, avec le vrai finder en `meta_path` : résoudre un
    sous-module suppose que son parent soit dans `sys.modules`, donc
    `find_spec` seul ne prouve rien.
    """
    import importlib
    import sys as _sys

    racine = tmp_path / "stdlib"
    pkg = racine / "paqtst"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("MARQUEUR = 'parent'", encoding="utf-8")
    (pkg / "sousmodule").mkdir()
    (pkg / "sousmodule" / "__init__.py").write_text(
        "MARQUEUR = 'enfant'", encoding="utf-8"
    )

    monkeypatch.setattr(tl, "PAQUETS_STDLIB", ("paqtst",))
    finder = tl._construire_finder([racine])
    _sys.meta_path.insert(0, finder)
    # `racine` n'est PAS dans sys.path : c'est bien le finder qui résout, et
    # non un effet de bord du chemin.
    for nom in list(_sys.modules):
        if nom.split(".")[0] == "paqtst":
            del _sys.modules[nom]
    try:
        assert importlib.import_module("paqtst").MARQUEUR == "parent"
        assert importlib.import_module("paqtst.sousmodule").MARQUEUR == "enfant", (
            "le sous-module doit etre servi, sinon l'echec originel revient"
        )
        assert finder.find_spec("torch") is None, "un paquet hors liste ne doit pas etre servi"
        assert finder.find_spec("autre.truc") is None
    finally:
        _sys.meta_path.remove(finder)
        for nom in list(_sys.modules):
            if nom.split(".")[0] == "paqtst":
                del _sys.modules[nom]


def test_la_liste_des_paquets_stdlib_couvre_ce_que_torchvision_importe():
    """`xml` et `html` doivent être routés vers le venv, sinon l'inférence échoue.

    Relevé sur l'installation réelle, ce sont les deux dont l'archive gelée est
    incomplète. Les retirer ferait réapparaître `No module named 'xml.etree'`
    au premier lancement, chez l'utilisateur — et seulement chez lui, puisque
    la machine de développement a une stdlib complète sous les pieds.

    `logging` en doit être ABSENT : il a un état global (handlers, niveau) que
    la purge de `sys.modules` casserait.
    """
    for paquet in ("xml", "html"):
        assert paquet in tl.PAQUETS_STDLIB, f"{paquet} doit être routé vers le venv"
    assert "logging" not in tl.PAQUETS_STDLIB, (
        "purger logging casserait ses handlers et le niveau du logiciel"
    )


def test_les_tailles_de_uv_sont_comprises():
    """`2.4GiB`, `12.0MiB`, `512KiB` doivent être converties en octets.

    On dépend du format de `uv` pour la barre ; s'il change, la conversion doit
    échouer proprement (0) et non lever — sinon toute l'installation s'écroule
    sur une chaîne de caractères.
    """
    assert tl._analyser_taille("2.4GiB") == int(2.4 * 1024**3)
    assert tl._analyser_taille("12.0MiB") == int(12.0 * 1024**2)
    assert tl._analyser_taille("512KiB") == 512 * 1024
    assert tl._analyser_taille("") == 0
    assert tl._analyser_taille("torch") == 0


# ---------------------------------------------------------------------------
# 5. Emplacements
# ---------------------------------------------------------------------------


def test_le_cache_est_dans_localappdata_pas_program_files():
    """Ni `Program Files`, ni rights administrateur : le cache est à l'utilisateur.

    Un premier lancement qui demande d'être élevé perd la moitié des
    utilisateurs, et l'UAC finit refusée. Le test le verrouille parce que le
    chemins le plus « naturel » pour un installeur est précisément le mauvais.
    """
    cache = tl.dossier_cache()
    assert cache.parts[-1] == "CompteurManifestation"
    assert "Program Files" not in str(cache)


def test_le_cache_survit_au_redemarrage(monkeypatch):
    """%LOCALAPPDATA%, pas %TEMP% : un temporaire est vidé au redémarrage.

    Un téléchargement de 2,4 Go interrompu doit pouvoir reprendre ; dans un
    temporaire, il repartirait de zéro à chaque coupure de courant.
    """
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\test\AppData\Local")
    cache = tl.dossier_cache()
    assert "Temp" not in str(cache)
    assert cache.parts[-2:] == ("CompteurManifestation",) or cache.parts[-1] == (
        "CompteurManifestation"
    )


def test_le_python_du_venv_est_different_de_sys_executable(tmp_path):
    """Le test du venv ne doit JAMAIS utiliser `sys.executable`.

    Dans un exécutable gelé, `sys.executable` est notre `.exe` : l'appeler en
    sous-processus pour vérifier un import relancerait l'application entière,
    fenêtre comprise. Ce bug est invisible depuis les sources, où
    `sys.executable` est le bon interpréteur.
    """
    chemin = tl.chemin_python(tmp_path)
    assert chemin != pathlib.Path(sys.executable)
    assert chemin.parent.name == "Scripts"
    assert chemin.parent.parent.name == "venv"


def test_les_tailles_annoncees_sont_realistes():
    """Les tailles affichées doivent correspondre aux vrais poids des roues.

    Elles servent à deux choses : prévenir l'utilisateur avant de télécharger
    2,4 Go, et vérifier l'espace disque. Une taille fausse à 10 % près donne un
    contrôle de disque faux — le défaut qu'on veut éviter.
    """
    cuda = tl.taille_attendue(True)
    cpu = tl.taille_attendue(False)
    assert 2 * 1024**3 < cuda < 3 * 1024**3, "torch CUDA fait ~2,4 Go"
    assert 100 * 1024**2 < cpu < 200 * 1024**2, "torch CPU fait ~120 Mo"