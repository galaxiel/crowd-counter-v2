"""Résolution des chemins sous PyInstaller (`sys.frozen`).

Ces tests simulent un exécutable gelé en posant `sys.frozen = True` et
`sys._MEIPASS` — c'est exactement ce que fait le bootloader PyInstaller. Ils
sont la contrepartie vérifiable du correctif de `compteur/config.py` : sans
eux, une régression (revenir à `parent.parent` sur `__file__`) ne se verrait
qu'au premier double-clic sur `dist/CompteurManifestation.exe`, c'est-à-dire
après une reconstruction de dix minutes.

`compteur.config` est importé au niveau module, mais il résout son chemin à
chaque appel : poser `sys.frozen` après l'import suffit, et c'est même
seulement ce qui permet de tester les deux régimes dans une même session.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from compteur.config import RACINE, Config, _racine, chemin_defaut_config

#: Attributs que le bootloader PyInstaller pose, et qu'un `python main.py` n'a pas.
ATTRIBUTS_GEL = ("frozen", "_MEIPASS")

#: Le dépôt, déduit de l'emplacement de ce fichier de test.
DEPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def source(monkeypatch):
    """Retire tout attribut de gel : le régime `python main.py`."""
    for attr in ATTRIBUTS_GEL:
        monkeypatch.delattr(sys, attr, raising=False)
    return monkeypatch


@pytest.fixture
def gele(monkeypatch):
    """Fabrique un faux bootloader : ``geler(meipass, executable=...)``."""

    def _geler(meipass, executable=None):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
        if executable is not None:
            monkeypatch.setattr(sys, "executable", str(executable), raising=False)
        return monkeypatch

    return _geler


def _ecrire_config(dossier: pathlib.Path, **champs) -> pathlib.Path:
    """Pose un ``config/default.json`` sous ``dossier``, défauts du dataclass inclus.

    Les clés non fournies reprennent `Config()` : seul ce qui est réellement
    écrit dans le fichier distingue une lecture correcte d'un repli.
    """
    p = dossier / "config" / "default.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    donnees = Config().vers_dict()
    donnees.update(champs)
    p.write_text(json.dumps(donnees, indent=2) + "\n", encoding="utf-8")
    return p


def _exe_factice(dossier: pathlib.Path) -> pathlib.Path:
    """Un `.exe` factice, là où le bootloader mettrait le vrai."""
    p = dossier / "dist" / "CompteurManifestation.exe"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"MZ")
    return p


# -- Régime source -------------------------------------------------------


def test_source_resout_le_depot(source):
    """Sans gel, la racine est le dépôt, deux niveaux au-dessus de `compteur/`."""
    assert _racine() == DEPO


def test_source_trouve_le_fichier_versionne(source):
    assert chemin_defaut_config() == DEPO / "config" / "default.json"
    assert chemin_defaut_config().exists(), "config/default.json doit être versionné"


def test_source_lit_reellement_le_fichier(source):
    """`Config.defauts()` doit venir du JSON, pas des valeurs du dataclass."""
    assert Config.defauts() == Config.depuis_fichier(chemin_defaut_config())


def test_racine_module_reprend_la_racine_du_depot(source):
    """`RACINE` reste utilisable tel quel hors gel (outils, scripts)."""
    assert RACINE == DEPO


# -- Régime gelé ---------------------------------------------------------


def test_gele_resout_meipass(gele, tmp_path):
    """Sous gel, la racine est `sys._MEIPASS`, PAS `__file__` et son parent.

    C'est le cœur du correctif : en onefile, `__file__` pointe dans le
    dossier temporaire d'extraction (`%TEMP%\\_MEIxxxxxx`), et `parent.parent`
    ne remonte nulle part — `config/default.json` était introuvable et
    `Config.defauts()` partait sur un repli silencieux.
    """
    gele(tmp_path / "_MEI123456")
    assert _racine() == tmp_path / "_MEI123456"


def test_gele_trouve_la_config_dans_meipass(gele, tmp_path):
    meipass = tmp_path / "_MEI123456"
    attendu = _ecrire_config(meipass)
    gele(meipass)
    assert chemin_defaut_config() == attendu
    assert chemin_defaut_config().exists()


def test_gele_defauts_lit_le_fichier_embarque(gele, tmp_path):
    """La preuve utile : une valeur ABSENTE du dataclass doit être lue.

    Avec le repli silencieux, `fenetre_lissage` vaudrait 1 (valeur codée en
    dur) au lieu de 7 (valeur embarquée). Ce test échoue si le chemin gelé
    n'est pas résolu.
    """
    meipass = tmp_path / "_MEI123456"
    _ecrire_config(meipass, fenetre_lissage=7, seuil_confiance=0.42)
    gele(meipass)
    c = Config.defauts()
    assert c.fenetre_lissage == 7
    assert c.seuil_confiance == pytest.approx(0.42)


def test_gele_sans_config_journalise_le_repli(gele, tmp_path, caplog):
    """Dossier gelé SANS `config/` : le repli doit rester journalisé.

    Un utilisateur ne voit pas le journal. Ce cas ne doit donc pas se
    produire — mais s'il se produit, il ne doit pas être silencieux non plus.
    """
    meipass = tmp_path / "_MEI_vide"
    meipass.mkdir()
    gele(meipass)
    with caplog.at_level("WARNING", logger="compteur.config"):
        c = Config.defauts()
    assert c == Config()
    assert "default.json" in " ".join(r.getMessage() for r in caplog.records)


def test_gele_sans_meipass_retombe_sur_lexecutable(gele, tmp_path, monkeypatch):
    """`sys.frozen` sans `_MEIPASS` (one-folder) : le dossier de l'exécutable.

    Sans ce repli, `pathlib.Path(None)` lèverait et l'application ne
    démarrerait pas du tout.
    """
    exe = _exe_factice(tmp_path)
    gele(tmp_path / "inconnu", executable=exe)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert _racine() == exe.parent
    assert chemin_defaut_config() == exe.parent / "config" / "default.json"


def test_gele_meipass_prioritaire_sur_executable(gele, tmp_path):
    """Les deux points sont définis en one-folder : `_MEIPASS` gagne.

    C'est le dossier où le `.spec` dépose `config/`, pas le dossier du `.exe`.
    """
    meipass = tmp_path / "_MEI"
    exe = _exe_factice(tmp_path)
    gele(meipass, executable=exe)
    assert _racine() == meipass


def test_gele_ne_depend_pas_du_repertoire_courant(gele, tmp_path, monkeypatch):
    """Le chemin gelé ne doit rien attendre du dossier courant.

    Un double-clic dans l'explorateur donne un dossier courant qui n'est ni
    le dépôt ni `dist/` : toute dépendance au CWD casserait au premier
    essai utilisateur, et le symptôme (config introuvable) ne dirait rien
    de la vraie cause.
    """
    cwd = tmp_path / "ailleurs"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    meipass = tmp_path / "_MEI"
    _ecrire_config(meipass, seuil_matching=0.71)
    gele(meipass)
    assert chemin_defaut_config() == meipass / "config" / "default.json"
    assert Config.defauts().seuil_matching == pytest.approx(0.71)