"""Le paquet compteur/ ne doit jamais dépendre d'un toolkit graphique."""
import ast
import pathlib

import pytest

LIVRABLES = list(pathlib.Path("compteur").rglob("*.py"))
MODULES_INTERDITS = {"PySide6", "PySide6.QtCore", "tkinter", "PyQt5", "PyQt6", "wx"}


def test_le_paquet_compteur_existe():
    assert (pathlib.Path("compteur") / "__init__.py").exists(), (
        "le paquet compteur/ doit exister dès la tâche 1"
    )


@pytest.mark.parametrize("chemin", LIVRABLES, ids=lambda p: p.name)
def test_aucun_import_graphique_dans_compteur(chemin):
    arbre = ast.parse(chemin.read_text(encoding="utf-8"))
    importes = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            importes.update(a.name for a in noeud.names)
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            importes.add(noeud.module)
    interdites = importes & MODULES_INTERDITS
    assert not interdites, f"{chemin} importe {interdites}, interdit dans compteur/"
