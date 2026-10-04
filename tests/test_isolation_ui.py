"""Le paquet compteur/ ne doit jamais dépendre d'un toolkit graphique."""
import ast
import pathlib

import pytest

# Ancré sur l'emplacement de ce fichier, pas sur le répertoire courant : la
# suite doit passer même lancée depuis un autre dossier (scripts de la tâche 7,
# exécution PyInstaller depuis dist/ à la tâche 12).
RACINE = pathlib.Path(__file__).resolve().parent.parent
LIVRABLES = sorted((RACINE / "compteur").rglob("*.py"))
# Racines des toolkits. La détection se fait par préfixe, donc "PySide6.QtWidgets"
# est rattrapé par "PySide6" et n'a pas besoin d'être listé séparément.
MODULES_INTERDITS = {"PySide6", "tkinter", "PyQt5", "PyQt6", "wx"}


def imports_interdits(fichier: pathlib.Path) -> set[str]:
    """Modules graphiques importés par `fichier`, par comparaison de préfixe.

    Couvre `import X`, `import X.Y` et `from X.Y import Z` : l'AST donne
    "X.Y" dans les trois cas, et le préfixe rattrape tous les sous-modules
    d'une racine interdite.
    """
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    importes = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            importes.update(a.name for a in noeud.names)
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            importes.add(noeud.module)
    return {
        i
        for i in importes
        if any(i == m or i.startswith(m + ".") for m in MODULES_INTERDITS)
    }


def test_le_paquet_compteur_existe():
    assert (RACINE / "compteur" / "__init__.py").exists(), (
        "le paquet compteur/ doit exister dès la tâche 1"
    )


@pytest.mark.parametrize("chemin", LIVRABLES, ids=lambda p: p.name)
def test_aucun_import_graphique_dans_compteur(chemin):
    interdites = imports_interdits(chemin)
    assert not interdites, f"{chemin} importe {interdites}, interdit dans compteur/"


def test_detecte_import_qt_profond(tmp_path):
    """Non-vacuité : la forme `from PySide6.QtWidgets import X` doit être vue.

    C'est la forme qu'écriront les tâches 8-10. Avant la correspondance par
    préfixe, elle passait sous le radar du test paramétré.
    """
    source = tmp_path / "widget.py"
    source.write_text(
        "from PySide6.QtWidgets import QApplication\n"
        "import PySide6.QtCore\n"
        "from PyQt6.QtWidgets import QLabel\n"
        "import numpy\n",
        encoding="utf-8",
    )
    assert imports_interdits(source) == {
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PyQt6.QtWidgets",
    }


def test_module_propre_ne_declenche_pas(tmp_path):
    """Contre-test : un module sans import graphique ne déclenche rien."""
    source = tmp_path / "propre.py"
    source.write_text("import numpy as np\nimport json\n", encoding="utf-8")
    assert imports_interdits(source) == set()
