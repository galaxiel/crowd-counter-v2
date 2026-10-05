r"""Hook PyInstaller V2 : n'embarquer que Qt6Core, Qt6Gui et Qt6Widgets.

**Pourquoi, et pourquoi `excludes` ne suffit pas.** Le hook officiel
`pyinstaller_hooks_contrib/stdhooks/hook-PySide6.py` fait
`collect_dynamic_libs("PySide6")`, ce qui copie **toutes** les DLL du paquet :
Qt6Quick, Qt6Qml, Qt6Pdf, Qt6OpenGL, Qt6Network, Qt6Svg, Qt6VirtualKeyboard,
`opengl32sw.dll`. Mesuré sur la distribution PySide6 installée ici : 79 Mo au
total, dont **36 Mo** qui ne servent à rien à un logiciel de comptage qui
n'utilise que des widgets.

Ajouter `"PySide6.QtQuick"` etc. à `excludes` du `.spec` ne retire PAS ces
DLL : les exclusions s'appliquent à l'analyse des imports, pas aux hooks — la
même raison que pour `hook-torch.py` (voir ce fichier). On remplace donc le
hook, exactement comme pour torch.

**Les DLL gardées, et pourquoi chacune est nécessaire.**

- `Qt6Core` : le noyau, importé par QtGui et QtWidgets.
- `Qt6Gui` : paint, images, polices — le compteur et la vidéo sont dessinés.
- `Qt6Widgets` : `QLabel`, `QPushButton`, `QScrollArea`, `QComboBox`.
- `Qt6Network` : PAS gardée volontairement. Le téléchargement de torch passe
  par `urllib` (voir `compteur/telechargement.py`), jamais par Qt, pour que
  le module reste testable sans boucle d'événements.

**Ce qui est exclu, et le risque réel.** `Qt6OpenGL` et `opengl32sw.dll`
(19,7 Mo) fournissent le rendu logiciel de Qt, utilisé quand aucun pilote
OpenGL n'est disponible — cas d'un vieux portable ou d'une session RDP. Une
fenêtre qui ne s'affiche pas sur une machine de terrain est pire qu'un
logiciel un peu plus gros : d'où un TEST (`test_demarrage.py`,
`test_le_repli_cpu_bascule_la_variante_telechargee` et surtout la
vérification manuelle de lancement documentée dans le rapport). Si un jour
une machine refuse d'afficher la fenêtre, `opengl32sw.dll` est la première
chose à remettre — c'est le SEUL retrait de cette liste qui soit un vrai
risque, et il se répare en une ligne.

**Ne pas confondre avec les `.pyd`.** Les fichiers `QtCore.pyd`,
`QtGui.pyd`, `QtWidgets.pyd` sont des EXTENSIONS Python et sont traités par
`collect_submodules` ; ce hook les laisse faire. Ce qui est listé ici, ce sont
les DLL natives voisines.

**Si ce hook casse l'affichage**, la liste de `DLL_GARDEES` est le point de
départ du débogage : ajouter une DLL se fait en ajoutant son nom, sans
toucher au reste du `.spec`.
"""

from __future__ import annotations

import os

from PyInstaller.utils.hooks import collect_dynamic_libs

#: DLL natives réellement utilisées. Tout le reste part à la poubelle.
DLL_GARDEES = (
    "Qt6Core",
    "Qt6Gui",
    "Qt6Widgets",
)

def _garder(src: str, dest: str) -> bool:
    """Vrai si `src` est une DLL qu'on garde."""
    nom = os.path.basename(src)
    sans_ext = os.path.splitext(nom)[0]
    # `Qt6Core` couvre `Qt6Core.dll` ; on tolère aussi les DLL de debug
    # suffixees (`Qt6Cored.dll`) en ne comparant que le début du nom, sinon le
    # suffixe de build ('d', 'debug') ferait réapparaître les doublons.
    return any(sans_ext.startswith(d) for d in DLL_GARDEES)


#: On appelle `collect_dynamic_libs` PUIS on filtre : c'est la seule façon
#: d'obtenir les couples `(src, dest)` que PyInstaller sait normaliser en
#: triplets. Filtrer la liste de sortie — plutôt que de chercher les fichiers à
#: la main — garantit que les noms de destination sont ceux que le runtime
#: attend, donc que `os.add_dll_directory` les trouve.
binaries = []
for _src, _dest in collect_dynamic_libs("PySide6"):
    if _garder(_src, _dest):
        binaries.append((_src, _dest))

#: `hiddenimports` : les trois modules réellement importés. Les autres binding
#: Python de PySide6 (`QtQuick`, `QtPdf`...) sont exclus explicitement, sinon
#: l'analyse statique de PyInstaller les découvre via `QtWidgets` et les
#: embarque — et un binding sans sa DLL planterait à l'import.
hiddenimports = [
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

excludedimports = [
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQml",
    "PySide6.QtQmlModels",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtNetwork",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtDBus",
    "PySide6.QtHelp",
    "PySide6.QtDesigner",
    "PySide6.QtUiTools",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
]

#: Les `.py` de PySide6 font plusieurs Mo et ne servent à rien : les binding
#: sont des `.pyd`. `module_collection_mode = "pyz"` (le défaut) évite
#: de les embarquer ; on le dit explicitement pour que le mode ne soit pas
#: changé par défaut plus tard.
module_collection_mode = "pyz"