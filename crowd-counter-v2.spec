# -*- mode: python ; coding: utf-8 -*-
r"""PyInstaller : construit la version 2, SANS torch, qui le télécharge au premier
lancement.

    python -m PyInstaller --clean --workpath build/v2 crowd-counter-v2.spec

Le `COLLECT` porte `name="CompteurManifestationV2"`, donc la sortie est
`dist/CompteurManifestationV2/`. NE PAS passer `--distpath` en plus : le
dossier serait cree deux fois (`dist/X/CompteurManifestationV2/`).

Puis copier les poids à côté de l'exécutable :

    python tools/copier_modeles.py --dest dist/CompteurManifestationV2

**Ce que change la v2, et pourquoi c'est possible.** La v1 pesait 4,25 Go, dont
~3,8 Go de DLL CUDA amenées par `hook-torch.py` via `collect_dynamic_libs`
(voir `HISTOIRE.md` ; l'ancien `crowd-counter.spec` est dans l'historique git).
La v2 AJOUTE `torch`, `torchvision`
et `torchaudio` aux `excludes`, et embarque à la place un seul `uv.exe`.
Le logiciel installe torch au premier lancement dans
`%LOCALAPPDATA%\CompteurManifestation\`, sans droits administrateur — voir
`compteur/telechargement.py`.

**La taille obtenue n'est pas ~50 Mo, et il faut le dire franchement : 369,7 Mo**
(dossier complet, modèles `.pt` compris — comme la v1, qui pèse 4 349,5 Mo).
C'est un gain réel de 91,5 %, mais la cible de 50 Mo n'est pas atteignable, et
le rapport `docs/design/task-22-v2-telechargement-report.md` explique pourquoi
avec le détail des mesures.

Décomposition de `_internal/` (254,5 Mo) :

| Poste | Poids | Réductible ? |
|---|---|---|
| `cv2` | 111,8 Mo | non — OpenCV lit la vidéo |
| `PySide6` (QtCore/Gui/Widgets + 3 `.pyd`) | 70,2 Mo | non — c'est l'interface |
| racine (`uv.exe` 49 Mo, `python311.dll`…) | 66,5 Mo | non |
| `ultralytics`, `shiboken6`, `yaml`… | 6,1 Mo | non |

Autrement dit : **ce qui restait du torch n'était pas « 4,3 Go moins 50 Mo ».**
Le budget de 50 Mo supposait que le reste du logiciel tiendrait dans 50 Mo une
fois torch retiré ; il en fait 254. Deux surprises l'ont fait grimper :
`uv.exe` fait **49 Mo** là où l'estimation initiale en annonçait 15, et
`numpy`/`PIL`/`scipy` (41 Mo) ne pouvaient être exclus qu'APRÈS avoir vérifié
que `uv` les réinstalle bien dans le venv — **vérification qui s'est révélée
fausse pour le chemin critique, voir ci-dessous**.

**⚠ `numpy` est ré-introduit (05/10/2026) après un binaire qui ne démarrait
pas.** L'exclusion de `b24d8cc` était correcte pour l'inférence et fausse
pour le démarrage : `main.py` importait `interface.app` → `cv2` → `numpy`
AVANT que `preparer()` ait pu poser le venv sur `sys.path`. Sur le binaire
gelé : fenêtre ouverte, fermée aussitôt,
`ModuleNotFoundError: No module named 'numpy'`. Coût mesuré : `_internal`
passe de **254,5 à 280,7 Mo**, soit **+26,2 Mo** (`numpy` 5,9 +
`numpy.libs` 20,0 + `dist-info` 0,2) ; le paquet complet fait **397,4 Mo**,
dont 105,4 Mo de poids `.pt`. `PIL` et `scipy` restent exclus : rien ne les
touche sur le chemin critique. Douze tests verrouillent la règle :
`tests/test_demarrage_sans_venv.py`.

**Un point important, à ne pas confondre** : `cv2` et `PySide6` ne sont PAS
des doublons. Le venv construit au premier lancement contient `numpy`, `PIL`,
`torch` et `torchvision` — **ni `cv2`, ni `PySide6`**. Les embarquer est donc
nécessaire, pas redondant : les exclure casserait le démarrage, puisque
`main.py` importe PySide6 avant même que le venv soit interrogeable.

Ce qui ferait réellement descendre sous 200 Mo, et qui n'a PAS été fait parce
que cela sort du cadre de la tâche : passer à `opencv-python-headless`
(~80 Mo gagnés, parfaitement faisable car le logiciel n'affiche rien
d'OpenCV).

**Les exclusions torch sont RÉELLEMENT nécessaires ici, et le nom
« excludes » est trompeur.** PyInstaller applique `excludes` à l'analyse
statique, mais les HOOKS sont exécutés séparément et ne consultent pas la
liste : `hook-torch.py` appellerait `collect_dynamic_libs("torch")` et
recollecterait 3,8 Go de DLL quoi qu'il arrive. D'où les deux points suivants,
qui sont la vraie raison de ce fichier.

**1. `hooks-v2/` prend le pas sur le hook officiel.** `hookspath` est
répercuté dans `sys.meta_path` par `pyimod03_importers.PyiFrozenImporter` :
PyInstaller teste le chemin UTILISATEUR **avant** ceux de
`pyinstaller_hooks_contrib`, et en cas de collision, l'ordre d'insertion
décide. Un `hook-torch.py` qui retourne des listes VIDES gagne donc la
course contre le hook officiel, et torch n'est pas collecté.

**2. `hooks-v2/hook-torchvision.py` fait la même chose.** Le hook officiel
comme l'ancien hook v1 (`hooks/hook-torchvision.py`, retiré du projet) font
`import torchvision` et
`collect_dynamic_libs("torchvision")` : les deux amèneraient 600 Mo de DLL CUDA
au build, et le faire échouer en est pire. La version v2 ne le charge pas
du tout — voir `hooks-v2/README` pour la preuve par la taille.

**3. `main.py` doit importer torch TARD.** `sys.path` reçoit le dossier de
cache au tout premier import (voir `interface.demarrage.preparer`), donc
PyInstaller ne peut plus voir torch comme analysable. C'est pourquoi
`main.py` importe `compteur.telechargement` AVANT `interface.app`, et que le
paquet `compteur` reste importable sans torch — hypothèse que les 481 tests
existants vérifient déjà.

Construction : voir la commande en tête de fichier.
"""

import os
import pathlib
import shutil
import sys

from pathlib import Path

RACINE = Path(SPECPATH)
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

#: `uv.exe`, l'outil qui installe torch au premier lancement. Cherché dans
#: trois endroits, par ordre de préférence :
#:
#: 1. le dépôt lui-même (`uv.exe` à la racine) — c'est là qu'on le met pour
#:    que le build soit reproductible et qu'aucune installation globale ne
#:    puisse rendre le build dépendant d'un poste ;
#: 2. le répertoire des outils utilisateur (`%USERPROFILE%\.local\bin`), où
#:    l'installeur uv le pose ;
#: 3. le `PATH`.
#:
#: **S'il n'est trouvé nulle part, le build s'arrête ici.** Embedder un `uv`
#: pris dans le `PATH` sans le dire produirait un exécutable qui fonctionne
#: chez le développeur et échoue chez l'utilisateur — le défaut exact que la
#: v2 doit éviter. Mieux vaut un build qui échoue bruyamment.
def _trouver_uv():
    candidats = [
        RACINE / "uv.exe",
        pathlib.Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin" / "uv.exe",
    ]
    trouve = shutil.which("uv")
    if trouve:
        candidats.append(pathlib.Path(trouve))
    for candidat in candidats:
        if candidat.is_file():
            return candidat
    raise SystemExit(
        "uv.exe est introuvable. Installez uv (https://astral.sh/uv), ou "
        "copiez uv.exe a la racine du depot, puis relancez le build."
    )


UV_EXE = _trouver_uv()

a = Analysis(
    [str(RACINE / "main.py")],
    pathex=[str(RACINE)],
    binaries=[],
    datas=[
        (str(RACINE / "config" / "default.json"), "config"),
        # `uv.exe` part dans `_internal/`, donc dans `sys._MEIPASS`. C'est le
        # premier endroit que `compteur.telechargement.uv_exe` cherche.
        (str(UV_EXE), "."),
    ],
    # torch et torchvision sont VOLONTAIREMENT absents d'`hiddenimports`,
    # contrairement à ce que faisait le spec v1. Les lister ici alors qu'ils sont
    # exclus produit un avertissement « hidden import not found » et, pire,
    # fait tenter à PyInstaller de les emballer quand même. Ils sont
    # chargés depuis le cache à l'exécution par `compteur.telechargement`.
    hiddenimports=[
        # -- Détection : le cœur du logiciel. `ultralytics` importe torch au
        #    niveau de son propre `__init__` : ce n'est analysé qu'AU
        #    moment de l'inférence, donc torch a pu être posé sur
        #    `sys.path` entre-temps. Pas de problème en pratique — et de toute
        #    façon PyInstaller ne suit pas les imports dynamiques d'un
        #    module déjà exclu.
        "ultralytics",
        "ultralytics.yolo",
        "ultralytics.nn.autobackend",
        "ultralytics.nn.tasks",
        "ultralytics.utils.nms",
        "ultralytics.utils.plotting",
        "ultralytics.engine.predictor",
        "ultralytics.data.augment",
        "ultralytics.cfg",
        # -- Interface.
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        # -- Vidéo / numérique (importés par opencv et ultralytics).
        "cv2",
    ],
    # `hooks-v2/` prime sur `pyinstaller_hooks_contrib/stdhooks/`, comme
    # expliqué dans le docstring : c'est ce qui empêche le hook torch
    # officiel de recollecter 3,8 Go de DLL CUDA.
    hookspath=[str(RACINE / "hooks-v2")],
    hooksconfig={},
    # Même sonde de diagnostic que la v1 (`tools/injection_probe.py`) : sans
    # effet en usage normal, mais c'est elle qui prouve que
    # `config/default.json` est bien lu depuis l'exécutable gelé.
    runtime_hooks=[str(RACINE / "tools" / "injection_probe.py")],
    excludes=[
        # -- Les trois modules torch. Le spec v1, lui, n'en listait aucun et
        #    portait tout le poids. Le retirer de
        #    cette liste ne fait pas que grossir l'exécutable : le hook
        #    officiel recollecte les DLL CUDA même exclus.
        "torch",
        "torchvision",
        "torchaudio",
        # -- Modules Qt JAMAIS importés par ce logiciel, embarqués parce que
        #    PySide6 les déclare comme dépendances de QtWidgets. Mesuré : 38 Mo
        #    de DLL et de `.pyd` qui ne servent à rien ici.
        #      Qt6Quick (6,3) + Qt6Qml (5,1) + Qt6QmlModels (0,9)
        #        + Qt6QmlMeta/QmlWorkerScript (0,3)  -> rendu Qt Quick/QML
        #      Qt6Pdf (4,4)                          -> impression PDF
        #      opengl32sw.dll (19,7) + Qt6OpenGL (1,9) -> rendu logiciel
        #      Qt6Network + QtNetwork.pyd (2,7)       -> réseau
        #      Qt6Svg (0,6) + Qt6VirtualKeyboard (0,4)
        #    Seuls QtCore, QtGui et QtWidgets sont réellement utilisés (voir
        #    `hiddenimports` et `interface/`). Un `QQuickWidget` ou un
        #    `QPdfDocument` ferait réapparaître ces DLL au premier import —
        #    d'où des TESTS plutôt qu'un commentaire.
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQml",
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
        # -- Les paquets que le VENV installe ET que le venv est censé
        #    fournir au premier lancement.
        #
        #    ⚠ `numpy` et `numpy.libs` ont été RÉ-INTRODUITS le 05/10/2026.
        #    Les exclure (commit `b24d8cc`) était le raisonnement « le venv
        #    les réinstallera » — vrai pour l'INFERENCE, faux pour le
        #    DÉMARRAGE. `interface/app.py` fait `import cv2` au chargement du
        #    module, `cv2` fait `import numpy` au sien, et `main.py` atteignait
        #    ce chemin avant que le venv existe. Résultat sur le binaire
        #    gelé : fenêtre ouverte puis fermée aussitôt, traceback
        #    `ModuleNotFoundError: No module named 'numpy'`.
        #
        #    On les réintroduit pour une raison simple : le coût d'un doublon
        #    se mesure en Mo, celui d'un binaire qui ne démarre pas se mesure
        #    en utilisateurs. MESURÉ sur le paquet reconstruit : `_internal`
        #    passe de 254,5 à 280,7 Mo, soit **+26,2 Mo** pour `numpy`
        #    (5,9 Mo) + `numpy.libs` (20,0 Mo) + le `dist-info` (0,2 Mo).
        #    La régression structurelle est verrouillée par
        #    `tests/test_demarrage_sans_venv.py::test_le_spec_embarque_numpy`.
        #
        #    `PIL`/`pillow`/`scipy` restent exclus : rien ne les importe sur le
        #    chemin critique, et ils sont réinstallés par torch dans le venv.
        "PIL",
        "pillow",
        "scipy",
        # -- Ce que la v1 excluait déjà, et pour les mêmes raisons.
        "matplotlib",
        "pandas",
        "notebook",
        "IPython",
        "jupyter",
        "PyQt5",
        "PyQt6",
        "tkinter",
        "polars",
        "pyarrow",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure)

#: La v2 est TOUJOURS en one-dossier, et `COMPTEUR_ONEFILE` n'a pas à être
#: respecté ici. Le plafond de 2 Go du format onefile est une raison de
#: moins de s'en préoccuper maintenant que le paquet est léger — mais l'IGNORER
#: serait une incohérence : le poids est redevenu Petit alors que le chemin
#: `--onefile` reste theoreticalement possible. On garde donc exactement la même
#: structure que la v1, avec le même basculement, pour que les deux versions
#: se comparent correctement.
ONEFILE = os.environ.get("COMPTEUR_ONEFILE", "0") not in ("0", "", "false", "False")

_COMMUN_EXE = dict(
    # Nom différent de la v1 : les deux installations doivent coexister sur
    # le même poste de travail. Un même nom ferait que la v2 écrase la v1 à
    # l'installation — ce qui est explicitement interdit ici.
    name="CompteurManifestationV2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    # `console=True` comme la v1 : une exception au démarrage doit rester
    # VISIBLE. C'est encore plus vrai ici, puisqu'un échec de téléchargement
    # au premier lancement est précisément le cas où il faut pouvoir lire le
    # journal.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        runtime_tmpdir=None,
        **_COMMUN_EXE,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        runtime_tmpdir=None,
        **_COMMUN_EXE,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="CompteurManifestationV2",
    )
    Path(coll.name, "pyi_mode.txt").write_text("onedir\n", encoding="utf-8")