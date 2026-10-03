# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller : construit un `.exe` Windows autonome, avec CUDA.

Trois décisions méritent d'être lues avant de modifier ce fichier. Toutes
sont dictées par la machine cible : **torch est installé en version CUDA**
(`+cu126`) et c'est ce qui donne le multiplicateur de performance.

**1. one-dossier (`COLLECT`), et `console=True`.** Le logiciel est livré
dans `dist/CompteurManifestation/` : un `.exe` et ses DLL. Ce n'est pas un
choix esthétique — le format onefile de PyInstaller a un **plafond de 2 Go**
(positions sur 32 bits signés) que les DLL CUDA de torch dépassent de deux
fois ; en onefile, le binaire se construit sans erreur et **plante au
lancement**, sans message. Voir le bloc `ONEFILE` plus bas pour la preuve et
le contournement (`COMPTEUR_ONEFILE=1`).

`console=True` n'est pas un oubli : une fenêtre qui se ferme en silence
laisse l'utilisateur devant un double-clic qui ne fait rien. Le traceback
doit rester lisible.

**2. torch CUDA est embarqué, pas filtré.** Le hook officiel
(`pyinstaller_hooks_contrib/stdhooks/hook-torch.py`) collecte
`torch/lib/*.dll` — soit ~3,8 Go de DLL dont `torch_cuda.dll` (1 Go),
`cublasLt64_12.dll` (500 Mo) et l'ensemble cuDNN. C'est la source du poids
de l'exécutable, et c'est voulu : un `.exe` avec torch CPU serait ×4 plus
lent sur les scènes longues, donc inexploitable ici. **Ne pas ajouter torch
aux `excludes`.**

**3. torchvision est une dépendance dure, pas optionnelle.**
`ultralytics.nn.autobackend.warmup()` fait `import torchvision` dès la
première inférence, et `ultralytics.utils.nms` utilise `torchvision::nms`
si le module est chargé. Sans l'extension `_C`, l'inférence plante.

Depuis torchvision 0.29, l'extension ne s'appelle plus `torchvision._C`
mais `torchvision._C_stable`, et elle n'est pas un module Python importable :
`_get_extension_path()` la localise par `FileFinder` sur
`importlib.machinery.EXTENSION_SUFFIXES`, puis la charge avec
`torch.ops.load_library()`. Le hook officiel cherche encore `torchvision._C`.
`hooks/hook-torchvision.py` le remplace donc (le `hookspath` de l'utilisateur
prime sur `pyinstaller_hooks_contrib`) et nomme les vraies extensions.

Construction :

    python -m PyInstaller --clean crowd-counter.spec

Puis copier les poids à côté de l'exécutable (ils sont volontairement hors
du `.exe` : l'utilisateur doit pouvoir remplacer ou ajouter un `.pt`) :

    python tools/copier_modeles.py
"""

import os
import sys
from pathlib import Path

RACINE = Path(SPECPATH)
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

a = Analysis(
    [str(RACINE / "main.py")],
    pathex=[str(RACINE)],
    binaries=[],
    datas=[(str(RACINE / "config" / "default.json"), "config")],
    # `compteur.detecteur` importe torch et ultralytics à l'intérieur de
    # `charger_modele()` : l'analyse statique voit l'appel mais pas toujours
    # le chemin de résolution. On les déclare donc explicitement.
    hiddenimports=[
        # -- Détection : le cœur du logiciel.
        "ultralytics",
        "ultralytics.yolo",
        "ultralytics.nn.autobackend",
        "ultralytics.nn.tasks",
        "ultralytics.utils.nms",
        "ultralytics.utils.plotting",
        "ultralytics.engine.predictor",
        "ultralytics.data.augment",
        "ultralytics.cfg",
        # -- torch + CUDA.
        "torch",
        "torch.nn",
        "torch.utils.data",
        # -- torchvision : l'extension NMS vit dans `ops/boxes.py`, pas dans
        # un module `ops/nms.py`. Voir hooks/hook-torchvision.py pour pourquoi
        # `_C_stable.pyd` doit être embarqué en `binaries`.
        "torchvision",
        "torchvision.ops",
        "torchvision.ops.boxes",
        "torchvision.transforms",
        # -- Interface.
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        # -- Vidéo / numérique (importés par opencv et ultralytics).
        "cv2",
        "scipy",
        "scipy.ndimage",
        "PIL.Image",
        "PIL.ImageDraw",
    ],
    hookspath=[str(RACINE / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    # `matplotlib` est importé par ultralytics.utils.plotting mais seulement
    # dans des fonctions de tracé de courbes, jamais sur le chemin
    # `modifie(img)` -> `boxes`. Vérifié : l'exclusion ne casse ni l'import
    # d'ultralytics ni l'inférence. `pandas` n'est pas installé ici et n'est
    # requis par aucun chemin du logiciel ; torch._dynamo.trace_rules l'appelle
    # par `find_spec()`, qui renvoie None pour un module absent — sans effet.
    excludes=[
        "matplotlib",
        "pandas",
        "notebook",
        "IPython",
        "jupyter",
        "PyQt5",
        "PyQt6",
        "tkinter",
        # `polars` : `_polars_runtime_32.pyd` est précisément l'entrée dont
        # l'extraction échouait en onefile. Elle ne sert à rien ici — c'est
        # une dépendance optionnelle d'ultralytics pour la lecture de
        # datasets, absente de tout chemin d'exécution du logiciel.
        "polars",
        "pyarrow",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

# Les DLL CUDA de torchvision (`cudart64_12.dll`, `nvjpeg64_12.dll`,
# `libwebp.dll`...) et les extensions `_C_stable.pyd` / `image_stable.pyd`
# sont chargées par `torch.ops.load_library()` et par cuDNN, jamais par un
# `import` : PyInstaller ne les voit pas. `hooks/hook-torchvision.py` les
# embarque — ne pas les ajouter ici.
#
# Pourquoi pas ici : `collect_dynamic_libs` renvoie des COUPLES (src, dest)
# alors que `a.binaries` contient des TRIPLETS (src, dest, typecode).
# PyInstaller normalise la sortie d'un HOOK, mais pas ce qu'on ajoute à
# `a.binaries` après coup : mélanger les deux fait échouer `normalize_toc`
# sur « not enough values to unpack (expected 3, got 2) », sans message utile.

pyz = PYZ(a.pure)

#: `True` = un seul `.exe` autonome (dossier temporaire d'extraction au
#: lancement). `False` = un dossier `dist/CompteurManifestation/` contenant
#: l'exécutable et ses DLL.
#:
#: **Pourquoi `False` par défaut — plafond de 2 Go du format onefile.**
#: Le format d'archive CArchive de PyInstaller (utilisé en onefile) écrit les
#: positions des entrées sur **32 bits signés**. Une archive au-delà de 2 Go
#: fait déborder ces positions : le TOC est relu comme `-1454946709` au lieu
#: de `2840020587`, et le bootloader échoue au lancement sur
#:
#:     [PYI-…:ERROR] Failed to extract _polars_runtime_32\_polars_runtime.pyd:
#:     decompression resulted in return code 0!
#:
#: en quittant sur le code `4294967295`, **sans jamais ouvrir la fenêtre**.
#: Ce n'est pas contournable ici : les DLL CUDA de torch pèsent 3,83 Go à
#: elles seules, et les retirer revient à livrer un `.exe` ×4 plus lent sur
#: les scènes longues — voir la section « Performance et GPU » du README.
#:
#: Le mode one-dossier (COLLECT) n'a pas ce plafond : les DLL sont de vrais
#: fichiers sur disque, sans table d'offsets. En prime, le démarrage est
#: **immédiat** au lieu de décompresser 3 Go à chaque lancement.
#:
#: Pour une installation CPU qui tient sous 2 Go, on peut forcer le onefile :
#:     COMPTEUR_ONEFILE=1 python -m PyInstaller --clean crowd-counter.spec
#:
#: Le mode retenu est écrit dans `_internal/pyi_mode.txt` pour que
#: `tools/verifier_lancement_exe.py` puisse rapporter le bon chemin.
ONEFILE = os.environ.get("COMPTEUR_ONEFILE", "0") not in ("0", "", "false", "False")

_COMMUN_EXE = dict(
    name="CompteurManifestation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX sur des DLL signées NVIDIA/cuDNN : long à compresser, et le
    # risque de casser une DLL CUDA ne vaut pas les quelques Mo gagnés.
    upx=False,
    upx_exclude=[],
    # Une exception au démarrage doit rester VISIBLE. `console=False` la
    # jetterait dans un dialogue que l'utilisateur ne voit pas s'il n'a pas
    # le temps de cliquer.
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
        name="CompteurManifestation",
    )
    # Trace du mode pour l'outil de vérification : `sys.frozen` ne dit pas
    # si l'application est en onefile ou en one-dossier. Le fichier est écrit
    # à côté de l'exécutable, pas dans `coll.contents_directory` (`_internal`),
    # pour que l'utilisateur puisse le lire et le supprimer.
    pathlib.Path(coll.name, "pyi_mode.txt").write_text("onedir\n", encoding="utf-8")