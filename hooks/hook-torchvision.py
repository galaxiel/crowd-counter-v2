"""Hook PyInstaller : embarquer l'extension C++ de torchvision.

Le hook officiel (`pyinstaller_hooks_contrib/stdhooks/hook-torchvision.py`)
déclare `hiddenimports = ['torchvision._C']`. Ce module **n'existe plus**
depuis torchvision 0.29 : l'extension a été renommée `_C_stable.pyd` et
n'est pas un module Python importable.

Le chemin réel, dans `torchvision/extension.py` :

    _load_library("_C_stable")
      -> _get_extension_path("_C_stable")   # torchvision/_internally_replaced_utils.py
         -> FileFinder(dossier_torchvision, ExtensionFileLoader).find_spec(...)
      -> torch.ops.load_library(chemin)

Trois conséquences pour l'empaquetage :

1. `_C_stable.pyd` n'est atteint par aucun `import` — PyInstaller ne le voit
   pas. Il faut le nommer en `binaries`.
2. Les DLL chargées par cette extension (`cudart64_12.dll`,
   `nvjpeg64_12.dll`, `libwebp.dll`, `jpeg8.dll`, `libpng16.dll`,
   `libsharpyuv.dll`, `zlib.dll`) ne le sont pas davantage. Le `.spec` les
   ajoute via `collect_dynamic_libs("torchvision")`.
3. `os.add_dll_directory(dossier)` est appelé par torchvision au chargement :
   sous PyInstaller onefile, cela suffit pour trouver les DLL voisines, à
   condition qu'elles soient dans la même collection (`torchvision/`).

Sans ce hook, l'inférence s'arrête sur
`RuntimeError: Couldn't load custom C++ ops` dès la première frame — le
message ne dit rien de la cause, et le même `.exe` marche parfaitement depuis
les sources.
"""

import pathlib

from PyInstaller.utils.hooks import collect_dynamic_libs

#: Extensions C++ réellement livrées. Le nom a changé en 0.29 ; on liste les
#: deux noms pour que le hook reste valable sur les versions antérieures.
EXTENSIONS = ("_C_stable.pyd", "_C.pyd", "image_stable.pyd", "image.pyd")

binaries = collect_dynamic_libs("torchvision")

try:
    import torchvision
except ImportError:  # torchvision absent : le hook n'a rien à faire
    pass
else:
    _dossier = pathlib.Path(torchvision.__file__).parent
    for _nom in EXTENSIONS:
        _chemin = _dossier / _nom
        if _chemin.is_file():
            # Couples (src, dest), comme ceux que renvoie
            # `collect_dynamic_libs` : PyInstaller normalise ensuite en
            # triplets (src, dest, typecode).
            binaries.append((str(_chemin), "torchvision"))

hiddenimports = [
    "torchvision",
    "torchvision.ops",
    # Le NMS est dans `ops/boxes.py` : il n'existe pas de module `ops/nms.py`.
    "torchvision.ops.boxes",
    "torchvision.transforms",
]

# Les sources `.py` de torchvision sont lues au runtime par `torch.jit`
# (sérialisation TorchScript) : sans elles, un `.pt` TorchScript échoue.
module_collection_mode = "pyz+py"