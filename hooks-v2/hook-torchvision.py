"""Hook PyInstaller V2 : ne rien collecter de torchvision non plus.

**Pourquoi un fichier séparé de `hook-torch.py`.** Le hook officiel
`pyinstaller_hooks_contrib/stdhooks/hook-torchvision.py` déclare
`hiddenimports = ['torchvision._C']` et appelle `collect_dynamic_libs` /
`collect_all`. Le hook maison `hooks/hook-torchvision.py`, lui, va chercher
`_C_stable.pyd` et les DLL voisines — ce qui est exactement ce qu'il ne faut
PAS faire en v2, puisque torchvision sera lui aussi téléchargé.

Sans ce fichier, la seule chose qui nous sauve est que `hook-torch.py`
retourne des listes vides : `excludedimports` n'arrête pas les hooks, il
n'empêche que la *collecte de modules*. Un `hook-torchvision.py` vide dans le
même répertoire ferme la dernière porte, pour la même raison et par le même
mécanisme (ordre `reversed(base_hooks + user_hooks)`, voir
`hooks-v2/hook-torch.py`).

**Attention à la tentation de supprimer `hooks/` du `hookspath`.** Le
`hookspath` de la v2 est `hooks-v2` SEUL, pas `["hooks", "hooks-v2"]` :
`hooks/hook-torchvision.py` y ferait `import torchvision` et
`collect_dynamic_libs("torchvision")` au build,.remettant ~600 Mo de DLL CUDA
et, si l'extension change de nom, plantant le build sur un `ImportError`. Les
deux répertoires ne doivent jamais être activés ensemble.
"""

datas: list = []
binaries: list = []
hiddenimports: list = []
excludedimports: list = [
    "torchvision",
    "torchvision.ops",
    "torchvision.transforms",
]