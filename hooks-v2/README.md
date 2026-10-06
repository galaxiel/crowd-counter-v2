# Hooks PyInstaller de la version 2

Ce répertoire est utilisé exclusivement par `crowd-counter-v2.spec` (via
`hookspath`). Depuis le retrait de la v1 (06/10/2026), c'est le seul jeu de
hooks du projet : l'ancien répertoire `hooks/` et son `crowd-counter.spec`
sont dans l'historique git, et `HISTOIRE.md` raconte pourquoi la v1 avait
les siens (elle embarquait torch ; la v2 l'exclut du build et le télécharge
au premier lancement).

## Pourquoi des hooks qui ne font rien

`excludes = ["torch", "torchvision", "torchaudio"]` ne suffit pas. Les
exclusions ne sont appliquées qu'à l'analyse statique des imports ; les
hooks sont exécutés dans une passe séparée et n'en tiennent pas compte.
`pyinstaller_hooks_contrib/stdhooks/hook-torch.py` appelle
`collect_dynamic_libs("torch")`, qui recollecte ~3,8 Go de DLL CUDA.

Un hook qui retourne des **listes vides** gagne la course. L'ordre de
traitement des hooks est `reversed(base_hooks + user_hooks)`, et `user_hooks`
(`hookspath`) est concaténé en dernier : il est donc traité en premier. Voir
le docstring de `hook-torch.py` pour la référence exacte.

## Ne rien ajouter au `hookspath`

Le `hookpath` de la v2 est `hooks-v2` **seul**. Y ajouter tout autre
répertoire contenant un `hook-torch.py` ou `hook-torchvision.py` « normal »
ferait un `import torchvision` et un `collect_dynamic_libs` au build — donc
~600 Mo de DLL CUDA remises, et un build cassé si l'extension change de nom.

## Vérifier après un build

    python -c "import os;print(sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk('dist/CompteurManifestationV2') for f in fs)/2**20, 'Mo')"

Si le chiffre dépasse ~120 Mo, le hook officiel a gagné : `hook-torch.py` ou
`hook-torchvision.py` doit être absent de `hooks-v2/`.