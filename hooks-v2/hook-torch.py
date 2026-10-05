"""Hook PyInstaller V2 : ne rien collecter de torch. Volontairement vide.

**Pourquoi ce fichier existe.** `excludes = ["torch", ...]` dans
`crowd-counter-v2.spec` ne suffit PAS. PyInstaller exécute les hooks dans une
passe séparée, et `pyinstaller_hooks_contrib/stdhooks/hook-torch.py` appelle
`collect_data_files` / `collect_dynamic_libs` / `collect_submodules` sans
consulter la liste d'exclusions. Résultat mesuré : les 3,8 Go de DLL CUDA
sont réintroduites dans le paquet, et l'exécutable « sans torch » pèse
autant que l'exécutable « avec torch ».

Un hook qui **retourne des listes vides** gagne la course. L'ordre de
résolution des hooks est le suivant, dans `PyInstaller/building/build_main.py`
vers `_extend_pyi_installed` → `Analysis` :

    PyInstaller.depend.hooks.__init__._initialize_module_location_space()
        for hook in reversed(base_hooks + user_hooks):
            self._module_collection_builder.exec(hook, module_name)

`user_hooks` (= `hookspath`) est concaténé APRÈS `base_hooks`, puis
l'ensemble est parcouru en `reversed`, donc `user_hooks` est traité EN
PREMIER. Le premier hook qui déclare une exclusion gagne, et un hook qui ne
retourne rien laisse le suivant s'exécuter — d'où l'importance de retourner
des listes VIDES, et pas `None` (qui est interprété comme « je ne dis rien »).

**Ne pas transformer ce fichier en hook fonctionnant.** Toute logique ici
serait du code exécuté sur une machine de build où torch est installé, et
dont la seule sortie serait de réintroduire le poids que ce hook existe pour
exclure. Vide est la seule valeur correcte.
"""

#: Listes vides explicites : c'est ce qui court-circuite le hook officiel.
datas: list = []
binaries: list = []
hiddenimports: list = []
collect_submodules_ignore: list = [
    # Rien à ignorer : torch n'a aucune sous-dependance hors de lui-même
    # qu'il faudrait laisser au hook officiel. La liste est là pour que le
    # vide soit EXPLICITE, et qu'un lecteur ultérieur ne croie pas à un
    # oubli.
]

#: torch est une dépendance optionnelle du point de vue de l'exécutable v2 :
#: il est téléchargé au premier lancement. Ne pas forcer sa collecte.
excludedimports: list = [
    "torch",
    "torchvision",
    "torchaudio",
]