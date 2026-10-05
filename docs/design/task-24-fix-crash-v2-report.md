# Tâche 24 — l'exécutable v2 ne démarrait pas

**Statut : RÉSOLU.** L'exécutable démarre, le chemin d'installation est
atteignable, la release est réuploadée.

| | |
|---|---|
| Commit | `6f56ec7` |
| Tests | **535 passés**, 3 xfailed (521 + 14 nouveaux) |
| Stratégie | **(a) + (b)** — les deux, mesurées |
| Taille | **397,4 Mo** le dossier, **217 Mo** le zip (dont 105,4 Mo de poids `.pt`) |
| Release | `v2.0.0`, asset 226 534 459 octets, réuploadé |

## La preuve qu'il démarre

Pas « le processus a rendu 0 » : un double-clic qui ouvre une fenêtre puis
la ferme passe tous les tests de code de sortie. La preuve est le **titre de
fenêtre**, lu dans l'API Win32 (`EnumWindows` + `GetWindowText`, filtré sur
le PID du processus lancé). C'est `setWindowTitle` qui l'écrit, donc un titre
vu est un titre que Qt a réellement affiché.

**Test 1 — cache vide** (`%LOCALAPPDATA%\CompteurManifestation` absent, le
cas exact du double-clic de l'utilisateur) :

```
2026-10-05 14:50:46 INFO interface.demarrage: torch absent de notre cache — installation CUDA
2026-10-05 14:50:46 INFO compteur.telechargement: uv : ...\_internal\uv.exe venv --python 3.11 ...
2026-10-05 14:50:48 INFO compteur.telechargement: uv : ... pip install ... torch==2.14.1 torchvision==0.29.1 --torch-backend cu126
  +  1.2s FENETRE [576, 305] 'Préparation du moteur de calcul'
```

Les **deux barres** sont là, et `uv` tourne pour de vrai. Après le
téléchargement, `venv\Lib\site-packages` contient bien `torch-2.14.1+cu126`,
`torchvision-0.29.1+cu126` et `numpy-2.4.6` — vérifié sur disque.

**Test 2 — cache déjà rempli** (le second lancement, jamais testé par
l'utilisateur) :

```
[14:56:31] FENETRE PRINCIPALE : 'Compteur de manifestation — 2.0.0'
LES DEUX ECRANS SONT PASSES — chemin complet OK
processus vivant : True
```

Fourni en **7 s**, sans réinstaller quoi que ce soit, et le journal montre
`site-packages du venv posé sur sys.path` puis `interface.app` se charge —
donc `cv2` et `numpy` se résolvent.

## La stratégie : les deux, et pourquoi j'ai mesuré avant de trancher

La note recommandait (a) pour débloquer, (b) ensuite. En mesurant, (b) s'est
révélée presque gratuite, donc les deux.

**Ce que la mesure a donné.** `interface/demarrage.py`, `interface/style.py`
et `compteur/telechargement.py` n'importent **que PySide6 et la stdlib** —
`interface.demarrage` ne touche ni `cv2` ni `numpy`, et son seul
`import torch` est paresseux, sous `try/except`, dans `version_torch()`.
L'écran d'installation n'a donc *aucune raison structurelle* de dépendre du
venv : il ne fait que piloter `uv`, qui est un exécutable externe. Rendre
l'import paresseux est un décalage de trois lignes, pas un chantier.

**Pourquoi (a) reste malgré tout.** Le (b) rend le chemin *possible* ; seul
le (a) le rend *impossible à casser*. Sans `numpy` dans le paquet, il suffit
qu'un jour un `import cv2` remonte d'un cran — un nouveau `from
compteur.compteur import …` dans `app.py`, un `overlay.py` chargé plus tôt —
et l'exécutable recommence à mourir au double-clic, avec 521 tests verts.
`numpy` embarqué, le crash devient structurellement impossible quel que soit
l'état du venv. **Coût réel : +26,2 Mo** (`_internal` 254,5 → 280,7 Mo),
soit 6,6 % du dossier, pour un binaire qui s'ouvre.

`PIL`, `pillow` et `scipy` restent exclus : rien ne les touche sur le chemin
critique, et torch les réinstalle dans le venv.

## Un deuxième bug, trouvé par le test de lancement

Le premier lancement affichait bien l'écran d'installation, mais le
téléchargement échouait en une demi-seconde :

```
ERROR interface.demarrage: installation terminée sur une exception :
  name 'meipass' is not defined
  File "compteur\telechargement.py", line 275, in uv_exe
```

`uv_exe()` référençait `meipass` **sans jamais l'affecter**. Le `_Worker`
attrapait l'exception, `preparer` renvoyait « absent », et l'application
démarrait sans moteur en affichant « moteur de calcul absent » — un
diagnostic **faux**, puisque le réseau marche très bien et que c'est le
logiciel qui n'appelle personne.

Pourquoi aucun test ne l'avait vu : les quatre tests qui touchaient
`installer_torch` court-circuitaient `uv_exe` par `monkeypatch` (justifié,
ils testent la classification d'un échec, pas `uv`). La fonction elle-même
n'avait **aucun** test. Corrigé par `getattr(sys, "_MEIPASS", None)`, plus
deux tests qui l'exécutent pour de vrai.

C'est le genre de bug qui ne se voit qu'en lançant le binaire. Les tests
unitaires ne l'auraient pas trouvé non plus — ils ne l'ont pas trouvé.

## Les tests qui verrouillent la règle

`tests/test_demarrage_sans_venv.py`, 12 tests :

- **statiques** (AST) — `interface.demarrage`, `interface.style`,
  `compteur.telechargement` n'importent aucun paquet du venv, ni `cv2` ; et
  `main.py` importe `interface.app` **après** `preparer()` (on vérifie
  l'ordre des lignes, pas seulement leur présence) ;
- **structurel** — le `.spec` n'exclut plus `numpy` ni `numpy.libs` ;
- **d'exécution** — un sous-processus **vierge** pose une trappe sur
  `sys.meta_path` qui lève sur `cv2`/`numpy`/`torch`, puis construit
  réellement `DialogueInstallation` et vérifie ses libellés. C'est la
  reproduction exacte du traceback du gelé.

**Vérifié non-vide** : réinjecter un `import cv2` dans `interface/demarrage.py`
fait échouer 2 tests. Le test en sous-processus échoue pour la bonne
raison — il ne passe pas par hasard.

Un point de méthode : ma première version de l'analyse AST interdit *tout*
import de paquet venv, y compris dans une fonction. Ça rejetait
`version_torch()`, qui est correct. La règle utile distingue l'import **au
chargement** (tue le processus) de l'import **paresseux**, qui n'a besoin
d'être protégé que par un `try/except`. Le test le dit maintenant
explicitement, sinon le prochain lecteur « corrige » du code sain.

## Un piège trouvé au passage

`tools/copier_modeles.py` ne connaissait que `dist/CompteurManifestation` —
**la v1**, 4,3 Go, explicitement interdite. Sans `--dest`, il y déposait les
poids. Constat : les poids étaient identiques (md5 vérifié) et la v1 fait
toujours 4 357 Mo, donc **aucun dégât** — mais l'accident était possible. La
v2 passe en premier dans la liste.

À noter aussi : l'ancienne release ne contenait que `medium.pt` et `nano.pt`,
pas `yolov8n-head.pt`. Comparer 322 Mo à 397,4 Mo mélangerait donc le
correctif et un modèle de plus. La comparaison honnête est `_internal` :
254,5 → 280,7 Mo.

## Le reste

`dist/CompteurManifestation/` (v1) et `crowd-counter.spec` n'ont pas été
touchés. Le zip est sans séparateur backslash (672 entrées, vérifié, via
`pwsh` 7) et son intégrité est bonne. `build-v2/` n'est pas versionné.

### Ce que je n'ai pas fait

Le gain de taille que la v2 promettait (≈100 Mo) reste hors d'atteinte :
`cv2` 111,8 Mo et `PySide6` 70,2 Mo domine le paquet, et `uv.exe` 49,0 Mo
s'y ajoute. Le plan déjà décrit dans le `.spec` — passer à
`opencv-python-headless`, ~80 Mo gagnés — n'a pas été fait : c'est une
autre tâche, et le faire maintenant aurait mêlé deux changements dans une
release dont le but est de rouvrir le logiciel.
