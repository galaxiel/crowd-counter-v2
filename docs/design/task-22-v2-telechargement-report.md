# Version 2 : exécutable sans torch, téléchargé au premier lancement

**Statut : fonctionnel, mesuré, commité.** La v2 démarre, installe torch CUDA
pour de vrai depuis un cache vide, et fait tourner une inférence YOLO sur la
RTX 4070 SUPER. Le décompte n'a pas bougé d'un iota.

---

## 1. Le chiffre qui décide : la taille réelle

| | v1 | v2 | |
|---|---|---|---|
| **`dist/` complet** | **4 349,5 Mo** | **369,7 Mo** | **−91,5 %** |
| `_internal/` (le logiciel) | — | 254,5 Mo | |
| `.exe` seul | — | 9,8 Mo | |
| les 3 `.pt` | 105 Mo | 105 Mo | hors paquet, volontaires |
| `torch` embarqué | ~3 800 Mo | **0** | téléchargé |

**L'objectif de 50 Mo n'est PAS atteint, et il n'était pas atteignable.** La
décomposition, mesurée poste par poste :

| Poste | Poids | Réductible ? |
|---|---|---|
| `cv2` | 111,8 Mo | non — OpenCV lit la vidéo |
| `PySide6` (QtCore/Gui/Widgets + 3 `.pyd`) | 70,2 Mo | non — c'est l'interface |
| racine `_internal` (dont **`uv.exe` 49 Mo**) | 66,5 Mo | non |
| `ultralytics`, `shiboken6`, `yaml`… | 6,1 Mo | non |
| **somme** | **254,5 Mo** | |

Le budget de 50 Mo supposait que le reste du logiciel tiendrait dans 50 Mo une
fois torch retiré. Il en fait 254. **`uv.exe` fait 49 Mo là où l'estimation
annançait 15 Mo** — c'est la seule surprise, et elle annule à elle seule une
bonne partie du gain attendu.

### Ce qui a été fait pour descendre jusqu'ici

- `torch`, `torchvision`, `torchaudio` exclus, **et** neutralisation des hooks
  officiels qui les recollectaient (voir §3) ;
- `hooks-v2/hook-PySide6.py` : **40 Mo de Qt inutile retirés** (`Qt6Quick`,
  `Qt6Qml`, `Qt6Pdf`, `Qt6Network`, `opengl32sw.dll`…). `excludes` ne suffisait
  pas — le hook officiel fait `collect_dynamic_libs("PySide6")` sans consulter
  la liste ;
- `numpy`, `numpy.libs`, `PIL`, `scipy` exclus : **41 Mo**, ces paquets étant
  réinstallés par `uv` dans le venv (vérifié : ils y sont).

### Ce qui n'a PAS été fait, et pourquoi

**`cv2` (112 Mo) et `PySide6` (70 Mo) ne sont PAS des doublons.** Vérifié dans
le venv réellement construit : il contient `numpy`, `PIL`, `torch`,
`torchvision` — **ni `cv2`, ni `PySide6`**. L'hypothèse « le venv installe aussi
ça » est fausse. Les embarquer est donc nécessaire, pas redondant : les
exclure casserait le démarrage, puisque `main.py` importe PySide6 avant même
que le venv soit interrogeable.

Pour descendre sous 100 Mo il faudrait remplacer Qt, ou passer à
`opencv-python-headless` (~80 Mo gagnés, faisable car le logiciel n'affiche
rien d'OpenCV). **Non fait ici** : cela sort du cadre de la tâche et risque la
régression visuelle.

---

## 2. Ce qui a été VÉRIFIÉ

### Téléchargement réel depuis zéro — **OUI, fait**

Le cache `%LOCALAPPDATA%\CompteurManifestation` a été **vidé**, puis
l'installation lancée. Ce n'était pas faisable au début (torch installé
globalement) ; c'est devenu faisable grâce au garde-fou (§4).

```
cache vidé  →  uv venv --python 3.11        (Python autonome)
            →  Downloaded numpy, pillow, sympy, torchvision, networkx
            →  Downloaded torch             2,4 Go à 66,6 Mo/s
            →  Prepared N packages          ← bascule en phase 2
            →  Installed N packages
résultat : True en 94 s
marqueur : True      torch importable : True
```

Les **deux phases** se déclenchent bien, et le cache est nettoyé ensuite
(`uv cache clean` : 16 457 fichiers, **4,0 Go libérés**).

### Inférence réelle dans l'exécutable gelé — **OUI, fait**

Le binaire lui-même, pas les sources :

```
site-packages du venv posé sur sys.path
stdlib du venv ajoutée en fin de sys.path
33 module(s) stdlib purgé(s) pour re-routage vers le venv
finder stdlib venv installé sur 7 paquets
périphérique de détection : cuda:0      ← VRAI GPU
SELFTEST inference OK 1                  ← YOLO nano, 1 résultat
```

Indicateur affiché : `CUDA 2.14.1+cu126 — NVIDIA GeForce RTX 4070 SUPER`.
La v1 lance toujours, et son empreinte est **inchangée au bit**
(3 388 fichiers, `d99c9e58cea87a5ca79361e35bde0ccc`).

### Tests — **521 passés, 0 échec** (481 avant, +40)

```
521 passed, 3 xfailed in 8.72s
```

---

## 3. Ce qui n'a PAS été vérifié — à dire franchement

- **Le test « téléchargement réel depuis zéro » n'a pas été rejoué sur un PC
  vierge au sens strict** : la machine a un CUDA déjà installé ailleurs, et
  `uv` a réutilisé `C:\Python311` comme base du venv (`pyvenv.cfg: home =
  C:\Python311`) au lieu de télécharger son Python autonome. Sur un vrai PC
  vierge, `uv` téléchargerait Python et le chemin de stdlib différerait —
  **c'est le point le plus fragile qui reste**.
- **Le mode « Continuer sur CPU » n'a pas été exercé de bout en bout** avec un
  vrai téléchargement CUDA interrompu au milieu. Le mécanisme de bascule est
  testé unitairement, la bascule réelle pendant 2,4 Go ne l'est pas.
- **Le repli CPU après échec n'a pas été vu à l'écran** : le test vérifie
  l'état et le texte, pas le rendu de la fenêtre.
- **Pas de test sur une session RDP ni sur une machine sans OpenGL** — or
  `opengl32sw.dll` (19,7 Mo) a été retiré par le hook PySide6. Une fenêtre qui
  ne s'affiche pas sur un vieux portable est le risque résiduel principal de
  cette version ; le remonter est **une ligne** dans
  `hooks-v2/hook-PySide6.py` (`DLL_GARDEES`).
- **Le nettoyage du cache n'a pas été testé en échec** (cache verrouillé sous
  Windows) : le code journalise un avertissement et continue, ce qui est le
  comportement voulu, mais ce chemin n'est pas exercé.

---

## 4. Le garde-fou : torch vient de NOTRE dossier, et de nulle part ailleurs

C'est ce qui rend la v2 testable — et c'est ce qui la fait fonctionner ou non
chez un vrai utilisateur.

- `chemin_python()` renvoie le `python.exe` **du venv**, jamais
  `sys.executable` (dans un `.exe` gelé, c'est notre propre binaire : le
  lancer en sous-processus relancerait l'application) ;
- `_verifier_import()` lance le Python du venv avec `-I` (mode isolé) et exige
  que `torch.__file__` soit **sous notre dossier** — code de sortie 3 sinon ;
- `ajouter_au_sys_path()` ne pose **que** `venv\Lib\site-packages` ;
- `torch_installe()` exige cette preuve, au lieu de supposer qu'un dossier
  présent suffit.

### Le piège que le garde-fou a révélé (et qu'il a fallu corriger trois fois)

Dans l'exécutable gelé, **`sys.path` ne suffit pas** :

1. `import torch` échouait sur `No module named 'timeit'` — la stdlib est
   servie depuis la `base_library.zip` embarquée, jamais par le `sys.path` ;
2. `torchvision` échouait ensuite sur `No module named 'html.parser'` ;
3. puis sur `No module named 'xml.etree'`.

Ajouter la stdlib au `sys.path` : sans effet. Un finder en tête de `meta_path` :
sans effet non plus, parce que `xml` est **déjà dans `sys.modules`** quand le
finder s'installe. **Ce qui marche : purger `sys.modules` des paquets stdlib
concernés (33 modules), puis installer le finder** — d'où les lignes de log
`33 module(s) stdlib purgé(s)`. `logging` est volontairement exclu de la liste :
il a un état global réel.

---

## 5. Les deux barres de progression

Deux phases **visuellement distinctes**, comme demandé :

- **Phase 1 — Téléchargement** : octets, pourcentage, **vitesse (Mo/s)** et
  **temps restant**. Détail de l'interface : sans la vitesse, « 1,2 Go / 2,4 Go
  — 50 % » ne distingue pas 2 Mo/s de 200 Mo/s, qui demandent des décisions
  opposées ;
- **Phase 2 — Installation** : **une seconde barre qui repart de zéro**, pour
  la décompression. Elle compte des **paquets**, et l'affiche comme tel : `uv`
  ne communique pas les octets qu'il écrit, et un pourcentage d'octets
  inventé serait une barre qui ment.

Le bouton **« Continuer sur CPU (120 Mo, plus lent) »** est disponible pendant
toute la phase 1 et bascule la variante **en cours de téléchargement** (testé).

Un défaut trouvé par l'observation réelle et corrigé : la barre **reculait**
(50 % → 0 %) quand `uv` annonçait torch en dernier, et affichait **100 %
prématurément** quand seul numpy était connu. Cause : `uv` annonce les paquets
au fur et à mesure. Correctif : le dénominateur part de la taille **mesurée**
des roues (`TAILLES`), le pourcentage ne peut plus mentir ni reculer.

---

## 6. Le cache : emplacement et taille

- **Emplacement** : `%LOCALAPPDATA%\CompteurManifestation\`. **Pas de
  `C:\Program Files`, donc pas de droits administrateur et pas d'UAC** au
  premier lancement — un logiciel de terrain qui demande d'être élevé perd la
  moitié de ses utilisateurs. Test verrouillé.
- **Après installation CUDA** : `venv` 4,1 Go. Le cache `uv` (4,1 Go, les
  mêmes octets en double) est **nettoyé automatiquement** → **≈ 4,1 Go** au
  total.
- Ce n'est pas un temporaire : un téléchargement interrompu doit survivre à un
  redémarrage.

---

## 7. Fichiers

| Fichier | Rôle |
|---|---|
| `crowd-counter-v2.spec` | build sans torch, embarque `uv.exe`, `numpy`/`PIL` exclus |
| `hooks-v2/hook-torch.py`, `hook-torchvision.py` | neutralisent les hooks officiels |
| `hooks-v2/hook-PySide6.py` | ne garde que Qt6Core/Gui/Widgets (−40 Mo) |
| `compteur/telechargement.py` | installation via `uv`, garde-fou, deux phases, nettoyage |
| `interface/demarrage.py` | fenêtre à deux barres, repli CPU, démarrage garanti |
| `main.py` | pose le `sys.path` du venv avant tout `import torch` |
| `tools/verifier_installation_v2.py` | installe / lance pour vérifier, hors tests |
| `tests/test_telechargement.py`, `test_demarrage.py` | +40 tests |

**Intacts : `crowd-counter.spec`, `hooks/`, et `dist/CompteurManifestation/`
(v1) au bit près.** Le cœur de comptage (`compteur/compteur.py`, `tracker.py`,
`ligne.py`) n'a **aucune modification** : le décompte de référence
(257 personnes sur 3 000 frames, `manif_test.mp4`) est inchangé par
construction.

---

## 8. La suite

1. **Rejouer sur un PC vierge** (sans CUDA, sans Python) — c'est le seul test
   qui manque vraiment, et `uv` y téléchargerait son Python autonome : le
   chemin de stdlib serait différent de celui testé ici.
2. **Exercer le repli CPU** pendant un vrai téléchargement CUDA.
3. Décider si on accepte les 254 Mo, ou si on passe à
   `opencv-python-headless` (−80 Mo) pour viser ~175 Mo.