# Compteur de manifestation

Compte les personnes franchissant une ligne virtuelle dans une vidéo de
caméra fixe. Moteur de comptage sans interface (`compteur/`), fenêtre PySide6
par-dessus (`interface/`).

Une scène, une ligne, un chiffre : combien de personnes ont traversé, dans
quel sens, à quelle cadence. Rien de plus — pas de reconnaissance faciale,
pas d'identification, pas de densimétrie.

---

## Sommaire

- [Ce que fait le logiciel](#ce-que-fait-le-software)
- [Lancer sans rien installer](#lancer-sans-rien-installer)
- [Installer et lancer depuis les sources](#installer-et-lancer-depuis-les-sources)
- [Où placer les modèles](#où-placer-les-modèles)
- [Performance et GPU](#performance-et-gpu)
  - [Choisir où faire le calcul](#choisir-où-faire-le-calcul)
- [Construire l'exécutable](#construire-lexécutable)
- [Utiliser le logiciel](#utiliser-le-logiciel)
- [Exporter](#exporter)
- [Limites connues](#limites-connues)
- [Architecture](#architecture)
- [Tests](#tests)

---

## Ce que fait le logiciel

1. **Détection** — YOLO (Ultralytics) repère des têtes sur chaque frame. Le
   modèle est un paramètre, jamais codé en dur.
2. **Suivi** — association par IoU : chaque tête garde un identifiant stable
   d'une frame à l'autre.
3. **Comptage** — quand un identifiant passe d'un côté de la ligne à
   l'autre, dans le sens choisi, c'est un comptage.
4. **Restitution** — un gros chiffre à l'écran, un CSV et un JSON à
   l'export.

Le modèle par défaut (`medium.pt`) est un détecteur de têtes entraîné sur
SCUT-HEAD. Sur la vidéo de référence il trouve des têtes de 29 px là où
`yolov8n-head.pt` en trouve de 21, soit 38 % de marches réelles en plus.

## Lancer sans rien installer

`dist/CompteurManifestation/` est autonome : Python n'a pas besoin d'être
installé sur la machine.

1. Copier le dossier `dist/CompteurManifestation/` **entier** à l'endroit
   voulu (une clé USB, un bureau, `C:\Programmes\`) — le `.exe` seul ne
   suffit pas, il a besoin de ses DLL voisines.
2. Double-cliquer sur `CompteurManifestation.exe`.

Une fenêtre de console noire s'ouvre à côté de l'application : c'est
voulu. Elle affiche la journalisation et, en cas de problème, le traceback.
Fermer le logiciel ferme aussi la console.

Le démarrage est **immédiat** : rien n'est décompressé, tout est déjà sur
disque.

### Pourquoi un dossier et pas un seul `.exe`

PyInstaller sait tout regrouper dans un fichier unique (« onefile »).
Avec torch CUDA, ce mode **ne fonctionne pas** : le format d'archive
utilise des positions sur 32 bits signés, et les DLL CUDA seules pèsent
3,8 Go. Le binaire se construit sans erreur puis **plante au lancement**, en
quittant sans message. Le détail est dans `crowd-counter.spec` ; le
contournement (pour une installation CPU qui tient sous 2 Go) est
`COMPTEUR_ONEFILE=1`.

Le mode dossier a un avantage direct sur ce logiciel : pas de
décompression de 3 Go à chaque lancement.

### Ce qu'il faut mettre à côté de l'exécutable

```
dist/CompteurManifestation/
├── CompteurManifestation.exe
├── _internal/            <- DLL et modules Python, ne pas y toucher
├── medium.pt             <- modèle par défaut
├── nano.pt               <- modèle léger (CPU ou vieille machine)
└── yolov8n-head.pt      <- détecteur de têtes Ultralytics standard
```

Les poids ne sont **pas** dans l'exécutable : ce sont des paramètres, pas
des dépendances. Voir [Où placer les modèles](#où-placer-les-modèles).

## Installer et lancer depuis les sources

Python 3.11 ou supérieur (testé sur 3.14).

```bash
python -m venv venv
venv\Scripts\activate
python installer.py
python main.py
```

`installer.py` détecte la carte graphique et installe la version de torch qui
va avec. **N'utilisez pas `pip install -r requirements.txt`** pour cette étape :
torch se publie en deux familles de Wheels, CPU et CUDA, et le paquet nommé
`torch` sur PyPI ne dit pas laquelle on obtient — c'est l'installation qui a
l'air correcte et qui tourne à 7 img/s.

Le script se vérifie sans rien installer :

```bash
python installer.py --dry-run                    # affiche la branche retenue
python installer.py --dry-run --sans-gpu         # vérifie la branche CPU
python installer.py --dry-run --peripherique cuda # échoue si pas de carte
```

`--dry-run` n'installe rien : c'est le moyen de vérifier ce que le script
comprit de votre machine avant de lui laisser toucher à `pip`.

Pour travailler sur le logiciel lui-même (tests, build) :

```bash
python -m pip install -r requirements-dev.txt
```

## Où placer les modèles

Le panneau de réglages liste les fichiers `.pt` qu'il trouve dans deux
dossiers, dans cet ordre :

1. le dossier courant ;
2. `modeles/`.

Conséquence pratique :

| Où je lance | Où je mets les `.pt` |
|---|---|
| `python main.py` depuis la racine | à la racine du dépôt |
| `dist\CompteurManifestation\CompteurManifestation.exe` | dans `dist\CompteurManifestation\` |

Un double-clic sur l'exécutable donne pour dossier courant le dossier de
l'exécutable : les poids déposés dans `dist\CompteurManifestation\`
apparaissent donc dans la liste déroulante sans configuration.

Le champ reste **éditable** : on peut taper un chemin absolu vers un `.pt`
qui n'est dans aucun de ces dossiers.

Après une reconstruction :

```bash
python tools/copier_modeles.py
```

recopie les poids de la racine vers `dist/`. `--propre` ne recopie que ce qui
manque.

## Performance et GPU

**torch doit être installé en version CUDA.** Sur cette machine :

```
torch 2.14.1+cu126
torchvision 0.29.1+cu126
```

C'est ce que fait `installer.py` quand il détecte une carte NVIDIA.

### Combien ça change

Mesuré sur cette machine, sur la vidéo de référence :

| Périphérique | Débit | Vidéo de 4 minutes |
|---|---|---|
| CUDA (RTX 4070 SUPER) | ~27 img/s | ~9 min |
| CPU | ~7 img/s | ~35 min |

Un torch CPU fonctionne, mais le compteur devient inutilisable en direct, et
les scènes longues demanderont des heures. L'exécutable embarque CUDA — d'où
ses 3 Go. `torch_cuda.dll` seule pèse 1 Go, `cublasLt` 500 Mo, cuDNN
plus de 1 Go.

### Choisir où faire le calcul

Le réglage *Calcul sur*, dans le panneau *Périphérique de calcul* :

| Choix | Effet |
|---|---|
| **Automatique** (défaut) | CUDA si la machine en a un, CPU sinon |
| **GPU NVIDIA (CUDA)** | Le GPU est exigé. Sans carte, l'analyse **bascule sur le CPU avec un message** — un plantage sur le terrain coûte plus cher qu'une analyse lente qu'on peut au moins regarder |
| **CPU uniquement** | Le processeur est exigé même avec un GPU. C'est le remède quand le GPU plante sur une scène particulière |

Ce choix ne change **pas** le décompte : la même scène est analysée dans les
deux cas.

L'indicateur *Calcul : CUDA — NVIDIA GeForce RTX 4070 SUPER* ou *Calcul :
CPU uniquement* est affiché **en permanence** dans la barre de statut, sous
le compteur. Il est mis à jour dès qu'un réglage change, et il dit ce qui est
réellement utilisé — pas ce qui était demandé.

Sous PyInstaller, `nvidia-smi` n'interroge que le pilote : il ignore
`CUDA_VISIBLE_DEVICES`, qui est une convention Linux. Pour vérifier la branche
CPU sur une machine qui a une carte, l'installateur expose `--sans-gpu`.

### AMD n'est pas supporté

Choix assumé : **un seul exécutable, buildé CUDA**. Un wheel torch CPU-only
pèse ~200 Mo contre ~3 Go pour le CUDA ; livrer les deux ferait doubler la
taille du dossier `dist/` pour un cas d'usage qui n'est pas le nôtre (une
carte AMD en 2026 est rare sur le parc de manifestation visé). Le programme
tourne donc sur AMD en mode CPU, à ~7 img/s — utilisable pour une courte
vidéo, pas pour une manifestation entière.

### Si le GPU n'est pas utilisé

L'application le signale dans la barre de statut et dans la console, et
analyse quand même en CPU, très lentement.

Sous PyInstaller, deux pièges ont été traités explicitement (voir
`crowd-counter.spec` et `hooks/hook-torchvision.py`) :

- l'extension NMS de torchvision s'appelle `_C_stable` depuis la 0.29 et
  n'est pas un module Python : sans hook dédié, l'inférence s'arrête sur
  `Couldn't load custom C++ ops` ;
- les DLL CUDA sont chargées par `torch.ops.load_library()`, jamais par un
  `import` : PyInstaller ne les voit pas tout seul.

## Construire l'exécutable

```bash
python -m pip install -r requirements-dev.txt
python -m PyInstaller --clean crowd-counter.spec
python tools/copier_modeles.py
```

Résultat : `dist/CompteurManifestation/` — l'exécutable, son dossier
`_internal/` et les poids à côté.

La construction prend **6 à 10 minutes** : PyInstaller analyse torch, qui
contient à lui seul quelques milliers de modules, puis recopie 4 Go de DLL
sur disque. Un `upx=True` rendrait la compression interminable sur des DLL
signées NVIDIA ; le `.spec` le désactive.

`--clean` vide le cache entre deux builds. Sans lui, le second build est
nettement plus rapide mais peut réutiliser un graphe de dépendances
périmé — après un changement de version de torch, utilisez `--clean`.

### Vérifier que ça se lance

```bash
python tools/verifier_lancement_exe.py
```

Lance l'exécutable comme un utilisateur le ferait, vérifie qu'une fenêtre
titrée *Compteur de manifestation* apparaît, que `medium.pt` est à côté, et
que la sortie console ne contient pas le repli silencieux de `Config`. Un
build qui réussit ne prouve pas qu'un double-clic fonctionne — c'est
exactement ce que cet outil couvre.

## Utiliser le logiciel

1. **Charger la vidéo** — bouton *Charger la vidéo*.
2. **Tracer la ligne** — bouton *Tracer la ligne*, puis deux clics sur
   l'image, en haut et en bas de la ligne qui vous intéresse. Le tracé est
   « armé » : un clic parasite hors de ce geste est ignoré.
3. **Régler le sens** — dans *Ligne de franchissement*, choisissez la
   direction comptée. La flèche verte indique le sens actif.
4. **Ajuster la sensibilité** — le paramètre qui compte le plus est *Frames
   de confirmation* : 1 compte tout de suite (sensible aux faux positifs),
   5 ne valide qu'une personne vue plusieurs frames de suite.
5. **Lancer** — le compteur s'actualise en continu, la ligne clignote à
   chaque passage.
6. **Exporter** — voir ci-dessous.

### Mode audit

Pendant une relecture, le panneau *Audit* permet de signaler les erreurs à
l'œil. Le taux d'erreur affiché ne vaut que pour la portion que vous avez
effectivement vérifiée : c'est une **mesure**, pas une estimation
automatique.

## Exporter

*Exporter* écrit dans le dossier choisi :

- un **CSV** : un événement par ligne (frame, timestamp, identifiant suivi,
  position, score) ;
- un **JSON** : total, statistiques de débit, configuration complète.

Le dossier proposé par défaut est `sortie/`, relatif au dossier courant —
c'est-à-dire `dist\sortie\` si on lance l'exécutable.

## Limites connues

**Le comptage rend 0 sur certaines scènes.** C'est la limite la plus
importante, et elle ne se devine pas à l'écran : le logiciel s'ouvre
normalement, la vidéo défile, les boîtes de détection s'affichent, et le
chiffre reste à zéro. Ce n'est pas un plantage, c'est un réglage — ou une
limite géométrique de l'algorithme.

Ce qui rend le passage detectable est mesuré sur la vidéo de référence : une
tête y fait 20 à 29 px et bouge de **0,69 px par frame**. Le déplacement
réel d'une frame à l'autre est donc du même ordre que le bruit du
détecteur — le déplacement *médian* mesuré entre deux frames consécutives
est de 0,00 px. À cette échelle, deux personnes distinctes sont aussi
proches l'une de l'autre que chacune ne l'est de sa propre position à la
frame d'avant.

Trois réglages font tomber le compteur à zéro, et le symptôme est le même
pour les trois :

- **Mouvement trop faible + seuil de confirmation trop haut.** *Frames de
  confirmation* vaut 3 par défaut : une track n'apparaît en sortie qu'après
  trois frames. Sur un flux où le mouvement par frame est inférieur au bruit
  du détecteur, l'identifiant peut être perdu avant d'atteindre ce seuil, ou
  les trois frames peuvent se passer loin de la ligne. **Remède : *Frames de
  confirmation* à 1**, et/ou réduire *Épaisseur de la bande* pour que le
  franchissement tienne dans moins de frames. Les deux augmentent la
  sensibilité — au prix des faux positifs.
- **Le lissage retarde le franchissement jusqu'à l'annuler.** *Fenêtre de
  lissage* est à 1 (mode brut) par défaut, et c'est délibéré : le test
  `a_traverse` porte alors sur la position instantanée, ce qui est la seule
  position qui « bouge » à cette échelle. Au-delà de K=1, le lissage retarde
  la détection du croisement d'une demi-fenetre, pendant que le verrou
  anti-rebond lit toujours le côté de la position **brute** — déjà passée de
  l'autre côté. Le verrou rejette alors, et la marche de référence passe de
  **1 comptage à 0**. **Remède : laisser à 1.** Ce défaut est connu et
  documenté dans `compteur/config.py` ; le corriger suppose de faire lire
  `_cotes`/`_stabilite` à la position lissée, pas seulement le test de
  franchissement.
- **Ligne mal placée.** `a_traverse` interpole le point exact de croisement
  et vérifie qu'il tombe **le long du segment** dessiné. Une ligne tracée en
  travers de la rue compte les gens qui passent sous le milieu du cadre ;
  une ligne qui s'arrête trop haut ou trop bas ignore les passages hors de
  son segment. Rallonge la ligne sur toute la hauteur utile de l'image.

Autres limites :

- **Le taux d'erreur n'est jamais calculé automatiquement.** Il n'existe pas
  de vérité terrain sans annotation manuelle. Le mode audit mesure l'écart
  sur ce que vous vérifiez, et rien d'autre.
- **Foules compactes hors cas d'usage.** Le compteur vise un flux peu
  occlusé (rue, passage espacé). Quand les corps se chevauchent, les têtes se
  fusionnent en une seule boîte et une personne peut en masquer deux.
- **Vue de dessus uniquement.** Une caméra en hauteur, cadrée large, sans
  contre-jour ni pluie. Le modèle est un détecteur de têtes : il compte des
  têtes, pas des personnes entières.
- **Sens unique par analyse.** Compter « dans les deux sens » demande deux
  passes ou deux fenêtres.
- **Pas de multidécaméra, pas d'API web.** Le moteur est prêt pour l'API
  (`analyser_video` est une fonction `vidéo + config -> résultat`) ; la
  décision est reportée.

## Architecture

```
compteur/    moteur pur, sans dépendance graphique — testable sans écran
interface/   fenêtre PySide6
config/      default.json : les valeurs par défaut
installer.py détection du GPU + installation de torch (CUDA ou CPU)
tools/       scripts en ligne de commande
tests/       pytest
hooks/       hooks PyInstaller (torchvision)
```

Le point clé : `compteur/compteur.py::Compteur` est le moteur. L'interface
l'appelle, les tests l'appellent, et une éventuelle API web pourra l'appeler
plus tard sans rien réécrire. La contrainte est tenue par un test
(`tests/test_isolation_ui.py`) : aucun module de `compteur/` n'importe
PySide6, Tkinter ou quoi que ce soit de graphique.

`config/default.json` est la source unique des valeurs par défaut — le
dataclass `Config` ne fait que le refléter. Un test compare les deux pour
qu'ils ne divergent jamais. Sous PyInstaller, ce fichier est embarqué et
retrouvé via `sys._MEIPASS` (`tests/test_config_gelee.py`).

## Tests

```bash
python -m pytest tests/ -v
```

## Licence

À définir par l'auteur.