# Histoire du projet

*Pourquoi ce logiciel est fait comme il est. À lire avant de modifier le code.*

Ce document n'est pas une documentation technique — le `README.md` couvre ça.
C'est le récit des décisions : ce qu'on a essayé, ce qui a marché, ce qui a
marché **par erreur**, et pourquoi le logiciel ressemble à ce qu'il est.

Il est écrit pour être lu dans trois ans, par quelqu'un qui a oublié les
détails. Si une phrase te paraît évidente maintenant, elle ne le sera peut-être
plus dans trois ans — c'est pour ça qu'elle est écrite quand même.

---

## Le point de départ

Une caméra fixe, en hauteur, filme une rue. Des cortèges défilent devant,
espacés. La question : **combien de personnes sont passées ?**

Trois idées reçues ont été écartées d'emblée :

**Pas la reconnaissance faciale.** Compter des personnes, oui ; savoir qui
elles sont, non. Au-delà de l'éthique, ça change le profil légal du projet de
façon radicale. On ne discutera pas ce point.

**Pas la détection de « personnes ».** C'était l'idée de départ. Elle marche mal
sur ce type de vidéo : les banderoles cassent la détection, et deux personnes
côte à côte fusionnent en une seule boîte.

**Pas la densimétrie.** C'est le contre-exemple naturel. Elle convient à une
photo d'une foule sur un plan large ; pas à un cortège qui défile devant une
caméra. La densité ne dit rien du nombre de personnes qui *traversent une
ligne*. La distinction est intuitive mais il faut l'avoir comprise : deux
problèmes qui se ressemblent ne se résolvent pas avec la même méthode.

---

## Le modèle : détecter des têtes

Un détecteur de têtes détecte ce qui reste visible quand tout le reste est
couvert. Au-dessus d'une banderole, d'un panneau, d'un autre manifestant, une
tête reste là où un corps disparaît.

C'est pourquoi le modèle par défaut est un détecteur de têtes entraîné sur
**SCUT-HEAD**, et non un détecteur de personnes.

**Ce qui a été mesuré**, sur la vidéo de référence :

| modèle | têtes détectées | taille |
|---|---|---|
| `yolov8n-head.pt` | 21 px | la plus rapide |
| `nano.pt` | 28 px | score plus bas |
| **`medium.pt`** | **29 px** | **le défaut — 38 % de marches en plus que le petit** |

La « vitesse » du modèle est mesurée en px de tête détectée, pas en image par
seconde. Un modèle plus rapide qui ne voit que des petites têtes est plus rapide
pour rien.

---

## Le tracker : l'algorithme « moderne » rejeté

**ByteTrack a été essayé en premier.** C'est le tracker standard de l'écosystème
YOLO, et c'était le choix évident.

Il a été rejeté sur mesure. Sur cette vidéo, les têtes font 29 px et bougent de
0,69 px par frame. L'association de ByteTrack est trop agressive pour cette
échelle — elle perd l'identité des gens.

Un tracker par **simple IoU** a été écrit et mesuré : **0,85 de chevauchement
inter-frames**, association fiable. Il fait mieux sur cette vidéo que
l'algorithme plus récent.

**Ce qu'il faut retenir** : « plus récent et plus sophisticated » n'est pas un
argument. Un tracker IoU de 60 lignes bat ByteTrack sur cette scène parce que
les mesures le disent. Le code le prouve (`compteur/tracker.py`).

Deux choses qui sont allées de travers, et qu'on ne refiera pas :

- **Suppression** : le premier `tracker.py` ne conservait pas l'identité à
  travers une occultation. Une personne disparaissait derrière un panneau et
  revenait avec un nouvel identifiant — comptée deux fois.
- **Confirmation** : le seuil de confirmation était interpreté comme un seuil de
  confiance au lieu d'un nombre de frames. C'était le plan, pas le code qui
  avait tort ; le plan a été corrigé.

---

## La ligne : le bug le plus coûteux

`a_traverse` — « cette personne a-t-elle franchi la ligne ? » — **renvoyait
toujours `False`**. Pas parfois. Toujours.

Deux garde-fous incompatibles s'empilaient : l'un exigeait qu'un point de la
track tombe dans la bande de la ligne, l'autre exigeait un déplacement d'au
moins l'épaisseur de la bande (30 px). Les deux ne pouvaient jamais être
vraies en même temps. Les personnes lentes (moins de 30 px par frame) étaient
rejetées ; les rapides (qui sautaient par-dessus la bande) aussi.

**Le correctif** : au lieu de demander « la position est-elle dans la bande ? »,
demander **« la trajectoire a-t-elle croisé la bande ? »**, en calculant le point
exact de croisement par interpolation.

Une convention a dû être tranchée : quand une personne pose un pied **pile sur
la ligne**, elle est comptée ou pas ? Décision : **la ligne appartient au côté
d'arrivée**. Une personne ne peut pas être comptée deux fois parce qu'elle a
oscillé dessus.

Le test de contre-épreuve : l'ancien code fait 6 échecs, le nouveau 0.

---

## Le lissage : une solution qui ne marche pas

Quand le compteur rendait zéro, l'idée était de **lisser les positions** —
moyenne mobile sur 10 frames — pour que le bruit du détecteur ne fausse pas les
positions.

**Ça ne peut pas fonctionner, et c'est démontrable en une ligne.** Si la moyenne
d'une fenêtre atteint x = 320, c'est qu'au moins une position brute de cette
fenêtre vaut ≥ 320. Le lissage ne crée donc jamais de franchissement. Il ne
peut que le retarder.

Ce qui a été vérifié : 16 128 combinaisons et 3 000 signaux synthétiques. Zéro
franchissement créé par lissage. Zéro.

Pire, le lissage **dégradait** le comptage. Le verrou anti-rebond lisait le côté
de la position **brute**, déjà passée de l'autre côté, pendant que le test de
franchissement lisait la position lissée, encore en retard. Le verrou rejetait.
**1 compté devenait 0.**

Le lissage est désactivé par défaut (fenêtre = 1) et le mode brut est le seul
qui fonctionne à cette échelle. Trois `xfail` subsistent dans la suite de
tests, documentant le problème au lieu de le cacher.

---

## La bande de détection : le retournement inattendu

L'idée est simple : **ne détecter que dans une bande autour de la ligne**.
Si quelqu'un n'est pas près de la ligne, il n'a aucune raison d'être analysé.

Le résultat mesuré a été **le contraire de ce qui était attendu** :

| configuration | personnes détectées / frame | compté |
|---|---|---|
| image entière | 79,1 | **231** |
| bande 200 px | 13,2 | **257** |

**Détecter 6 fois moins de monde, et en compter plus.**

L'explication n'a été trouvée qu'après coup, et c'est le pivot du projet :
hors bande, le tracker s'efforce d'associer des dizaines de personnes qui se
gênent entre elles — les boîtes fusionnent, les identités se permutent. Dans
la bande, il a quelques cibles propres à suivre, et il les suit correctement.

**Le bruit des autres lui coûtait des comptes.** Restreindre l'analyse a
supprimé la source d'erreur, pas la signal.

Un second résultat, mesuré : **à largeur totale égale, la répartition compte
plus que la quantité.** 200 px d'un seul côté (257) font mieux que 100 px de
chaque côté (254). C'est ce qui a conduit à la bande dissymétrique actuelle.

---

## L'abandon après la ligne

Une personne qui a franchi la ligne ne peut plus jamais être comptée. Elle peut
donc être oubliée.

Implémenté, le compteur est **resté identique** (256 avec ou sans), et 67 561
tracks ont été lâchées sur une analyse de 3000 frames. Le tracker travaille
désormais sur une population minuscule.

**Ce que ça n'a pas donné** : un gain de temps (−4 %, dans le bruit). Le goulot
n'était pas là. L'optimisation a été conservée parce qu'elle est gratuite et
qu'elle réduit la mémoire, pas parce qu'elle accélère.

---

## Ce qui n'a pas marché : « une image sur deux »

Hypothèse : le cortège bouge de 0,69 px par frame — si lent qu'une image sur
deux suffirait, et on doublerait la vitesse.

| images vues | compté | écart |
|---|---|---|
| une sur une | 231 | référence |
| une sur deux | 200 | **−13,4 %** |
| une sur trois | 167 | **−27,7 %** |

Sauter une image, ce n'est pas « regarder moins souvent ». C'est **briser la
continuité** dont dépend toute l'identification. Le tracker voit quelqu'un se
déplacer de deux pas d'un coup, et parfois directement de l'autre côté de la
ligne.

Piste fermée, définitivement.

---

## Le sens : l'erreur qui a coûté le plus cher

Pendant plusieurs heures, le compteur « ne marchait pas » — 0 personne, alors
que le logiciel était fonctionnel.

Le problème n'était pas le logiciel. **C'était le script de mesure** : il
utilisait `sens=+1`, et le cortège va vers la gauche. Sur la vidéo de
référence :

- `sens=-1` → **257 personnes**
- `sens=+1` → **1 personne**

Toutes les conclusions tirées pendant des heures — « le comptage ne marche
pas », « le tracker est inutile », « le lissage ne peut rien sauver » — étaient
fausses. Elles décrivaient le mauvais sens de comptage.

**Leçon** : avant de conclure qu'un logiciel est cassé, vérifier que le test
utilise les mêmes paramètres que l'utilisateur. Ici, l'utilisateur obtenait 314
et le script obtenait 0 sur la même vidéo : l'écart était le signal.

**Conséquence permanente** : l'interface affiche un rappel quand le compteur
reste à zéro, et les libellés de sens sont explicites (« Droite → gauche »)
plutôt qu'ambiqus (« Avant → après »).

---

## La Résolution d'analyse : le plafond mal choisi

Le réglage `taille_entree` indique à quelle résolution le modèle regarde
l'image. Sur une vidéo 720p, une valeur de 640 réduit l'image de moitié — les
têtes font 14 px au lieu de 29, et le modèle perd 17 % du décompte.

| résolution | compté | img/s | coût |
|---|---|---|---|
| 640 | 231 | 43 | ×1,0 |
| **1280** | **257** | **27** | **×1,59** |
| 1920 | 271 | 18 | ×2,36 |

Le rendement s'effondre : 640 → 1280 donne +26 personnes pour 41 s de plus ;
1280 → 1920 donne +14 pour 54 s. Le plafond à 1280 est une décision de **rapport
coût/précision**, pas de faisabilité.

L'hypothèse « plus c'est haute résolution, mieux c'est » était intuitive et
fausse. Sur une vidéo 1280 de large, analyser à 1920 revient à agrandir une
image au-delà de sa résolution réelle : rien n'est gagné, tout est interpolé.

---

## Le panneau de réglages : trop long, puis trop court

Première version : chaque réglage avait une explication de 6 à 12 lignes. Le
panneau faisait **2335 px** dans une fenêtre de 850. L'utilisateur ne pouvait
pas voir le bouton « Lancer » sans défiler.

Ce qui a été mesuré : **12 encarts d'aide, 480 px chacun**. Les encarts — un
texte visible sous chaque champ — pas la longueur des textes.

La solution : un bouton **« Afficher l'aide des réglages »** qui affiche ou
masque tous les encarts d'un coup. Panneau à **984 px**, il tient.

Les explications restent accessibles par info-bulle au survol, et le bouton les
montre en entier quand on en a besoin. Rien n'est perdu, mais rien n'est
permanement à l'écran non plus.

**Deux affichages au lieu d'un** : l'info-bulle pour le réglage survolé, le
bouton pour la vue d'ensemble. Le premier est contextuel, le second est
lisible d'un coup.

---

## Le récapitulatif : ce qui n'a pas été fait

Un taux d'erreur automatique a été envisagé, plusieurs fois.

Il n'a pas été fait, et la raison est simple : **sans vérité terrain, il n'y a
rien sur quoi se baser**. Une fourchette inventée (« entre 130 et 160 ») serait
un mensonge habillé en statistique.

Ce qui a été fait à la place : **des boîtes vertes**. Une personne comptée voit
sa boîte passer au vert pendant 145 frames — le temps mesuré de traverser la
zone après la ligne. Une personne qui traverse **sans** que sa boîte ne
devienne verte est un raté, visible à l'œil.

C'est un changement de nature : au lieu d'un chiffre à comparer, on **voit**
les erreurs pendant l'analyse. Vingt secondes de visionnage suffisent.

La seule mesure chiffrée qui reste possible, c'est le compte manuel sur un
segment. C'est ce qui reste à faire.

---

## L'exécutable : ce que CUDA impose

**Le mode onefile ne fonctionne pas.** PyInstaller sait tout regrouper dans un
fichier unique, mais le format d'archive utilise des positions sur 32 bits
signés, et les DLL CUDA seules pèsent 3,8 Go. **Le binaire se construit sans
erreur, puis plante au lancement**, sans message.

La première explication donnée — « la limite ZIP de 4 Go » — était **fausse**.
La cause a été mesurée : offsets 32 bits signés, limite de 2,15 Go.

Deux pièges sous PyInstaller, traités explicitement :

- l'extension NMS de torchvision s'appelle `_C_stable` depuis la 0.29 et n'est
  pas un module Python — sans hook dédié, l'inférence s'arrête sur `Couldn't
  load custom C++ ops` ;
- les DLL CUDA sont chargées par `torch.ops.load_library()`, jamais par un
  `import` — PyInstaller ne les voit pas.

**Trois pièges Torchvision/torch qui ont été corrigés** : installer le
torchvision CPU alors que torch est CUDA fait planter `torchvision::nms` sur
CUDA. Les deux doivent venir du même index.

---

## Ce que le projet ne fait pas, et pourquoi

Ces exclusions ne sont pas « pas encore implémenté ». Elles ont été **arbitrées** :

- **identification de personnes** : hors périmètre, pour l'éthique et le droit
- **multi-caméra** : une caméra, une ligne
- **API web** : le moteur est prêt (`analyser_video` est une fonction
  `vidéo + config -> résultat`), la décision est reportée — l'hébergement et
  le RGPD sont d'autres chapitres
- **densimétrie** : voir plus haut
- **entraînement de modèle** : modèles locaux uniquement
- **AMD** : un seul exécutable buildé CUDA. Un build CPU ferait 200 Mo au lieu
  de 3 Go ; livrer les deux doublerait la taille pour un cas d'usage qui n'est
  pas le nôtre
- **détection hors bande** : choix assumé, voir plus haut

---

## L'architecture, et pourquoi elle est contrainte

```
compteur/    moteur pur, AUCUNE dépendance graphique
interface/   fenêtre PySide6
config/      default.json, source unique des valeurs par défaut
```

Le point clé : `compteur/compteur.py::Compteur` est le moteur. L'interface
l'appelle, les tests l'appellent, et une future API web l'appellera sans rien
réécrire.

**La contrainte est tenue par un test** (`tests/test_isolation_ui.py`) : aucun
module de `compteur/` n'importe PySide6, Tkinter, PyQt ou wx. L'analyse
statique vérifie l'AST, pas l'exécution — un `import` caché dans une fonction
serait détecté, une chaîne volée non.

Ce test a déjà attrapé une vraie erreur : `ast.alias.nom` au lieu de
`ast.alias.name` — le test plantait au lieu de signaler l'interdit, et
l'isolation n'était donc pas vérifiée du tout.

`config/default.json` est la source unique des valeurs par défaut ; le dataclass
`Config` ne fait que le refléter. Un test compare les deux pour qu'ils ne
divergent jamais. Sous PyInstaller, le fichier est embarqué et retrouvé via
`sys._MEIPASS`.

---

## Ce qu'il reste à faire

1. **Reconstruire l'exécutable** — il ne contient pas la bande dissymétrique ni
   les boîtes vertes
2. **Réduire de 9,7 Go à ~4 Go** — ultralytics embarque TensorRT, TensorFlow,
   ONNX Runtime, xformers, bitsandbytes, tous inutilisés
3. **Compter à la main** — le seul test qui dirait si 257 est juste
4. **Nettoyer** les fichiers de test en double

Et plus tard : un exécutable qui télécharge torch au premier lancement, pour
passer de 9 Go à 50 Mo.

---

## Ce qu'il faut retenir

**Les mesures contredisent les intuitions, presque toujours.** Le plafond de
résolution, le sens de comptage, la bande de détection, la répartition de la
bande : quatre fois, l'intuition était fausse et la mesure dit le contraire.
« Plus c'est mieux » et « plus récent c'est mieux » n'ont jamais rien prouvé ici.

**Une erreur de mesure coûte plus cher qu'un bug.** Le sens inversé a produit
des heures de travail sur un logiciel qui fonctionnait. Vérifier que le test
utilise les mêmes paramètres que l'utilisateur prend dix secondes.

**Une restriction bien placée vaut mieux que plus de puissance.** La bande de
200 px n'est pas une optimisation : elle supprime la source d'erreur.

**Ce qui n'est pas mesuré ne doit pas être affiché.** Pas de taux d'erreur
inventé, pas de fourchette sans base. Le logiciel dit ce qu'il sait et se tait
sur le reste.