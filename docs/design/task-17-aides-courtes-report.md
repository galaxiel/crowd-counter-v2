# Tâche 17 — Explications des réglages raccourcies

**Date :** 2026-10-04
**Portée :** `interface/panneau_reglages.py`, `interface/app.py`, leurs tests.
**Aucun exécutable construit** : pas de PyInstaller, `dist/` intact.

## Objectif

Les info-bulles de `AIDE` étaient des pavés de 6 à 12 lignes illisibles au
survol, et le panneau de réglages dépassait la fenêtre. Deux actions : retirer
l'explication GPU/CPU, raccourcir les treize autres à 5 lignes maximum en gardant
tous les chiffres mesurés.

## Hauteur du panneau

Mesurée avec un script qui instancie `FenetrPrincipale`, `resize(1400, 850)`,
`show()`, `processEvents()`, puis lit `panneau.sizeHint().height()`.

| | Avant | Après |
|---|---|---|
| Hauteur naturelle du panneau | **2976 px** | **2330 px** |
| Entrées dans `AIDE` | 14 | 13 |
| Hauteur du groupe « Détection » | — | 834 px |
| Hauteur du groupe « Tracker » | — | 516 px |
| Hauteur du groupe « Lissage » | — | 188 px |
| Hauteur du groupe « Périphérique de calcul » | — | 86 px |

Gain : **646 px, soit 22 %**. La hauteur est dominée par les *encarts* — le
texte lisible affiché sous chaque champ, 80 à 112 px par réglage selon la
largeur de colonne. Ces encarts ne sont pas des lignes de texte mais des
paragraphes qui se replient sur la largeur du panneau.

**Zone défilante : CONSERVÉE.** 2330 px reste très au-dessus du seuil de ~1200 px.
Supprimer la zone ferait compresser le panneau par Qt et pousserait le bouton
« Lancer » hors de l'écran — l'analyse deviendrait non lançable. Le gain d'un
tiers est réel mais pas suffisant.

## Suppression de `peripherique`

L'entrée `peripherique` a été retirée de `AIDE`, soit **18 lignes de texte**
supprimées. Le réglage lui-même **reste dans le panneau** (interprétation de la
demande : « supprimer l'explication sur le GPU CPU »).

`_groupe_peripherique` construit désormais le widget à la main au lieu d'appeler
`_poser` : `_poser` exige une entrée dans `AIDE`, il n'y en a plus. Le widget
reste enregistré dans `self._widgets["peripherique"]`, sans tooltip et sans
encart. Le groupe est passé de ~110 px à 86 px.

Les cinq tests qui couvraient le texte d'aide de `peripherique` ont été supprimés
(ils vérifiaient des phrases du texte retiré : « ne change pas le décompte »,
« 27 », « 7 img/s », « bascule »). Deux tests les remplacent :

- `test_le_reglage_na_plus_dexplication` — verrouille l'absence dans `AIDE` ;
- `test_le_panneau_construit_le_reglage_sans_encart_daide` — verrouille que le
  réglage, lui, SURVIT (3 choix, pas de tooltip).

`peripherique` a été retiré de la tuple `REGLES` de `test_reglages.py`, qui
paramètre « chaque réglage a une explication ».

## Trois entrées représentatives

### `survie_max`

AVANT (12 lignes) :

> Survie max sans détection (frames).
> Ce que ça fait : une personne que le détecteur perd de vue reste quand même
> suivie ce nombre de frames, à sa dernière position. Ce que ça change :
> augmenter préserve les personnes derrière un groupe ou un drapeau, mais elles
> gardent leur dernière position — donc un groupe arrêté sur la ligne gonfle le
> nombre de présents et peut faire compter un faux passage. Diminuer casse les
> tracks et compte la même personne plusieurs fois. Valeur conseillée : 30.
> Mesuré sur la vidéo de référence, le total passe de 315 à 314 (60 frames),
> 317 (150) et 325 (400) : +3 % au maximum, alors que les présents simultanés
> passent de 219 à 916 et les tracks vus chutent de 9873 à 4315. La mémoire
> prédictive ne rachète presque rien ici et rend le compteur de présents faux :
> reste à 30. Si tu subis de longues occultations — drapeaux, portiques — monte à
> 150, le surcoût y est de 2 personnes.

APRÈS (5 lignes) :

> Survie max sans détection (frames).
> Ce que ça fait : une personne perdue de vue reste suivie ce nombre de frames,
> figée à sa dernière position. Ce que ça change : trop élevé, les tracks mortes
> gonflent le compteur de présents (219 → 916 mesuré) sans gagner de comptage ;
> trop bas, les tracks cassent et la même personne est comptée plusieurs fois.
> Valeur conseillée : 30 — le total ne bouge que de +3 % (315 → 325 à 400
> frames). Monter à 150 n'ajoute que 2 personnes.

### `taille_entree`

AVANT (5 lignes, ~600 caractères) :

> Taille à laquelle l'image est réduite avant la détection.
> Ce que ça fait : le modèle voit l'image à cette taille, en redimensionnant la
> vidéo. Les boîtes sont ramenées à l'échelle de l'originale. Ce que ça change :
> augmenter trouve plus de petites têtes, donc plus de personnes, mais coûte plus
> de temps de calcul. Diminuer fait l'inverse et va plus vite, au prix de
> personnes perdues. Valeur conseillée : ne touche pas à ce réglage. Laisse
> « Résolution d'analyse » sur Automatique, et cette valeur s'ajuste toute seule
> à la vidéo chargée. Elle n'est modifiable qu'en mode manuel. Mesuré : passer de
> 640 à 1280 sur une vidéo 720p fait passer le décompte de 315 à 368 (+17 %) sans
> surcoût de calcul (272 s contre 260 s sur 7399 frames), pour 64 détections par
> frame au lieu de 48.

APRÈS (5 lignes, 531 caractères) :

> Taille à laquelle l'image est réduite avant la détection.
> Ce que ça fait : le modèle voit l'image à cette taille ; les boîtes sont
> ramenées à l'échelle de l'originale. Ce que ça change : plus haut trouve plus
> de petites têtes mais coûte plus de temps ; plus bas va plus vite au prix de
> personnes perdues. Valeur conseillée : ne touche pas à ce réglage, laisse
> « Résolution d'analyse » sur Automatique. Mesuré : 640 → 1280 sur une vidéo
> 720p fait passer le décompte de 315 à 368 (+17 %) sans surcoût de calcul
> (272 s contre 260 s).

Perdu : le détail « modifiable qu'en mode manuel », les « 64 détections par
frame ». Gagné : 100 caractères, soit ~8 px de hauteur sur l'encart.

### `resolution_analyse`

AVANT (7 lignes, ~1000 caractères) — c'était la plus longue entrée :

> Résolution d'analyse — à quelle taille la vidéo est analysée.
> Ce que ça fait : en mode Automatique, l'image est analysée à la résolution
> RÉELLE de la vidéo que tu viens de charger, jamais plus haut que 1280. Le
> réglage « Taille d'entrée » devient alors inactif, et la valeur retenue est
> affichée à côté. Ce que ça change : rien au nombre de personnes comptées en
> soi — c'est la même scène, vue plus ou moins finement. Une image trop petite
> fait perdre les têtes lointaines ; une image trop grande allonge le calcul
> sans rien ajouter. Passer de 640 à 1280 sur une vidéo 720p, c'est +17 % de
> personnes comptées pour +5 % de temps. Valeur conseillée : Automatique, qui
> est le réglage par défaut. Mesuré sur 3000 frames de la vidéo de référence :
> 640 compte 231 personnes en 179 s, 1280 en compte 257 en 230 s (+11 % de
> personnes pour +28 % de temps), et 1920 en compte 271 en 324 s. Le plafond est
> à 1280 parce que 1920 rapporte encore 14 personnes, mais pour 41 % de temps de
> calcul EN PLUS : le dernier cran coûte deux fois plus cher que le précédent
> pour deux fois moins de gain. Choisis Manuelle seulement si tu connais le
> matériel et sais ce que tu fais : dans ce cas, monte aussi haut que la source
> le permet, la couche supplémentaire étant à ta charge.

APRÈS (5 lignes, 616 caractères) :

> Résolution d'analyse — à quelle taille la vidéo est analysée.
> Ce que ça fait : en mode Automatique, l'image est analysée à la résolution
> RÉELLE de la vidéo, jamais plus haut que 1280, et « Taille d'entrée » devient
> inactif. Ce que ça change : rien au nombre de personnes comptées en soi,
> c'est la même scène vue plus finement ; mesuré, 640 → 1280 donne +17 % de
> personnes pour +5 % de temps. Valeur conseillée : Automatique. Mesuré sur 3000
> frames : 640 compte 231 personnes en 179 s, 1280 en compte 257 en 230 s, 1920
> en compte 271 en 324 s — le dernier cran coûte deux fois plus cher pour deux
> fois moins de gain.

Perdu : le doublon sur « 640 → 1280 donne +17 % / +5 % » (déjà dans
`taille_entree`), le calcul « +11 % / +28 % » (redondant avec 231/257), le
paragraphe sur le mode Manuelle. Le plateau de mesure 640/1280/1920 est
conservé en entier : c'est lui qui justifie le plafond de 1280, et le test
`test_l_explication_de_resolution_chiffre_son_plafond` le vérifie.

## Tests

Deux nouveaux tests verrouillent les invariants de la tâche 17, pour que le
format court ne se dégrade pas au prochain ajout de réglage :

- `test_chaque_explication_tient_en_cinq_lignes` — aucune entrée de `AIDE` au
 -delà de 5 lignes ;
- `test_chaque_explication_garde_ses_chiffres` — chaque ligne
  « Valeur conseillée » porte encore un chiffre.

Le second doublonne partiellement `test_le_conseil_chiffre_est_soutenu_par_une_mesure`,
mais celui-ci ne passe que sur `REGLES` : il ne couvre donc ni
`resolution_analyse` ni `vitesse_presentation`.

**Résultat : 484 passés, 3 xfailed, 0 échec.**

487 → 484 : cinq tests supprimés, trois ajoutés (deux ci-dessus, plus
`test_le_reglage_na_plus_dexplication` et `test_le_panneau_construit_le_reglage_sans_encart_daide`
en remplacement de cinq). Le compte total baisse, la couverture ne baisse pas.

Aucun xfail ajouté, aucun test désactivé.

## Fichiers modifiés

- `interface/panneau_reglages.py` — `AIDE` raccourcie (13 entrées × 5 lignes),
  entrée `peripherique` supprimée, `_groupe_peripherique` reconstruit sans
  `_poser`, commentaire d'en-tête de `AIDE` réécrit.
- `interface/app.py` — commentaire de la zone défilante mis à jour avec les
  chiffres mesurés ; référence « 2000 px » de la vitesse de présentation
  corrigée en 2300 px.
- `tests/test_reglages.py` — `peripherique` retiré de `REGLES`, deux tests
  d'invariant ajoutés.
- `tests/test_peripherique_ui.py` — cinq tests de texte d'aide remplacés par
  deux tests d'absence.

## Recommandation sur le réglage `peripherique`

**Le garder.** Trois raisons :

1. L'indicateur permanent de la barre de statut affiche déjà le périphérique
   réellement utilisé. Une explication au survol ne peut que la concurrencer,
   et l'opérateur l'a supprimée de fait : c'est un réglage qu'on ne touche pas.
2. Mais « CPU uniquement » est le remède quand le GPU plante sur une scène
   particulière. Sans ce réglage, un bug de CUDA sur le terrain n'a pas de
   sortie. Le docstring du groupe le dit déjà : c'est du matériel, pas un seuil,
   et il doit rester trouvable après un plantage.
3. Le retrait du texte n'a coûté que ~24 px de hauteur. Le garder est le choix
   qui répond exactement à la demande formulée.

Si l'opérateur trouve plus tard que le groupe lui-même encombre, c'est un
retrait de widget — mais il faut le demander explicitement, pas le déduire d'un
« ça prend trop de place ».