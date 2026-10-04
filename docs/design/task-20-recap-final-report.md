# Tâche 20 — Récapitulatif de fin d'analyse

## Ce qui a été fait

Un récapitulatif de fin d'analyse s'affiche dans la fenêtre après l'arrêt du
traitement : quatre chiffres clés et une courbe du débit, tracée sur toute la
l'analyse.

```
Analyse terminée

Personnes comptées      256
Durée analysée          2 min 41 s
Débit moyen             95 pers/min
Pic de débit            147 pers/min à 1 min 12 s
```

plus la courbe « personnes par minute en fonction du temps », dessinée sous
les chiffres.

## Où il est affiché

Une **section de la colonne de droite**, entre la barre de statut et la zone de
réglages — pas un onglet.

La fenêtre n'a pas d'organisation par onglets : elle a deux colonnes, une
d'affichage (vidéo, compteur) et une de réglages. Un `QTabWidget` aurait
imposé de choisir à chaque instant entre la vidéo et son bilan, alors que
l'opérateur a besoin des deux au moment où il arrête. Une section masquée
jusqu'à la fin de l'analyse garde la vidéo et le bilan visibles ensemble, et ne
coûte aucune place avant qu'il y ait quelque chose à montrer.

Masqué au démarrage, effacé au lancement d'une nouvelle analyse et au
chargement d'une autre vidéo — un récapitulatif de l'analyse précédente
resterait en place pendant qu'un nouveau décompte démarre, et l'opérateur
lirait l'ancien total comme le nouveau.

## Les données existaient-elles déjà ?

**Oui, presque toutes.** Aucun chiffre n'a été recalculé par ce module :

- `compteur/rapport.statistiques` fournit déjà `total`, `duree_s`,
  `personnes_par_minute`, `debit_max_par_minute` et `debit_max_fenetre_s` ;
- la durée affichée est `duree_s` du rapport — donc la durée de la VIDÉO, pas
  le temps de calcul, qui sous-estimerait le débit d'un facteur ~2 ;
- le pic affiché est `debit_max_par_minute`, repris tel quel.

Recalculer ces valeurs dans l'interface aurait créé une **seconde version de
la vérité**, divergente de celle du JSON d'export dès qu'un export serait
produit ailleurs. L'écran et l'export disent donc la même chose, calculés une
seule fois.

Une seule chose a été calculée, parce que l'export ne la contient pas :
`instant_du_pic` localise le moment où la fenêtre glissante atteint le pic. Sa
**valeur** vient du rapport ; la fonction ne fait que dire *quand* elle a été
observée, avec la même fenêtre et le même décompte.

La **courbe** elle-même est également nouvelle : l'export donne un total et un
pic, pas la répartition entre les deux. `points_de_debit` la construit à partir
des événements datés.

## Choix qui méritent d'être explicités

**Le pic n'est pas toujours un débit par minute.** `debit_max_par_minute` est un
compte sur une fenêtre de 60 s, rétrécie à la durée réelle si l'analyse a duré
moins d'une minute. Sous 60 s observées, l'écran affiche « 12 pers / 40 s »
plutôt que « 18 pers/min » : extrapoler sur une minute qui n'a pas été
observée est exactement le genre de précision inventée que ce projet refuse
ailleurs.

**Les minutes vides sont des zéros, pas des trous.** Une analyse de 3 minutes
donne 3 points même si personne n'a traversé. Une courbe où les trous
disparaîtraient se lirait « personne ne passe » entre deux points collés, alors
que la réalité est « personne ne passe PENDANT une minute ».

**Le tracé est manuel** (`paintEvent`) et non via `QChartView` : QwtPlot et
QChart feraient deux dépendances de plus dans un exécutable déjà lourd, pour
cinq lignes et un axe.

## Ton

Sobre. Titre « Analyse terminée », aucun emoji, aucune congratulation. Le titre
reste dans le gris des titres de section du reste de l'interface : c'est un
constat, pas une annonce.

## Tests

2 tests, dans `tests/test_recap.py`. Pas de test de rendu pixel par pixel — il
échouerait sur un arrondi de Qt sans rien dire de faux sur le comportement.

- `test_recap_affiche_les_chiffres_du_rapport_et_une_courbe_par_minute` —
  les quatre valeurs affichées et la série de la courbe (3 points, 4/5/4).
- `test_points_de_debit_garde_les_minutes_vides` — une minute sans
  franchissement reste un point à zéro, et une durée nulle ne divise pas par
  zéro.

Le premier test verrouille surtout que le pic affiché (6, fenêtre glissante)
diffère du maximum de la courbe (5, tranche fixe) : afficher l'un pour l'autre
est la même erreur vue de deux côtés, et c'est le piège réel de cette tâche.

## État de la suite

**497 tests passent, 3 xfailed.** 495 + 2 nouveaux, aucune régression.

## Fichiers

- `interface/recap.py` — nouveau. `PanneauRecap`, `CourbeDebit`,
  `points_de_debit`, `instant_du_pic`, `formater_duree`, `formater_debit`.
- `interface/app.py` — la section est insérée dans la colonne de droite,
  remplie par `_afficher_recap()` depuis `arreter()`, effacée par `lancer()`
  et `_reset_analyse`.
- `interface/style.py` — trois règles pour les libellés du récapitulatif.
- `tests/test_recap.py` — nouveau.

`compteur/compteur.py` et `crowd-counter.spec` n'ont pas été touchés. Aucun
exécutable n'a été construit, aucune mesure n'a été faite sur `manif_test.mp4`.