# Tâche 16 — Avertissement « 0 compté : vérifie le sens »

## Statut

Terminé. L'avertissement est en place, testé, la suite est verte.

- Aucun exécutable n'a été construit (hors périmètre, conformément à la consigne).
- `compteur/compteur.py`, `compteur/ligne.py` et `crowd-counter.spec` sont intacts.
- Le défaut de `sens` reste `+1` ; le test qui l'épingle n'a pas été touché.

## Tests

```
487 passed, 3 xfailed in 8.79s
```

480 tests passaient avant, tous passent toujours. Les 7 nouveaux tests sont dans
`tests/test_app.py`. Les 3 `xfail` sont antérieurs et inchangés : aucun n'a
été ajouté.

## Le déclencheur, et pourquoi il est simple

Choix retenu : `total == 0` ET `frames / fps >= 3.0` secondes de vidéo analysée.

La consigne évoquait une variante « ET le sens vient d'être changé par
l'opérateur », en demandant de vérifier l'état réel de `interface/app.py`
avant de trancher. J'ai relevé le critère, et je pense que c'est un choix à
expliquer plutôt qu'à appliquer tel quel.

Le cas cible est l'opérateur qui ne change pas le sens. C'est littéralement le
scénario décrit dans la demande : il trace sa ligne, laisse le défaut `+1`,
obtient zéro. Or `sens` change rarement, et dans ce scénario précis il ne
change jamais. Une condition « le sens vient d'être changé » supprimerait donc
l'avertissement dans le cas exact pour lequel il existe, et ne le laisserait
apparaître que pour l'opérateur qui a déjà compris qu'il y avait un réglage en
cause. Le critère demandé aurait donc répondu à la question inverse.

Le risque qu'on cherche à éviter avec ce critère, alerter quelqu'un qui
analyse volontairement une scène où rien ne traverse, est couvert par le
seuil de 3 secondes et par la garde de session ouverte :

- une analyse de 2 secondes ne déclenche rien ;
- une scène vraiment vide déclenche le rappel une fois, en ambre, sans
  bloquer, et il disparaît au premier comptage.

Le coût d'une erreur est qu'un opérateur a cliqué une fois sur un réglage
qu'il n'avait pas à changer. Le coût de l'absence d'avertissement est une
analyse de plusieurs minutes qui se termine sur un zéro. Le premier est visible
en un clic, le second reste invisible jusqu'à la fin.

Deux gardes supplémentaires, non demandées mais nécessaires :

- **session ouverte** (`self._session_ouverte`) : charger une vidéo et lire
  l'aperçu ne suffit pas à afficher un avertissement, rien n'a été compté ;
- **le seuil est en secondes de VIDÉO**, pas en frames affichées. Le curseur
  de vitesse de présentation ne doit ni hâter ni retarder un message qui
  parle d'un temps vécu par l'opérateur.

## Où il est affiché, et pourquoi

Label dédié `label_avertissement`, dans la colonne de droite, entre
`label_peripherique` et `label_statut`, donc juste sous le compteur et juste
au-dessus de la barre de statut.

Pas dans `label_statut`, malgré la suggestion. `label_statut` est réécrit à
chaque changement d'état : vidéo chargée, tracé de ligne, pause, erreur,
modèle changé, fin de lecture. Le rappel y serait effacé précisément au moment
où l'opérateur en a le plus besoin, et il serait écrasé par le bilan de fin de
lecture, qui est précisément le moment où la question « pourquoi zéro ? » se
pose. C'est le même raisonnement que celui qui a déjà motivé
`label_peripherique` dans ce module, et le commentaire du code l'explique
déjà : un label dédié n'est jamais réécrit.

Le label est **masqué** (`setVisible(False)`) quand il n'a rien à dire, et non
laissé vide : un `QLabel` vide garde sa place dans la colonne et ferait sauter
la mise en page.

Style : `QLabel#avertissement`, ambre `#fbbf24`, gras. Pas rouge : un rouge
concurrencerait le flash de comptage et le bouton primaire, et en ferait une
erreur alors que c'est un conseil.

## Le mécanisme

Une seule méthode, `_maj_avertissement_sens(total, frames)`, appelée depuis
`maj_compteurs`, donc depuis `_afficher`, le seul endroit où un total change.

- `_avertissement_sens_affiche` : verrou « une fois par analyse ». Il est
  remis à faux dans `lancer()`, **pas** dans `_ouvrir_session()` : cette
  dernière est aussi appelée à la reprise après pause, et y remettre le verrou
  à faux ferait réapparaître le rappel à chaque « Reprendre ».
- `total > 0` : masqué, verrou remis à faux.
- session fermée : masqué.
- `frames / fps < 3.0` : rien.

Ni modale, ni popup : rien de ce qui pourrait bloquer un traitement en cours.

Texte, dans `AVERTISSEMENT_SENS` :

> 0 compté — le sens de la flèche correspond-il au sens de marche du cortège ?

## Les tests

Sept tests, dans `tests/test_app.py`, section « Avertissement » :

| Test | Ce qu'il verrouille |
|---|---|
| `..._absent_au_demarrage` | fenêtre neuve : pas de rappel |
| `..._pas_avant_le_seuil_de_trois_secondes` | cas 3, 2,8 s : rien |
| `..._apparait_au_dela_de_trois_secondes` | cas 1, 3,2 s : le rappel s'affiche, texte exact |
| `..._disparait_des_que_le_compteur_repart` | cas 2, total 1 : masqué |
| `..._ne_clignote_pas_a_haque_frame` | une écriture, puis plus jamais |
| `..._reapparait_a_la_relance` | le rappel se redéploie après `lancer()` |
| `..._ne_saffiche_pas_hors_analyse` | pas de session, pas de rappel |

Deux détails qui ont demandé une correction, et qui méritent d'être notés :

1. **Une vidéo de 100 frames à 25 i/s** (fixture `video_longue`) est
   nécessaire : la vidéo de 5 frames du reste de la suite est trop courte pour
   atteindre 3 secondes, l'analyse y finit avant que le seuil soit franchi.
2. **Les assertions portent sur `isHidden()`, pas sur `isVisible()`.** Sous
   `offscreen`, et pour toute fenêtre jamais `show()`e, `isVisible()` renvoie
   `False` sur tous les enfants, y compris un label que l'on vient d'afficher.
   Un test écrit avec `isVisible()` passerait donc toujours, ou échouerait
   pour la mauvaise raison. Le helper `avertissement_visible()` documente
   pourquoi.

## Préoccupations

- **La pollution du cas légitime existe.** Un opérateur qui analyse une scène
  réellement vide verra le rappel. Il est ambre, discret, sans modale, et il
  disparaît au premier comptage, mais c'est le prix assumé du choix de
  déclencheur décrit plus haut. Si ce bruit se révèle gênant en usage réel, la
  correction la plus simple est de n'armer le rappel qu'après un changement de
  sens réellement observé, en comparant le `sens` au moment du `Lancer` plutôt
  qu'à tout changement de réglage.
- **Le seuil est en secondes de vidéo.** Sur une vidéo dont le conteneur
  annonce une cadence fausse, le moment de déclenchement variera. C'est
  cohérent avec le repli `FPS_REPLI` déjà présent dans `_fps()`.
- **Rien n'a été vérifié sur `manif_test.mp4` en conditions réelles** : les
  tests sont sur une vidéo synthétique. Le rappel n'a pas été vu à l'écran en
  taille réelle.
- **Aucun accès au GPU ni au modèle** dans les nouveaux tests : `Detecteur`,
  `Tracker` et `Compteur` y sont remplacés par des fakes.
