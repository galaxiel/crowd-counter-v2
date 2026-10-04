# Tâche 18 — aide repliable derrière un bouton « Aide »

## Le problème, mesuré

Avant : **2132 px** de panneau (mesuré après la tâche 17) pour **618 px** de zone
dans une fenêtre de 850 px. Défilement : 1514 px.

La cause n'était pas la longueur des textes — c'est le **nombre** d'encarts.
Douze encarts dépliés en permanence entre l'opérateur et le réglage qu'il
cherche, c'est littéralement le symptôme décrit : « un pavé sous chaque
paramètre ».

## L'option retenue : le bouton « Aide » (option 1)

`_poser()` pose désormais toujours l'encart, mais **masqué**, et un bouton unique
en haut du panneau — `QPushButton` checkable, libellé auto — affiche ou masque
l'ensemble.

Pourquoi celle-là plutôt que l'option 2 (réduire chaque encart à sa valeur
conseillée seule) :

- Elle respecte la demande « une explication claire de **chaque** paramètre ».
  L'option 2 laisse 12 × ~30 px d'explication abrégée quand même affichée, donc
  la moitié du problème reste.
- Elle rend le panneau lisible **par défaut**, ce qui est la réclamation.
- Elle ne coûte rien : les textes sont déjà dans les info-bulles du champ et du
  titre. Le bouton ne déplace pas la documentation, il décide seulement de sa
  forme affichée.
- Le repli est **non destructif** : l'encart est construit et présent en mémoire
  dans les deux états, donc testable et jamais perdu.

Le libellé du bouton change avec l'état (« Afficher… » / « Masquer… ») : il n'y
a jamais à deviner si l'aide est ouverte.

## Hauteurs mesurées

| État | Hauteur du panneau | Défilement restant |
|---|---|---|
| Avant, encarts toujours dépliés | 2132 px | 1514 px |
| **Aide masquée (défaut)** | **984 px** | 366 px |
| Aide dépliée (bouton cliqué) | 2138 px | 1520 px |

Gain sur l'état par défaut : **−1154 px, soit −54 %**. Aucun mot d'explication
retiré : `AIDE` est inchangé.

## La zone défilante reste — elle n'a pas disparu

C'est la mesure, pas une intuition, qui tranche : la zone fait **618 px**, et le
panneau masqué **984 px**. Les 232 px manquants ne viennent pas des encarts mais
du reste de la fenêtre — vue vidéo, compteur, barre de statut, barre de
boutons. Sans zone, Qt comprimerait le panneau et le bouton « Lancer » deviendrait
inaudible : l'analyse ne serait plus lançable.

La zone est donc **conservée**, et `test_le_panneau_defile_dans_la_fenetre` a été
adapté (docstring) plutôt que supprimé : son assertion porte sur le bouton
« Lancer » atteignable, ce qui reste vrai et reste important. Ce qui a changé,
c'est le contenu de la zone — 366 px de parcours au lieu de 1514.

## Modifications

- `interface/panneau_reglages.py`
  - bouton `bouton_aide` (checkable) en tête du layout racine ;
  - `_poser()` : signature simplifiée (fin du paramètre `aide_visible`), pose
    l'encart systématiquement, titre vide = pas d'étiquette muette ;
  - `_encart()` : construit toujours, `setVisible(self._aide_visible)` ;
  - `_maj_aide()` : bascule tous les encarts + met le libellé à jour ;
  - `aide_visible()` : lecture d'état pour les tests ;
  - les deux `insertWidget` manuels (seuil de confiance, sens) supprimés : la
    nouvelle `_poser()` fait le même travail, correctement.
- `interface/app.py` : commentaire de la zone défilante réécrit avec les
  mesures réelles. **Aucun changement de code** dans `app.py`.
- `interface/style.py` : style `QPushButton#bouton_aide` (discret, aligné à
  gauche — « Lancer » garde la place de bouton principal) ; commentaire de
  `#aide_reglage` corrigé.
- `AIDE` : **inchangé**, comme demandé.

## Tests

Suite complète : **488 passés, 3 xfailed** (484 + 4 nouveaux). Aucun xfail
ajouté, aucun test désactivé.

`test_chaque_reglage_a_une_aide_visible_sous_le_champ` devient
`test_chaque_reglage_a_une_aide_survolable` : il vérifie maintenant **les deux**
canaux (info-bulle *et* encart), avec égalité au texte de `AIDE` — l'ancien nom
et son docstring affirmaient que l'encart est obligatoire en permanence, ce qui
est précisément ce que la tâche supprime.

Nouveaux :
1. `test_chaque_reglage_a_une_aide_survolable` — tooltip == `AIDE[champ]` pour
   chaque réglage, et chaque encart porte sa ligne « Valeur conseillée ».
2. `test_aucun_encart_d_aide_n_est_visible_sans_clic` — l'état par défaut est
   zéro encart visible. C'est le test qui verrouille la réclamation.
3. `test_le_bouton_aide_bascule_tous_les_encarts` — clic → les 12 visibles et
   libellé « Masquer » ; re-clic → 0 visible et libellé « Afficher ».
4. `test_le_bouton_aide_ne_change_aucun_reglage` — ouvrir l'aide n'émet aucune
   `Config` et ne déplace aucun curseur.
5. `test_le_panneau_tient_sans_defilement_aide_masquee` — compare les hauteurs
   masqué/déplié sur la vraie fenêtre et vérifie que « Lancer » reste atteignable
   dans les deux états.

## Reste à faire / limites

- Le bouton n'est pas mémorisé entre les sessions : à chaque ouverture le
  panneau est en mode « je règle ». Choix délibéré (un panneau qui s'ouvre avec
  366 px de parcours est le mode normal), mais c'est un second tour possible si
  l'utilisateur regulated beaucoup.
- La documentation existe toujours à deux endroits au lieu d'un. C'est le prix
  de l'option 1, et le bouton rend le doublon « encart + info-bulle » sans coût visible.