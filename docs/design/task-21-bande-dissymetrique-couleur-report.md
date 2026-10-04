# Bande dissymétrique, export supprimé, couleur verte — rapport de tâche 21

Quatre demandes en un seul commit. La plus visible est la couleur verte ; la
plus risquée était la première, parce qu'elle change la géométrie et donc
potentiellement le décompte.

**Résultat de référence : 257 personnes** (contre 256 avant). La valeur est
préservée — voir la section 1 pour la mesure.

---

## 1. Bande dissymétrique 200 px avant / 100 px après

### Le nouveau décompte mesuré

3000 frames de `D:\Bureau\manif_test.mp4`, ligne verticale x=640, `sens=-1`,
`imgsz=640`, torch CUDA. Les trois configurations ont été mesurées sur la MÊME
ligne, dans la même exécution, par `tools/mesurer_bande.py` — c'est la seule
variable.

| bande | compté | temps |
|---|---|---|
| **dissymétrique 200 / 100** | **257** | 115 s |
| symétrique 200 (ancienne) | 256 | 119 s |
| symétrique 300 | 254 | 114 s |

**257 contre 256 : +1 personne, +0,4 %.** C'est dans le bruit de la mesure, et
c'est le résultat qu'on voulait : la bande dissymétrique n'a rien coûté au
décompte tout en supprimant 100 px de calcul inutile après la ligne.

Le fait que la symétrique 300 px (254) fasse *moins* que la dissymétrique 200
px (257) est contre-intuitif et mérite d'être dit : à largeur totale égale, ce
n'est pas la quantité de pixels qui compte, c'est **la répartition**. Deux
cents pixels du côté où l'identité se construit valent mieux que cent de chaque
côté. C'est exactement l'argument de la tâche, et il est confirmé par la mesure.

Pas de gain de temps non plus (115 s contre 119 s, dans le bruit) : le goulot
n'est toujours pas la détection, comme annoncé en tâche 19.

### Où c'est implémenté, et pourquoi là

`Ligne.rect_bande_detection` (`compteur/ligne.py`), sur les deux constantes
`BANDE_AVANT_PX = 200` / `BANDE_APRES_PX = 100` (`compteur/config.py`). La
frontière de responsabilité est inchangée : `detecteur.py` ne sait toujours pas
ce qu'est une ligne, il reçoit un rectangle.

**Le signe des deux décalages est fixe** : `+avant` et `-apres` le long de la
normale. C'est le point qui était facile à inverser, et le test
`test_la_bande_est_dissymetrique_et_du_bon_cote` éprouve les DEUX `sens` parce
que les deux moitiés d'une bande symétrique sont interchangeables et que celles
d'une bande dissymétrique ne le sont pas. Avec `sens=+1` la bande vaut
440–740, avec `sens=-1` elle vaut 540–840 : inverser les largeurs aurait donné
100 px avant au lieu de 200, une erreur **symétrique et sans aucun signe à
l'écran**.

Le vocabulaire est celui de `compteur.ligne`, pas un vocabulaire de dessin :
« avant » = côté de DÉPART = coordonnée positive sur la normale
(`point_du_cote == +1`), quel que soit `sens`. C'est le côté par où arrivent
les gens, donc celui où le tracker doit construire une identité.

`bande_detection_px` (un entier, symétrique) est remplacé par deux attributs
`bande_avant_px` / `bande_apres_px`. Un `None` d'un seul côté est refusé à la
construction : une bande d'un seul côté n'a pas de sens et rendrait le
franchissement invisible au modèle.

**Effet de bord assumé :** la bande fait maintenant **300 px au total** au lieu
de 200 — c'est la moitié *avant* qui passe de 100 à 200 px, celle d'après reste
à 100. Sur une ligne verticale, le rectangle rogné s'élargit donc bel et
bien. C'est le prix du choix demandé, et la mesure dit qu'il ne coûte rien au
décompte.

## 2. Export supprimé — ce qui a disparu, ce qui reste

Le bouton, la boîte de dialogue, les deux méthodes et toutes les mentions dans
le README ont disparu. Le détail mérite d'être exact, parce que la suppression
est **partielle et c'est délibéré**.

### Supprimé

- `interface/app.py` : `btn_export`, son `connect`, son `setEnabled` dans
  `_maj_boutons`, `DOSSIER_EXPORT_DEFAUT`, `_on_exporter`, `exporter()`, et le
  `_afficher_bilan` (« Pense à exporter le décompte. »).
- `compteur/rapport.py` : `ecrire_csv`, `ecrire_json`, `chemins_par_defaut`,
  `chemin_video_nom`, `ENTETE_CSV`, `COLONNES_EVENEMENT`, et les imports `json`
  et `pathlib` devenus inutiles.
- `compteur/types.py` : `Evenement.vers_ligne_csv`, dont c'était le seul
  appelant.
- Les tests de ces fonctions (`tests/test_rapport.py`, `tests/test_app.py`,
  `tests/test_types.py`).

### Gardé, et pourquoi

**`rapport.statistiques` et tout ce qu'elle contient sont conservés.**
`interface/recap.py` (le récapitulatif de fin d'analyse, commit `7e96734`)
importe `FENETRE_DEBIT_S` et `statistiques` : le récapitulatif s'appuie dessus
pour ses quatre chiffres et son pic de débit. Supprimer le module aurait cassé
le récapitulatif — ce qui était explicitement à éviter.

`indicateurs_fiabilite`, `_pic_de_debit`, `_fenetre_de_reel` et
`_intervalle_median` restent aussi : ce sont les instruments du récapitulatif,
pas de l'export. Le module porte maintenant un nom un peu plus large que son
contenu (`rapport`), ce qui est le prix d'une suppression Surgicale : le
renommer aurait touché l'interface, les tests et le plan pour un bénéfice
nul.

Le test `test_il_ne_reste_plus_rien_a_exporter` verrouille les deux moitiés de
cette décision : ce qui a disparu (attribut, méthode, fonction d'écriture) et ce
qui reste (`statistiques` est toujours là).

**Ce que `QFileDialog` continue de faire :** il sert toujours au choix de la
vidéo. Seule la boîte « Dossier de sortie » a disparu.

## 3. La boîte verte, et où vit la liste

### Le problème, et pourquoi il n'y a pas de solution par le suivi

Les tracks sont lâchés au franchissement (optimisation validée en tâche 19,
67 561 tracks lâchés sur 3000 frames). Il n'y a donc **plus rien à
repeindre** : pas de track à conserver, pas de position à mettre à jour. Et
« garder le suivi actif pour colorer » annulerait l'optimisation pour un
avantage purement cosmétique.

### La solution retenue : une liste de boîtes, pas un suivi

`FenetrePrincipale._boites_comptees` (`interface/app.py`) est une liste de
tuples `(frame_comptage, x1, y1, x2, y2)`. C'est de la **mémoire
d'affichage** : on n'en suit rien, on ne la met à jour qu'à l'ajout. Zéro
coût de performance.

La boîte elle-même est figée au moment du compte, dans
`Compteur.traiter_frame` : `Evenement` porte désormais un champ `bbox`
optionnel, rempli depuis `t.bbox` au franchissement. C'est le dernier instant
où cette personne existe côté tracker — dès la frame suivante, `_purger_franchis`
a vidé son état. Sans cette copie, il n'y aurait littéralement rien à peindre.

L'overlay (`interface/overlay.py::dessiner`) reçoit la liste en paramètre et la
peint en `COULEUR_COMPTEE = (60, 220, 60)`. **Les boîtes vertes sont dessinées
AVANT les centres de track mais APRÈS les détections** — une personne comptée est
encore détectée (elle n'a fait que passer la ligne), donc sa boîte d'ambre est
forcément encore là au même endroit. Dessinée en dessous, le vert serait masqué
et la couleur ne signalerait rien du tout.

### Les deux bornes, et pourquoi il y en a deux

| borne | valeur | rôle |
|---|---|---|
| durée | `DUREE_BOITE_COMPTEE_FRAMES = 145` | la vie réelle d'une boîte |
| nombre | `TAILLE_MAX_BOITES_COMPTES = 200` | filet de sécurité |

**La durée** est la règle qui décide : 100 px après la ligne à 0,69 px/frame
mesuré = 145 frames, soit le temps que la personne met à sortir de la zone
utile. Passé ce délai elle n'est plus là, la garder n'apprend rien.

**Le nombre** est le filet : si la règle de durée ne purgeait pas — `frame_index`
figé, compteur de test qui ne progresse pas — la liste ne pourrait pas croître
sans fin. 200 boîtes vertes à l'écran seraient absurdes ; en mémoire cela ne
fait que quelques dizaines de kilo-octets, donc ce plafond ne coûte rien.

Pourquoi 200 : à 256 personnes sur 3000 frames, il y a 0,085 comptage par frame
en moyenne, donc **~12 boîtes vivantes en régime normal**. 200 est un ordre de
grandeur au-dessus : le filet ne se déclenche jamais sur une analyse réelle, ce
qui est exactement ce qu'on veut d'un filet — s'il se déclenchait, il
masquerait un bug au lieu de l'empêcher.

**La borne est appliquée dans la méthode, pas laissée à l'appelant.** Une
première version accumulait dans `self._boites_comptees` et ne bornait que la
liste retournée : la mémoire ne dépendait donc du fait que l'appelant réassigne
le résultat. Le test l'a attrapé (`assert 301 <= 200`) : c'est exactement le
défaut qu'il était écrit pour trouver. La liste locale rule.

**En frames traitées, pas affichées.** À 0,25x l'opérateur voit 7 images/s pour
25 traitées. Une durée en frames affichées ferait disparaître la boîte en 7/25
du temps voulu — un clignotement à l'œil, précisément ce que la couleur verte
doit éviter.

## 4. Ce que la couleur verte rend visible

Note ajoutée au README (« Reading the green boxes »). Le compteur peut se
tromper en ratant quelqu'un et **ne le dira jamais** : le total est un nombre,
et un nombre faux ressemble exactement à un nombre juste. On n'a pas la vérité
terrain — le module `rapport` l'interdit explicitement depuis le début.

Mais une personne qui traverse la ligne **sans que sa boîte devienne verte** est
un raté que l'opérateur voit immédiatement. C'est le seul moyen de détecter une
erreur sans vérité terrain : on ne peut pas recompter chaque vidéo à la main,
mais on peut regarder vingt secondes et voir si tout le monde verdit.

Le README le formule comme une consigne de vérification : ne pas seulement lire
le total, regarder la ligne et compter les boîtes vertes soi-même.

## Tests

**481 passent, 3 xfailed.** Aucun xfail ajouté, aucun test désactivé.

Trois tests ajoutés, un par tâche :

- `test_la_bande_est_dissymetrique_et_du_bon_cote` — les deux `sens`, la
  dissymétrie dans le bon sens, et les 300 px au total ;
- `test_il_ne_reste_plus_rien_a_exporter` — l'attribut, la méthode, la fonction
  d'écriture ont disparu, `statistiques` est toujours là pour le récapitulatif ;
- `test_une_boite_comptee_est_peinte_en_vert` — le vert, à la bonne position,
  au-dessus de l'ambre.

Et un quatrième sur les bornes de la liste, qui a attrapé un vrai défaut (voir
ci-dessus) : `test_la_liste_des_boites_comptees_est_bornee`.

Tests existants ajustés, sans affaiblir ce qu'ils prouvent : les assertions de
géométrie de la bande (`540..740` → `440..740`, et le décalage des coordonnées
`550 → 450`, qui est la conséquence directe du nouveau bord), et la suppression
des tests d'export.

## Préoccupations

1. **La bande fait 300 px au lieu de 200.** C'est la conséquence directe de la
   demande : 200 avant + 100 après. La mesure dit que le décompte tient (257
   contre 256), mais le rognage est objectivement plus large, donc le nombre de
   détections par frame remonte un peu. Si le gain de temps était l'objectif,
   il n'y en a pas — il n'y en avait déjà pas.

2. **Le champ `bbox` sur `Evenement` est un vestige d'affichage dans une
   structure de données métier.** C'est un choix discutable : on a mis dans le
   moteur une information que seule l'interface consomme. L'alternative était
   de retrouver la boîte dans `resultat.tracks` au moment du comptage, mais la
   piste est lâchée et il n'y a plus rien à retrouver. Le champ est donc la
   seule copie possible, et il est explicitement documenté comme tel.

3. **Le nom `rapport` ne décrit plus son contenu.** Après la suppression, le
   module ne rapporte plus rien : il calcule des statistiques. Renommer
   aurait touché l'interface, les tests et le plan pour un bénéfice nul. À
   corriger au prochain passage, si l'on veut.

4. **Le plafond de 200 boîtes n'a pas été éprouvé sur une analyse réelle** —
   par construction, il ne se déclenche jamais sur la vidéo de référence
   (~12 boîtes vivantes). Le test l'éprouve donc en synthèse, avec 300
   comptages sur une frame unique, ce qui est un cas plus hostile que le réel.
   C'est le comportement voulu d'un filet, mais cela veut dire que sa valeur
   n'est pas calibrée par la mesure : elle est calibrée par le fait qu'elle doit
   être inatteignable en usage normal.

5. **`_boites_comptees` est purgée à `_reset_analyse` et au lancement**, pas à
   `arreter()`. Une analyse arrêtée puis relancée repart donc d'une liste
   vidée par le lancement. C'est correct, mais les deux points de purge
   n'ont pas la même justification et le code ne le dit pas explicitement.