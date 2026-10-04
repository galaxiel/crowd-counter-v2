# Bande de détection de 200 px — rapport de tâche 19

## Ce qui a été fait

Cinq points, un seul commit. Le résultat de référence est **inchangé côté
API publique** : `analyser_video` et l'interface produisent désormais le même
décompte, et la bande est un attribut de la ligne, pas un réglage séparé.

## 1. Où est le rognage, et pourquoi là

**Le rognage est dans `compteur/detecteur.py::Detecteur.detecter`, sur
l'image native, avant la redimension `imgsz`.**

```python
detections = self.detecteur.detecter(img, self._bande_pour(img))
```

Le rectangle est calculé par la **géométrie** (`Ligne.rect_bande_detection`,
dans `compteur/ligne.py`), jamais par la couche d'affichage. C'est la
frontière retenue :

| couche | sait ce qu'est une ligne ? | rôle |
|---|---|---|
| `compteur/ligne.py` | oui | décide DU rectangle (pure géométrie) |
| `compteur/compteur.py` | oui | détient la ligne, fournit le rectangle |
| `compteur/detecteur.py` | **non** | applique le rectangle qu'on lui donne |

`detecteur.py` n'importe rien de graphique — le test d'isolation existant
(`tests/test_isolation_ui.py`) continue de passer sur tout le paquet, donc la
frontière est tenue par le test, pas par une intention.

**Pourquoi dans le détecteur et pas dans le compteur.** Le compteur ne possède
pas le modèle ; le détecteur est le seul endroit où l'image est encore
native. Rogner plus haut cutterait la frame destinée à l'affichage — donc
l'opérateur verrait un rognage qui n'a pas eu lieu. Rogner plus bas
signifierait redimensionner toute l'image puis jeter 80 % du résultat : le même
travail, aucun gain.

**La correction des coordonnées.** Le modèle ne voit que la bande, ses boîtes
sont donc relatives à la coupe. `decalage_x`/`decalage_y` les ramènent dans le
repère de l'image pleine. Sans cela, le tracker verrait toutes les personnes
ramassées contre x=0 et le compteur les ferait passer à côté de la ligne — un
décompte faux **sans aucun signe à l'écran**, puisque l'affichage se fait lui
aussi en coordonnées d'image pleine. C'est la famille de bugs la plus
coûteuse du projet : invisible à l'œil, fausse au chiffrage. Le test
`test_la_bande_rogne_l_image_et_les_coordonnees_reviennent_justes` l'éprouve.

`np.ascontiguousarray` sur la coupe : un slicing numpy est une vue avec un
stride, qu'OpenCV refuse ou lit de travers.

## 2. Le rognage hors cadre

`Ligne.rect_bande_detection(largeur, hauteur)` fait deux choses :

- **Rogne proprement.** Chaque bord est ramené dans l'image :
  `x1 = max(0, floor(...))`, `x2 = min(largeur, ceil(...))`. Une ligne près du
  bord de gauche donne une bande qui commence à 0, pas à −80.

  C'est le seul comportement correct, pour une raison qui ne se voit pas :
  **un slicing numpy négatif est une indexation depuis la fin**. Sans cette
  correction, `img[-80:120]` ne rognerait pas du tout — il prendrait les 200
  derniers pixels. La bonne coupe par un pur hasard, et la mauvaise pour toute
  autre ligne.

- **Ligne hors cadre → `None`.** Si le rectangle nu ne recoupe pas l'image, on
  renvoie `None` et le détecteur analyse tout. C'est plus sûr qu'une bande
  dégénérée : un slicing de largeur nulle fait échouer le modèle, et il n'y a
  rien à gagner puisque personne ne franchira une ligne invisible.

Sur une diagonale, c'est le **rectangle englobant** des quatre coins de la
bande qui sert au rognage — plus grand que la bande elle-même. Compromis
assumé : un rognage d'image est droit, il ne peut pas être oblique ; on rogne
donc le rectangle qui *contient* la bande, jamais moins.

## 3. L'abandon des tracks après la ligne — l'ordre des purges

Une personne qui a franchi la ligne ne peut plus jamais générer de comptage :
`a_traverse` exige un côté de départ *strictement* positif
(`c_avant > 0 >= c_apres`). Ses entrées sont donc mortes.

`_purger_franchis()` vide `_derniers_centers`, `_historiques`, `_cotes`,
`_stabilite` — et **rien d'autre**.

**L'ordre, qui est la partie délicate.** Le verrou anti-rebond lit `_cotes` et
`_stabilite` (cf. le bloc de `traiter_frame`). La purge est donc appelée
**après** la boucle de comptage, jamais pendant : purger pendant la boucle
retirerait au verrou la moitié de son information de décision, au milieu même
de la frame.

Deux garde-fous supplémentaires :

- `_deja_comptes` n'est **pas** purgé. C'est le journal anti-recomptage, pas un
  état par track : le vider permettrait à un identifiant réapparu d'être compté
  comme une nouvelle personne.
- Le marquage des tracks arrivés se fait **avant** le `continue` des tracks
  déjà comptés. Placé après, ce sont eux — le cas majoritaire — qui
  conserveraient des entrées mortes.

Le test `test_abandon_apres_la_ligne_ne_change_pas_le_decompte` construit le
même scénario avec et sans la purge et exige des décomptes **identiques**.

## 4. Déplacement : un seul objet

La bande n'est pas une coordonnée stockée : `rect_bande_detection` la **dérive**
de `p1`/`p2` à chaque appel. Il n'y a donc rien à synchroniser — déplacer la
ligne déplace la bande, par construction.

- `Ligne.deplacer(p1, p2)` déplace les deux. Le verrou et la validité sont
  testés **avant** toute écriture : un déplacement refusé ne laisse pas une
  ligne à moitié déplacée.
- `verrouiller()` / `deverrouiller()` — posés au `lancer()`, levés par
  `arreter()` et `_reset_analyse()` (sans quoi, après une analyse, l'opérateur
  ne pourrait plus déplacer sa ligne).
- Bande et voile visibles **pendant le tracé** aussi (`_rafraichir_affichage`),
  pas seulement pendant l'analyse : c'est le seul moment où l'opérateur peut
  encore corriger son geste.
- Aucune passe de configuration pour la bande : il n'y a pas de réglage
  séparé.

## 5. Voile sombre à 40 %

`Ligne.voiler(img)` — dans `compteur/ligne.py`, factor `FACTEUR_VOILE = 0.6`
dans `config.py`.

L'ordre est **tout** : `interface/overlay.py` applique le voile sur l'image
**nue, avant** les boîtes de détection. Assombri après, il aplatit les boîtes
avec le reste et l'opérateur ne voit plus ce que le modèle a trouvé.

L'image reste visible (40 % d'ombre, pas un masque) : un masque noir ferait
perdre à l'opérateur le contexte — il ne verrait plus ce qu'il a placé.

## 6. Retrait de l'affichage

`label_details` (« Présents : N • Frames : N ») est supprimé. **Les valeurs
restent calculées** : `maj_compteurs` continue de mémoriser `presents` et
`frames`, `presents_max`/`presents_moyen` alimentent toujours le rapport et
l'export, et le tracker n'est pas touché. Seul le libellé a disparu.

Un seul test existant a dû être ajusté (`test_compteur_se_met_a_jour`, qui
assertait `"7" in label_details.text()`) : il vérifiait l'affichage, pas le
calcul.

## Mesure de contrôle

3000 frames consécutives de `D:\Bureau\manif_test.mp4`, ligne verticale x=640,
`sens=-1`, `imgsz=640`, torch 2.14.1+cu126 (CUDA actif, RTX 4070 SUPER) :

| configuration | compté | temps |
|---|---|---|
| image entière | 231 | 83 s |
| bande 200 px | **256** | 80 s |
| écart | **+10,8 %** | −4 % |

Reproduit la mesure de référence (231 → 256). Le temps ne varie pas de façon
significative : **le goulot n'est pas la détection**, comme annoncé.

## Tests

**495 passent, 3 xfailed.** Aucun xfail ajouté, aucun test désactivé, les 488
tests d'origine passent tous.

Ajoutés (`tests/test_bande_detection.py`, 5 tests) :
- la bande rogne l'image et les coordonnées reviennent dans le repère plein ;
- contre-test : sans bande, le modèle voit toute l'image ;
- abandon post-ligne : décompte identique avec et sans ;
- l'abandon vide bien les registres sans toucher au journal anti-recomptage ;
- la bande suit la ligne déplacée, et ne bouge plus quand elle est verrouillée ;
- le voile assombrit hors bande et laisse la bande intacte.

Trois tests existants ont été ajustés, **sans affaiblir ce qu'ils prouvent** :
leurs scénarios franchissaient la ligne, ce que la nouvelle purge rend vacuous.
`test_purge_des_registres_quand_le_tracker_perd_un_track` marche désormais
jusqu'à la ligne sans la franchir (il éprouve la perte de track, pas
l'abandon) ; `test_track_revenu_apres_une_occlusion...` revient du même côté
qu'il est parti ; `test_lissage_reste_rapide...` garde sa borne mémoire en
« au plus » au lieu d'une égalité.

## Préoccupations

1. **Le rognage diagonal est un rectangle englobant.** Sur une ligne à 45°, la
   bande utile est fine mais le rectangle rogné est un carré — le gain de
   détection s'y rapproche de celui de l'image entière. Le comptage reste
   correct. Acceptable si on ne fait que des lignes verticales/horizontales,
   ce qui est le cas d'usage (un cordon de manifestation).

2. **Pas de gain de temps mesuré** (−4 %, dans le bruit). La bande a été
   retenue pour le *décompte*, pas pour la vitesse. Si le goulot est le
   tracker, c'est là qu'il faudra regarder, pas ici.

3. **Le compteur ne purge pas le tracker.** `_purger_franchis` vide l'état du
   `Compteur`, pas les pistes du `Tracker`. Vider le tracker serait une
   optimisation de performance que la mesure ne justifie pas, et dont le
   risque (un identifiant réapparu compté comme une nouvelle personne) ne vaut
   pas le gain.

4. **Une migration de profil JSON n'est pas nécessaire** : la bande est une
   constante de module, pas un champ de `Config`. `config/default.json` est
   donc inchangé, et le test de synchronisation dataclass/fichier passe sans
   nouvelle clé.
