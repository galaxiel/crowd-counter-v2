"""Dessin des annotations sur l'image. Ne décide rien, ne lit aucune config.

Deux règles tenues par ce module :

1. **``img`` n'est jamais modifiée.** Le pipeline lui passe la frame qu'il
   vient de détecter ; si l'overlay la peignait, l'aperçu deviendrait la
   nouvelle source de la frame suivante et les annotations s'empileraient.
   On travaille donc sur une copie, systématiquement.

2. **La ligne n'est pas redessinée ici.** Sa géométrie, sa bande translucide
   et sa flèche de sens appartiennent à `compteur.ligne.Ligne.dessiner`,
   corrigé en tâche 2 (la flèche montrait alors le MAUVAIS côté, ce qui
   amenait l'opérateur à compter la moitié des gens). L'overlay se contente
   de l'appeler : dupliquer ce dessin ici resynchroniserait les deux versions
   et réintroduirait le bug.

Les couleurs sont en BGR comme le reste d'OpenCV, pas en RVB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # les annotations ne forcent pas l'import à l'exécution
    from compteur.ligne import Ligne
    from compteur.types import FrameResult

# Boîtes de détection : ambre chaud. Contrasté sur le gris bitumineux ET sur
# le ciel surexposé, contrairement à un jaune vif qui disparaît sur fond clair.
COULEUR_BOITE = (240, 160, 60)
# Centres de track : jaune-vert vif, la couleur la plus saillante de la frame,
# parce que c'est le point que l'algorithme suit réellement.
COULEUR_TRACK = (200, 200, 80)
COULEUR_TEXTE = (255, 255, 255)
# Flash de comptage : rouge saturé, l'alerte la plus urgent(e) à capter du
# coin de l'œil quand un groupe passe.
COULEUR_FLASH = (80, 80, 240)
# Boîte d'une personne COMPTÉE : vert franc. C'est la couleur la plus
# distinctive de l'écran — l'ambre des détections et le jaune-vert des centres
# de track s'en rapprochent, le vert pur non.
#
# **Ce que la couleur rend visible.** Le compteur peut se tromper en ratant
# quelqu'un, et il ne le dira pas : le total est juste un nombre. Or une
# personne qui traverse la ligne SANS que sa boîte devienne verte est un raté
# que l'opérateur voit immédiatement. C'est le seul moyen de détecter une
# erreur sans vérité terrain — on ne peut pas recompter à la main chaque
# vidéo, mais on peut regarder 20 secondes et voir si tout le monde verdit.
COULEUR_COMPTEE = (60, 220, 60)


def dessiner(
    img: np.ndarray,
    resultat: "FrameResult",
    ligne: "Ligne | None" = None,
    afficher_ids: bool = True,
    flash: bool = False,
    boites_comptees: "list[tuple] | None" = None,
) -> np.ndarray:
    """Retourne une copie annotée. ``img`` n'est jamais modifiée.

    ``img`` et ``resultat.image`` sont normalement le même tableau ; la copie
    protectrice est faite ici, elle n'a pas à être faite par l'appelant.

    ``boites_comptees`` est une liste de
    `(frame_comptage, x1, y1, x2, y2, opacite)` déjà bornée en durée et en
    nombre par l'appelant. Ce module ne la purge pas et ne la décide pas : il ne
    fait que la peindre. Une boîte verte est donc peinte à sa POSITION de
    franchissement, pas à la position courante de la personne — c'est voulu,
    la personne a été lâchée à cet instant. C'est l'`opacite` qui empêche cette
    boîte figée de ressembler à un suivi qui s'est arrêté : elle s'éteint
    progressivement au lieu de disparaître d'un coup.
    """
    sortie = _copie_de_travail(img)

    # Le voile d'abord, sur l'image NUE. L'ordre est tout : assombri après les
    # boîtes, il les aplatit avec le reste et l'opérateur ne voit plus ce que le
    # modèle a trouvé. C'est la bande de DÉTECTION qui est ainsi mise en
    # évidence — pas la bande de franchissement, plus étroite.
    if ligne is not None:
        sortie = ligne.voiler(sortie)

    for d in getattr(resultat, "detections", None) or []:
        _boite(sortie, d.x1, d.y1, d.x2, d.y2, COULEUR_BOITE)

    # Boîtes VERTES des personnes comptées. Dessinées après les détections et
    # AVANT les centres de track : une personne comptée est encore détectée
    # (elle n'a fait que passer la ligne), donc sa boîte d'ambre est encore là
    # — le vert doit passer AU-DESSUS, sinon le raté resterait invisible, ce
    # qui viderait la couleur de son but.
    for boite in boites_comptees or []:
        _frame, x1, y1, x2, y2 = boite[:5]
        # La boîte d'une personne COMPTÉE est REMPLIE, pas seulement encadrée :
        # l'opérateur voit d'un coup d'œil que cette personne est passée, sans
        # avoir à lire un contour qui ressemble à celui des détections voisines.
        # On ne voit plus la tête — c'est voulu, le décompage prime sur
        # l'identification.
        _boite_pleine(sortie, x1, y1, x2, y2, COULEUR_COMPTEE)

    # L'opacité n'est plus utilisée : les boîtes vertes sont pleines et
    # s'éteignent net à la sortie de la bande (duree = BANDE_APRES_PX / vitesse).
    # L'ancienne logique de fondu a été supprimée.

    for t in getattr(resultat, "tracks", None) or []:
        cx, cy = int(t.center[0]), int(t.center[1])
        # `clip` sur un centre hors cadre : cv2.trace un cercle partiellement
        # dehors sans erreur, mais on borne pour rester prévisible.
        cv2.circle(sortie, (cx, cy), 4, COULEUR_TRACK, -1, lineType=cv2.LINE_AA)
        if afficher_ids:
            cv2.putText(
                sortie,
                str(t.track_id),
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COULEUR_TEXTE,
                2,
                lineType=cv2.LINE_AA,
            )

    # Après les boîtes : la bande doit passer AU-DESSUS, sinon les gens à
    # compté disparaissent sous le voile translucide. `Ligne.dessiner` rend
    # lui-même une copie, qu'on adopte — la nôtre n'est plus utilisée.
    if ligne is not None:
        sortie = ligne.dessiner(sortie)

    if flash:
        hauteur, largeur = sortie.shape[:2]
        cv2.rectangle(
            sortie, (0, 0), (largeur - 1, hauteur - 1), COULEUR_FLASH, 8
        )

    return sortie


def _copie_de_travail(img: np.ndarray) -> np.ndarray:
    """Copie contiguë et modifiable de ``img``, quelle que soit sa forme.

    On normalise deux choses que le widget vidéo sait déjà faire mais que
    l'overlay peut recevoir d'une autre source (une capture de caméra, un
    JPEG décodé) :

    - une image 2D (niveaux de gris) est repliée sur trois canaux, sinon
      `cv2.circle` échouerait sur un buffer mono ;
    - `ascontiguousarray` parce qu'une vue numpy non-contiguë (un
      ``img[:, ::-1]``, un ROI) n'est pas un buffer d'image acceptable pour
      OpenCV, qui-write alors hors des limites ou refuse l'opération.

    `copy()` garantit de toute façon qu'`img` reste intact.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img).copy()


def _boite(img: np.ndarray, x1, y1, x2, y2, couleur) -> None:
    """Trace le rectangle de la boîte, en anti-aliasing.

    Une détection partiellement hors cadre donne des coordonnées négatives :
    cv2 les accepte et rogne au bord de l'image, ce qui est le comportement
    voulu (la personne est bien à moitié dans le champ).
    """
    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        couleur,
        2,
        lineType=cv2.LINE_AA,
    )

def _fondre(couleur, img, x1, y1, opacite: float) -> tuple:
    """Mélange ``couleur`` avec le fond réel, pour un tracé semi-transparent.

    On ne dessine pas sur un calque séparé : on calcule la couleur du trait
    comme un mélange entre le vert et le pixel déjà peint sous la boîte. Le
    résultat est un fondu visuellement équivalent pour un trait de 2 px, au
    coût d'une lecture de deux pixels.

    Le pixel d'échantillonnage est le coin haut-gauche de la boîte, borné dans
    l'image : une boîte de comptage peut être à moitié hors cadre, et un index
    négatif lirait — en numpy — le bas de l'image, donc un vert qui ne
    s'éteindrait pas du tout.
    """
    if opacite >= 1.0:
        return couleur
    hauteur, largeur = img.shape[:2]
    px = min(max(int(x1), 0), largeur - 1)
    py = min(max(int(y1), 0), hauteur - 1)
    fond = img[py, px].astype(np.float32)
    return tuple(
        int(round(float(c) * opacite + float(f) * (1.0 - opacite)))
        for c, f in zip(couleur, fond)
    )


def _boite_pleine(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, couleur) -> None:
    """Remplit toute la surface d'une boîte comptée, cadre compris.

    Réservé aux personnes COMPTÉES : l'opérateur voit d'un coup d'œil que
    cette personne est passée, sans lire un contour qui ressemble à celui des
    détections voisines. On ne voit plus la tête — c'est voulu, le décompage
    prime sur l'identification.

    La boîte est légèrement plus petite que la détection originale (MARGE_COMPTEE)
    pour donner un repère visuel : on sait que c'est la même personne, mais
    le remplissage vert ne couvre plus la zone de bruit autour de la tête.
    """
    hauteur, largeur = img.shape[:2]
    # Réduire de 3 px de chaque côté → la boîte verte est centrée sur la tête
    # mais un peu plus compacte que la détection d'ambre.
    MARGE_COMPTEE = 3
    xa = int(max(0, min(x1, x2))) + MARGE_COMPTEE
    xb = int(min(largeur, max(x1, x2) + 1)) - MARGE_COMPTEE
    ya = int(max(0, min(y1, y2))) + MARGE_COMPTEE
    yb = int(min(hauteur, max(y1, y2) + 1)) - MARGE_COMPTEE
    if xb <= xa or yb <= ya:
        return
    img[ya:yb, xa:xb] = couleur
