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


def dessiner(
    img: np.ndarray,
    resultat: "FrameResult",
    ligne: "Ligne | None" = None,
    afficher_ids: bool = True,
    flash: bool = False,
) -> np.ndarray:
    """Retourne une copie annotée. ``img`` n'est jamais modifiée.

    ``img`` et ``resultat.image`` sont normalement le même tableau ; la copie
    protectrice est faite ici, elle n'a pas à être faite par l'appelant.
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