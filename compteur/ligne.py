"""Géométrie de la ligne de franchissement.

Tout le raisonnement « cette personne est-elle passée de l'autre côté, dans le
bon sens ? » tient dans ce fichier. Il ne dépend d'aucun autre module métier,
ce qui le rend testable de façon exhaustive.
"""

from __future__ import annotations

import math

import numpy as np


class Ligne:
    """Ligne orientée définissant une zone de franchissement.

    La ligne passe par ``p1`` et ``p2``. ``sens`` vaut ``+1`` : on ne compte
    que les passages dans le sens de la normale, ``-1`` : que les passages
    inverses.
    """

    def __init__(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        epaisseur: int = 30,
        sens: int = 1,
        hysteresis: int = 2,
    ) -> None:
        p1 = (float(p1[0]), float(p1[1]))
        p2 = (float(p2[0]), float(p2[1]))
        longueur = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if longueur < 1e-6:
            raise ValueError("les deux points de la ligne sont confondus")
        if sens not in (1, -1):
            raise ValueError("sens doit valoir +1 ou -1")
        if epaisseur < 1:
            raise ValueError("l'épaisseur doit être >= 1")
        if hysteresis < 0:
            raise ValueError("l'hystérésis doit être >= 0")

        self.p1 = p1
        self.p2 = p2
        self.epaisseur = int(epaisseur)
        self.sens = int(sens)
        # Valide et expose pour la tache 5 (anti-rebond du comptage) qui le lit
        # via ligne.hysteresis ; la geometrie de ce module ne s'en sert pas.
        self.hysteresis = int(hysteresis)

        # Direction le long de la ligne, puis normale (perpendiculaire).
        self._d = ((p2[0] - p1[0]) / longueur, (p2[1] - p1[1]) / longueur)
        self._n = (-self._d[1], self._d[0])
        # La normale est orientée dans le sens de traversée retenu.
        if self.sens < 0:
            self._n = (-self._n[0], -self._n[1])

    # -- Géométrie -------------------------------------------------------

    def vecteur_normal(self) -> tuple[float, float]:
        """Normale unitaire, orientée dans le sens de traversée retenu."""
        return self._n

    def coordonnee_projetee(self, p: tuple[float, float]) -> float:
        """Position signée de ``p`` sur la normale, en pixels (0 = sur la ligne)."""
        dx = float(p[0]) - self.p1[0]
        dy = float(p[1]) - self.p1[1]
        return dx * self._n[0] + dy * self._n[1]

    def est_dans_la_bande(self, p: tuple[float, float]) -> bool:
        return abs(self.coordonnee_projetee(p)) <= self.epaisseur / 2.0

    def parametre_along(self, p: tuple[float, float]) -> float:
        """Position de ``p`` projetée sur l'axe de la ligne, en pixels depuis ``p1``."""
        dx = float(p[0]) - self.p1[0]
        dy = float(p[1]) - self.p1[1]
        return dx * self._d[0] + dy * self._d[1]

    def contient(self, p: tuple[float, float]) -> bool:
        """``p`` est-il dans la bande ET le long du segment (pas à l'infini) ?"""
        if not self.est_dans_la_bande(p):
            return False
        t = self.parametre_along(p)
        longueur = math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
        return -self.epaisseur / 2.0 <= t <= longueur + self.epaisseur / 2.0

    def point_du_cote(self, p: tuple[float, float]) -> int:
        """``+1`` cote de la normale POSITIVE, ``-1`` cote negatif, ``0`` dans la bande.

        Ces noms de cotes sont relatifs a la normale, qui est elle-meme
        orientee par ``sens`` : « +1 » designe donc le cote d'arrivee quand
        ``sens = +1``, et le cote de depart quand ``sens = -1``. La regle
        seule, sans interpretation : ``+1`` si ``coordonnee_projetee(p) >
        epaisseur / 2``, ``-1`` si elle est ``< -epaisseur / 2``, ``0``
        entre les deux.
        """
        c = self.coordonnee_projetee(p)
        if c > self.epaisseur / 2.0:
            return 1
        if c < -self.epaisseur / 2.0:
            return -1
        return 0

    def a_traverse(self, avant: tuple[float, float], apres: tuple[float, float]) -> bool:
        """Le passage de ``avant`` a ``apres`` est-il un franchissement retenu ?

        Une traversee est un changement de cote de la ligne, ET le croisement
        doit avoir lieu le long du segment dessine. On interpole le point
        exact de croisement : peu importe la vitesse de la personne, qu'elle
        traverse lentement ou qu'elle saute par-dessus la bande, la reponse
        est la meme.
        """
        c_avant = self.coordonnee_projetee(avant)
        c_apres = self.coordonnee_projetee(apres)
        # Intervalle semi-ouvert : la ligne (c == 0) appartient au cote
        # d'arrivee. Une traversee est donc « cote depart > 0 et
        # cote arrive <= 0 ». Ce choix compte exactement une fois une
        # personne qui pose le pied pile sur la ligne, la ou une comparaison
        # stricte la ferait disparaitre, et deux fois si on ouvrait les deux
        # bouts. Le cote depart est strict : un simple tremblement autour de
        # la ligne sans franchissement n'est pas compte.
        if not (c_avant > 0 >= c_apres):
            return False

        # Position du croisement, interpolee entre les deux points.
        t = c_avant / (c_avant - c_apres)
        croisement = (
            avant[0] + (apres[0] - avant[0]) * t,
            avant[1] + (apres[1] - avant[1]) * t,
        )
        s = self.parametre_along(croisement)
        longueur = math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
        return -self.epaisseur / 2.0 <= s <= longueur + self.epaisseur / 2.0

    # -- Rendu -----------------------------------------------------------

    def dessiner(self, img: np.ndarray) -> np.ndarray:
        """Trace ligne, bande et flèche de sens. Ne mute pas ``img``."""
        sortie = img.copy()
        p1 = (int(self.p1[0]), int(self.p1[1]))
        p2 = (int(self.p2[0]), int(self.p2[1]))
        vert = (80, 220, 80)
        jaune = (60, 200, 255)

        # La bande est un quadrilatère épaissi perpendiculairement.
        nx, ny = self._n
        demi = self.epaisseur / 2.0
        quad = np.array(
            [
                [int(self.p1[0] - nx * demi), int(self.p1[1] - ny * demi)],
                [int(self.p2[0] - nx * demi), int(self.p2[1] - ny * demi)],
                [int(self.p2[0] + nx * demi), int(self.p2[1] + ny * demi)],
                [int(self.p1[0] + nx * demi), int(self.p1[1] + ny * demi)],
            ],
            dtype=np.int32,
        )
        cv2 = _cv2()
        cv2.fillPoly(sortie, [quad], (40, 40, 40))
        cv2.polylines(sortie, [quad], True, jaune, 1, cv2.LINE_AA)
        cv2.line(sortie, p1, p2, vert, 2, cv2.LINE_AA)

        # Flèche de sens, au milieu du segment.
        mx = (self.p1[0] + self.p2[0]) / 2.0
        my = (self.p1[1] + self.p2[1]) / 2.0
        pointe = (int(mx + nx * demi * 1.6), int(my + ny * demi * 1.6))
        base = (int(mx), int(my))
        cv2.arrowedLine(sortie, base, pointe, vert, 3, cv2.LINE_AA, tipLength=0.4)
        return sortie


def _cv2():
    """Importe OpenCV à la demande.

    ``compteur.ligne`` reste ainsi importable sans OpenCV : les tests de
    géométrie pure n'ont pas besoin du moteur de rendu.
    """
    import cv2

    return cv2
