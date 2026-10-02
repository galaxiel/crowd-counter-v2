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

    La ligne passe par ``p1`` et ``p2``. ``sens`` désigne le **côté
    retenu** : ``+1`` on ne compte que les personnes qui vont du côté de
    départ vers le côté d'arrivée, ``-1`` que les passages inverses.

    Deux vecteurs portent donc des noms distincts, et il ne faut pas les
    confondre :

    - ``vecteur_normal()`` / ``_n`` : la normale. Elle sert a mesurer la
      position signée d'un point (``coordonnee_projetee``) et a orienter les
      deux côtés de la bande. Son signe n'est PAS le sens de comptage.
    - ``sens_traversee`` : la direction de marche réellement comptée, soit
      toujours ``-_n``. C'est elle que la flèche de ``dessiner()`` doit
      montrer à l'opérateur.

    Le vocabulaire des côtés est unique dans ce module et vient du
    comportement testé de ``a_traverse`` : **côté de départ = coordonnée
    positive**, **côté d'arrivée = coordonnée négative**. Ce qui est donc
    vrai quel que soit ``sens``.
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
        # Le signe de la normale estChoisi par `sens`, mais ce n'est PAS pour
        # autant le sens de comptage : il sert a distinguer les deux cotes.
        if self.sens < 0:
            self._n = (-self._n[0], -self._n[1])
        # Direction effectivement comptée. a_traverse retient le passage
        # quand la coordonnée sur _n décroît : le sens compté est donc
        # opposé à _n quand sens=+1. Exposer cet invariant évite que la
        # flèche et l'interface se trompent de sens.
        self.sens_traversee = (-self._n[0], -self._n[1])

    # -- Géométrie -------------------------------------------------------

    def vecteur_normal(self) -> tuple[float, float]:
        """Normale unitaire : sert a situer les deux côtés, pas a dire le sens compté.

        Pour la direction de marche à montrer à l'opérateur, lire
        ``sens_traversee`` (``-_n``).
        """
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
        """Le côté de ``p`` : ``+1`` départ, ``-1`` arrivée, ``0`` dans la bande.

        Noms pris dans ``a_traverse``, qui est le comportement réel et testé :
        **côté de départ = coordonnée positive**, **côté d'arrivée =
        coordonnée négative**. Ces noms ne dépendent donc PAS de ``sens`` —
        avec ``sens=+1`` comme avec ``sens=-1``, ``+1`` désigne le côté de
        départ. Ce qui change avec ``sens``, c'est le côté retenu.

        La règle seule, sans interprétation : ``+1`` si
        ``coordonnee_projetee(p) > epaisseur / 2``, ``-1`` si elle est
        ``< -epaisseur / 2``, ``0`` entre les deux.
        """
        c = self.coordonnee_projetee(p)
        if c > self.epaisseur / 2.0:
            return 1
        if c < -self.epaisseur / 2.0:
            return -1
        return 0

    def a_traverse(self, avant: tuple[float, float], apres: tuple[float, float]) -> bool:
        """Le passage de ``avant`` a ``apres`` est-il un franchissement retenu ?

        Retenu signifie : départ du côté de départ (coordonnée positive) vers
        le côté d'arrivée (coordonnée négative) le long du segment dessine.
        La direction de marche ainsi comptée est ``sens_traversee``, soit
        l'opposé de la normale : c'est donc ``-_n``, pas ``_n``.

        On interpole le point exact de croisement : peu importe la vitesse de
        la personne, qu'elle traverse lentement ou qu'elle saute par-dessus la
        bande, la reponse est la meme.
        """
        c_avant = self.coordonnee_projetee(avant)
        c_apres = self.coordonnee_projetee(apres)
        # Intervalle semi-ouvert : la ligne (c == 0) appartient au cote
        # d'arrivee. Une traversee est donc « cote depart > 0 et
        # cote arrive <= 0 ». Ce choix compte exactement une fois une
        # personne qui pose le pied pile sur la ligne, la ou une comparaison
        # stricte la ferait disparaitre, et deux fois si on ouvrait les deux
        # bouts. Le cote depart est strict, ce qui garantit qu'une personne
        # qui fait demi-tour APRES avoir franchi n'est pas recomptee : elle
        # arrive cote negatif, son retour vers le cote positif n'est pas
        # retenu (l'aller-retour compte 1, pas 2).
        #
        # En revanche ce module ne garantit RIEN sur les tremblements : il
        # ne voit qu'une suite de coordonnees et ne distingue pas une
        # personne qui s'eloigne d'une personne qui oscille sur place. Mesure
        # faite : osciller de +/-1 px autour de la ligne donne 19 traversees
        # retenues en 39 frames, et une personne immobile a 1 px (alternance
        # x=99/100) en donne 14 en 29 frames. C'est a l'anti-rebond (_deja_comptes,
        # tache 5) qu'il revient de ne pas recompter la meme personne, pas a
        # cette geometrie. Ne pas ecrire ici de garantie sur les tremblements.
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
        # La bande est un APERCU de la zone de comptage, pas un voile : peinte
        # opaque elle masquerait exactement les gens qu'on cherche a compter
        # (sur fond blanc, un pixel sous la bande tombait a 40 au lieu de 255).
        # On la compose donc en translucide ; l'original reste lisible dessous.
        bande = sortie.copy()
        cv2.fillPoly(bande, [quad], (40, 40, 40))
        sortie = cv2.addWeighted(sortie, 0.65, bande, 0.35, 0)
        cv2.polylines(sortie, [quad], True, jaune, 1, cv2.LINE_AA)
        cv2.line(sortie, p1, p2, vert, 2, cv2.LINE_AA)

        # Flèche de sens, au milieu du segment. Elle montre le sens
        # RÉELLEMENT compté, donc `sens_traversee` (= -_n) et non `_n` :
        # c'est l'affordance qui dit à l'opérateur de quel côté on compte.
        # Utiliser `_n` ici afficherait la flèche du mauvais côté.
        sx, sy = self.sens_traversee
        mx = (self.p1[0] + self.p2[0]) / 2.0
        my = (self.p1[1] + self.p2[1]) / 2.0
        pointe = (int(mx + sx * demi * 1.6), int(my + sy * demi * 1.6))
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
