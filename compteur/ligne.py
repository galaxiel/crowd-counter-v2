"""Géométrie de la ligne de franchissement.

Tout le raisonnement « cette personne est-elle passée de l'autre côté, dans le
bon sens ? » tient dans ce fichier. Il ne dépend d'aucun autre module métier,
ce qui le rend testable de façon exhaustive.
"""

from __future__ import annotations

import math

import numpy as np

from .config import BANDE_APRES_PX, BANDE_AVANT_PX, FACTEUR_VOILE

#: Orientation d'une ligne, telle que l'interface la nomme.
#:
#: Elle est déduite de l'axe DOMINANT du segment p1 -> p2, jamais d'un réglage :
#: l'opérateur trace ce qu'il voit sur l'image, l'interface doit donc nommer ce
#: qu'il a tracé. Une diagonale est nommée d'après son axe dominant — le libellé
#: décrit alors le mouvement dominant, et la flèche reste l'affordance non
#: ambiguë.
ORIENTATION_VERTICALE = "verticale"
ORIENTATION_HORIZONTALE = "horizontale"

#: Libellés du sens compté, par orientation. L'ordre est FIXE et fait autorité :
#: l'entrée 0 est toujours le libellé du sens ``+1``, l'entrée 1 celui du sens
#: ``-1``. « Gauche -> droite » avant « Droite -> gauche », « Haut -> bas »
#: avant « Bas -> haut » : dans les deux cas, le libellé d'abord est celui du
#: mouvement vers les coordonnees CROISSANTES (x vers la droite, y vers le bas
#: de l'image). Un test verrouille cet ordre : le deranger ferait dire au menu
#: une chose et au compteur une autre.
LIBELLES_SENS: dict[str, tuple[str, str]] = {
    ORIENTATION_VERTICALE: ("Gauche → droite", "Droite → gauche"),
    ORIENTATION_HORIZONTALE: ("Haut → bas", "Bas → haut"),
}


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
        bande_avant_px: int | None = None,
        bande_apres_px: int | None = None,
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
        for valeur, nom in (
            (bande_avant_px, "la bande avant"),
            (bande_apres_px, "la bande après"),
        ):
            if valeur is not None and valeur < 1:
                raise ValueError(f"{nom} doit faire >= 1 px")

        self.p1 = p1
        self.p2 = p2
        self.epaisseur = int(epaisseur)
        self.sens = int(sens)
        # Valide et exposé pour la tache 5 (anti-rebond du comptage) qui le lit
        # via ligne.hysteresis ; la geometrie de ce module ne s'en sert pas.
        self.hysteresis = int(hysteresis)
        # Profondeur de la bande de détection de CHAQUE côté de la ligne, en
        # pixels de l'image native. PAS deux réglages séparés : ce sont des
        # attributs de la ligne, elle se déplace donc AVEC elle et ils ne
        # peuvent pas en être détachés.
        #
        # La bande est DISSYMÉTRIQUE : `bande_avant_px` (côté de DÉPART,
        # coordonnée positive sur la normale — voir `point_du_cote`) est plus
        # large que `bande_apres_px` (côté d'arrivée). Le tracker a besoin
        # d'espace pour construire une identité AVANT le franchissement ; après,
        # le compte est fait et 100 px suffisent à voir la personne passer.
        #
        # `None` des deux côtés signifie « pas de bande » : le modèle voit
        # l'image entière. C'est le seul cas qui désactive le rognage — un
        # `None` d'un seul côté serait une bande d'un seul côté, ce qui n'a pas
        # de sens et rendrait le franchissement invisible au modèle.
        self.bande_avant_px = None if bande_avant_px is None else int(bande_avant_px)
        self.bande_apres_px = None if bande_apres_px is None else int(bande_apres_px)
        if (self.bande_avant_px is None) != (self.bande_apres_px is None):
            raise ValueError(
                "la bande de détection se règle des deux côtés ou pas du tout"
            )
        # Verrou de l'opérateur : une fois l'analyse lancée, la ligne et sa
        # bande cessent de bouger. Sans cela, un réglage glissé pendant la
        # lecture déplacerait la zone de comptage sous les yeux de celui qui
        # regarde, et le décompte afficherait deux zones différentes dans la
        # même vidéo.
        self.verrouillee = False
        self._recalculer_geometrie()


    def _recalculer_geometrie(self) -> None:
        """Recalcule les vecteurs dérivés de ``p1``/``p2``.

        Isolé du constructeur parce que `deplacer` doit produire exactement la
        même géométrie : c'est le seul moyen de garantir qu'une ligne
        reconstruite et une ligne déplacée ne divergent pas sur un cas limite.
        """
        longueur = math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
        # Direction le long de la ligne, puis normale (perpendiculaire).
        self._d = (
            (self.p2[0] - self.p1[0]) / longueur,
            (self.p2[1] - self.p1[1]) / longueur,
        )
        self._n = (-self._d[1], self._d[0])
        # Le signe de la normale est choisi par `sens`, mais ce n'est PAS pour
        # autant le sens de comptage : il sert à distinguer les deux côtés.
        if self.sens < 0:
            self._n = (-self._n[0], -self._n[1])
        # Direction effectivement comptée. a_traverse retient le passage
        # quand la coordonnée sur _n décroît : le sens compté est donc
        # opposé à _n quand sens=+1. Exposer cet invariant évite que la
        # flèche et l'interface se trompent de sens.
        self.sens_traversee = (-self._n[0], -self._n[1])

    def deplacer(self, p1, p2) -> bool:
        """Déplace la ligne — et sa bande avec elle, par construction.

        La bande n'est pas une coordonnée stockée mais une grandeur DÉRIVÉE de
        ``p1``/``p2`` (voir `rect_bande_detection`) : elle ne peut pas être
        désolidarisée de la ligne. Déplacer la ligne déplace la bande ; voilà
        ce que « un seul objet » veut dire.

        Renvoie `False` sans rien changer si la ligne est verrouillée, ou si
        les deux points sont confondus. Le verrou et la validité sont testés
        AVANT toute écriture : un déplacement refusé ne doit pas laisser une
        ligne à moitié déplacée.
        """
        if self.verrouillee:
            return False
        q1 = (float(p1[0]), float(p1[1]))
        q2 = (float(p2[0]), float(p2[1]))
        if math.hypot(q2[0] - q1[0], q2[1] - q1[1]) < 1e-6:
            return False
        self.p1 = q1
        self.p2 = q2
        self._recalculer_geometrie()
        return True

    def verrouiller(self) -> None:
        """Fige la ligne et sa bande pour la durée de l'analyse."""
        self.verrouillee = True

    def deverrouiller(self) -> None:
        """Libre la ligne : l'opérateur peut la retracer."""
        self.verrouillee = False

    # -- Bande de détection -----------------------------------------------

    def rect_bande_detection(
        self, largeur_image: int, hauteur_image: int
    ) -> tuple[int, int, int, int] | None:
        """Rectangle ``(x1, y1, x2, y2)`` de la bande, en pixels de l'image.

        `None` signifie « pas de bande » : le modèle voit l'image entière.

        **C'est ici que se décide le rognage, et la décision est géométrique,
        pas graphique.** `compteur.detecteur` ne sait pas ce qu'est une ligne ;
        il reçoit un rectangle. Cette séparation est vérifiée par un test
        d'isolation.

        **La bande est DISSYMÉTRIQUE.** Elle s'étend de `bande_avant_px` sur le
        côté de DÉPART (coordonnée positive sur la normale) à `bande_apres_px`
        sur le côté d'ARRIVÉE — 200 / 100 px en production, donc 300 px au
        total et non 200.

        **Quel côté est « avant », et pourquoi c'est le côté de départ.** Le
        vocabulaire est celui de `point_du_cote` : côté de départ = coordonnée
        **positive**, côté d'arrivée = coordonnée **négative**, quel que soit
        `sens`. Les gens arrivent par le côté de départ ; c'est là que le
        tracker a besoin d'espace pour CONSTUIRE une identité fiable sur
        plusieurs frames. Après le franchissement, le compte est fait et cette
        personne ne peut plus jamais compter (`a_traverse` exige un côté de
        départ strictement positif) : il ne reste qu'à la voir s'éloigner, ce
        que 100 px suffisent à montrer.

        **L'implémentation est donc entièrement en coordonnées projetées.** On
        décale chaque extrémité du segment de `+avant` et de `-apres` le long de
        la NORMALE — pas « vers la gauche » / « vers la droite ». Le signe des
        deux décalages est fixe, et c'est ce qui rend la bande dissymétrique
        dans le bon sens quel que soit le `sens` : un décalage « vers -n »
        donnerait une bande inversée, donc 100 px avant et 200 px après pour
        `sens=-1`, exactement à l'opposé de ce qu'on veut.

        Elle est orientée par la NORMALE, donc elle suit une ligne diagonale
        aussi bien qu'une verticale — c'est le rectangle englobant qui sert au
        rognage, plus grand que la bande elle-même sur une diagonale.
        Compromis assumé : un rognage d'image est droit, il ne peut pas être
        oblique ; on rogne donc le rectangle qui CONTIENT la bande, jamais
        moins.

        **Rogne, ne déborde pas.** Chaque bord est ramené dans l'image : une
        ligne près du bord de gauche donne une bande qui commence à 0, pas une
        bande qui commence à -80. C'est le seul comportement correct, parce
        qu'un slicing numpy négatif est une indexation depuis la FIN : sans
        cette correction, ``img[-80:120]`` ne rognerait pas du tout, il
        prendrait les 200 derniers pixels — la bonne coupe par un pur hasard,
        et la mauvaise pour toute autre ligne.

        **Ligne hors cadre :** si le rectangle nu est entièrement hors de
        l'image, on renvoie `None` et le modèle analyse tout. C'est plus sûr que
        de rendre une bande dégénérée : un slicing de largeur nulle fait échouer
        le modèle, et il n'y a rien à gagner puisque personne ne franchira une
        ligne invisible.
        """
        if self.bande_avant_px is None:
            return None
        largeur_image = int(largeur_image)
        hauteur_image = int(hauteur_image)
        if largeur_image <= 0 or hauteur_image <= 0:
            return None

        # Les deux extrémités du segment, décalées de +avant et -apres sur la
        # normale : ce sont les deux bords de la bande. On prend le rectangle
        # englobant de ces quatre coins.
        nx, ny = self._n
        coins = [
            (px + nx * decalage, py + ny * decalage)
            for decalage in (float(self.bande_avant_px), -float(self.bande_apres_px))
            for (px, py) in (self.p1, self.p2)
        ]

        # Recouvrement avec l'image. `None` si la bande est entièrement dehors.
        x1 = max(0, int(math.floor(min(c[0] for c in coins))))
        x2 = min(largeur_image, int(math.ceil(max(c[0] for c in coins))))
        y1 = max(0, int(math.floor(min(c[1] for c in coins))))
        y2 = min(hauteur_image, int(math.ceil(max(c[1] for c in coins))))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)


    # -- Géométrie -------------------------------------------------------

    @property
    def orientation(self) -> str:
        """``"verticale"`` ou ``"horizontale"``, d'après l'axe dominant.

        C'est cette attribute qui permet à l'interface de nommer ce que
        l'opérateur a réellement tracé : une ligne verticale se compte de
        gauche à droite ou de droite à gauche, une ligne horizontale de haut
        en bas ou de bas en haut.

        Le critère est l'axe DOMINANT du segment, et non « dx == 0 » : une
        ligne légèrement penchée reste ce que l'opérateur a tracé de ses yeux,
        et un seuil exact la classerait « horizontale » ou « verticale » au
        choix selon un pixel. En cas d'égalité parfaite (diagonale à 45°), la
        verticale l'emporte — et comme les deux composantes sont alors non
        nulles, le libellé reste défini dans les deux cas.
        """
        dx = abs(self.p2[0] - self.p1[0])
        dy = abs(self.p2[1] - self.p1[1])
        return ORIENTATION_HORIZONTALE if dx > dy else ORIENTATION_VERTICALE

    def _sens_traversee_pour(self, sens: int) -> tuple[float, float]:
        """Vecteur de marche compté pour un `sens` QUELCONQUE.

        `sens_traversee` ne décrit que le sens de CETTE instance. Pour nommer
        les deux entrées du réglage, il faut la même quantité pour `+1` et pour
        `-1` : on la recalcule donc ici, à partir de la même formule que le
        constructeur, au lieu de reconstruire une `Ligne` par entrée.
        """
        signe = 1.0 if sens > 0 else -1.0
        return (self._d[1] * signe, -self._d[0] * signe)

    def libelle_sens(self, sens: int) -> str:
        """Nom à l'écran du mouvement compté pour `sens`, en clair.

        « Gauche → droite » / « Droite → gauche » sur une ligne verticale,
        « Haut → bas » / « Bas → haut » sur une ligne horizontale : le libellé
        est déduit de la GÉOMÉTRIE de la ligne tracée, pas d'un sens abstrait.
        C'est indispensable parce que le résultat dépend des deux : sur une
        ligne verticale tracée de bas en haut, `sens=+1` compte déjà de droite
        à gauche, alors que sur la même ligne tracée de haut en bas il compte
        de gauche à droite. Un libellé fixé sur le seul `sens` dirait donc la
        chose inverse à ce que fait le compteur.

        Le composant utilisé est celui de l'axe DOMINANT, jamais un pixel près
        de zéro : sur une diagonale, la composante transverse n'est jamais nulle
        non plus, mais c'est le mouvement dominant qui mérite d'être nommé.
        """
        if sens not in (1, -1):
            raise ValueError("sens doit valoir +1 ou -1")
        x, y = self._sens_traversee_pour(sens)
        if self.orientation == ORIENTATION_VERTICALE:
            return LIBELLES_SENS[ORIENTATION_VERTICALE][0 if x > 0 else 1]
        return LIBELLES_SENS[ORIENTATION_HORIZONTALE][0 if y > 0 else 1]

    def choix_sens(self) -> list[tuple[str, int]]:
        """Les deux choix du réglage, dans l'ordre d'affichage : ``(libellé, sens)``.

        L'ordre est celui de `LIBELLES_SENS` — « Gauche → droite » avant
        « Droite → gauche », « Haut → bas » avant « Bas → haut » — quelle que
        soit la direction de tracé. L'interface doit garder la même liste
        quelle que soit la façon dont la ligne a été tracée : c'est ce qui
        permet à l'opérateur de retrouver le même réglage d'une vidéo à
        l'autre.
        """
        paires = [(self.libelle_sens(s), s) for s in (1, -1)]
        ordre = LIBELLES_SENS[self.orientation]
        return sorted(paires, key=lambda paire: ordre.index(paire[0]))

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

    def voiler(self, img: np.ndarray) -> np.ndarray:
        """Assombrit HORS de la bande de détection. Ne mute pas ``img``.

        L'opérateur voit ainsi que seule la colonne compte, sans que la scène
        devienne illisible : c'est un VOILE (facteur 0,6 = 40 % d'ombre), pas
        un masque. Un masque noir serait plus net mais ferait perdre à
        l'opérateur le contexte — il ne verrait plus ce qu'il a placementé.

        **Appelé AVANT toute boîte de détection.** L'ordre est la seule chose
        qui compte ici : assombrir après dessiné aplatit les boîtes elles aussi,
        et l'opérateur ne voit plus ce que le modèle a trouvé. `overlay.dessiner`
        applique donc le voile en premier, sur l'image nue.

        Sans bande (`bande_avant_px is None`, ou bande entièrement hors cadre)
        l'image est rendue telle quelle : il n'y a rien à mettre en évidence,
        et noircir l'écran entier n'indiquerait aucune zone.
        """
        hauteur, largeur = img.shape[:2]
        rect = self.rect_bande_detection(largeur, hauteur)
        if rect is None:
            return img
        x1, y1, x2, y2 = rect
        # Copie obligatoire AVANT toute écriture : `img` appartient à l'appelant
        # et sert de source à la frame suivante. Le slicing produit une VUE
        # numpy, écrire dedans modifierait l'original — un bug invisible à
        # l'écran mais qui corromprait la vidéo analysée.
        sortie = img.astype(np.float32, copy=True) * FACTEUR_VOILE
        sortie[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        # L'arrondi est fait UNE fois, à la fin : arrondir puis recoller la
        # bande laisserait un liseré d'une nuance au bord de la zone.
        return np.clip(sortie, 0, 255).astype(img.dtype)

    def dessiner(self, img: np.ndarray) -> np.ndarray:
        """Trace ligne, bande et flèche de sens. Ne mute pas ``img``.

        Le voile de la bande de détection n'est PAS appliqué ici : ce serait
        trop tard, les boîtes de détection sont déjà dessinées par
        `overlay.dessiner` et seraient donc assombries avec le reste. Le voile
        passe par `voiler`, appelé en amont.
        """
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
