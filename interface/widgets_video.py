"""Affichage de l'image annotée, avec émission des clics pour tracer la ligne.

C'est le module où se joue la cohérence entre ce que l'opérateur VOIT et ce que
le moteur COMPTE. L'opérateur pose la ligne à la souris en regardant la scène ;
le clic est remonté en coordonnées de l'image source, et c'est cette valeur
que `compteur.ligne` transforme en ligne de comptage. Si la conversion
écran -> pixel est fausse, la ligne affichée a l'air correctement posée — le
pixel visé est bien sous le curseur — mais la ligne comptée, elle, est décalée.
Le défaut est donc INVISIBLE à l'écran et ne se révèle qu'en comptant faux.

La géométrie retenue :

- l'image est mise à l'échelle par un facteur UNIQUE, le plus petit des deux
  ratios ``largeur_widget / largeur_image`` et ``hauteur_widget / hauteur_image``
  : l'image ne peut donc jamais être déformée ;
- le reste de la surface est du vide (lettres noires), et l'image est CENTRÉE
  dans ce vide ;
- le clic est converti en retranchant le décalage puis en divisant par le
  facteur.

Un clic dans le vide est IGNORÉ : il ne correspond à aucun pixel de la source,
et émettre une coordonnée négative ou hors cadre produirait une ligne que
l'opérateur ne voit pas et qui ne peut rien compter.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

# Fond du cadre vidéo. NOUS N'IMITONS PAS LA BORDURE du brief : un `border:
# 1px` posé par la feuille de style RÉDUIT `contentsRect()` de 1 px sur chaque
# bord, donc une image 320x240 dans un widget 320x240 est affichée à 318x238,
# jamais à sa taille naturelle, et le facteur d'échelle vaut 0.9917 au lieu de
# 1. Le calcul de centrage devrait alors dépendre de `contentsRect()` et non
# du widget, pour une contrepartie purement cosmétique. Le fond presque noir
# suffit à détacher la vidéo du panneau (#14161a), donc la bordure est
# abandonnée. `contentsRect()` reste la référence dans le calcul : c'est
# elle que Qt utilise pour peindre, donc elle reste juste même si une feuille
# de style réintroduit une bordure plus tard.
MARGIN_HORS = "#0b0c0e"


class WidgetVideo(QLabel):
    """Affiche une image numpy et remonte les clics en coordonnées pixel.

    Signaux :
        `clic(x, y)` — clic gauche, en coordonnées PIXEL de l'image source.
        `image_changee(np.ndarray)` — nouvelle image affichée.
    """

    clic = Signal(int, int)
    image_changee = Signal(object)
    #: Glisser-déposer : `(dx, dy)` en pixels IMAGE, émis à chaque mouvement
    #: tant qu'un déplacement est armé. Le signataire l'applique à la ligne.
    deplacement = Signal(int, int)
    #: Fin du glissement : désarme le déplacement.
    deplacement_fini = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Le centrage est fait par NOUS (voir `_recalculer_echelle`), pas par
        # Qt : le décalage doit être connu pour convertir les clics. Aligné
        # centré, Qt placerait le pixmap au centre du widget, ce qui est
        # cohérent avec notre calcul — mais on préfère que l'offset soit
        # explicite et testable plutôt que délégué à Qt.
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {MARGIN_HORS};")
        self._image: np.ndarray | None = None
        self._pixmap: QPixmap = QPixmap()
        self._facteur = 1.0
        self._decalage = (0, 0)
        # Zone pour laquelle la géométrie ci-dessus a été calculée. Sert de
        # garde à `_synchroniser` : un QRect vide ne peut pas correspondre à
        # une zone réelle, donc le premier appel recalcule toujours.
        self._zone_connue = QRect()
        #: Déplacement de ligne armé : on retient le point d'ancrage (widget)
        #: et le décalage cumulé depuis le début du geste (image).
        self._deplacement_arme = False
        self._ancre_widget: tuple[float, float] | None = None
        self._decalage_arme = (0, 0)
        #: Le rectangle de la bande en pixels IMAGE, ou None. Tant qu'il est
        #: None — pas de ligne, ou analyse lancée — le clic reste un tracé.
        self._zone_deplacement: tuple[int, int, int, int] | None = None

    # -- API publique ----------------------------------------------------

    def definir_zone_deplacement(
        self, rect: "tuple[int, int, int, int] | None"
    ) -> None:
        """Zone où un clic déplace la ligne et sa bande, en pixels IMAGE.

        `None` rend le glisser-déposer INOPÉRANT : le clic redevient un tracé.
        C'est ce qu'on fait pendant une analyse, où la ligne est verrouillée —
        l'opérateur ne peut alors plus rien déplacer, et le glisser ne doit pas
        planter en déplaçant une ligne que `Ligne.deplacer` refuserait.
        """
        self._zone_deplacement = rect

    def definir_image(self, img: np.ndarray | None) -> None:
        """Affiche ``img`` (BGR ou gris), ou efface l'affichage si ``None``."""
        if img is None:
            self._image = None
            self._pixmap = QPixmap()
            self._zone_connue = QRect()
            self.clear()
            return

        # L'image pleine résolution reste la source de vérité (conversion des
        # clics, reconstruction au redimensionnement). Le pixmap, lui, est
        # produit à la TAILLE D'AFFICHAGE : convertir, copier puis rétrécir
        # 2 Mo de pixels par frame pour jeter 80 % du résultat était le poste
        # le plus lourd de l'affichage. Le redimensionnement numpy
        # (INTER_AREA) est plus rapide que `scaled(SmoothTransformation)` de
        # Qt et s'applique AVANT les conversions, qui ne portent alors que sur
        # l'image réduite.
        self._image = img
        self._peindre()
        self.image_changee.emit(img)

    def dimensions_image(self) -> "tuple[int, int] | None":
        """`(largeur, hauteur)` de l'image affichée, ou None si aucune.

        C'est l'image RÉELLEMENT à l'écran, donc celle dont les coordonnées
        sont celles du widget : une zone de déplacement calculée sur une autre
        image serait décalée, et le clic porterait à côté.
        """
        if self._image is None:
            return None
        hauteur, largeur = self._image.shape[:2]
        return largeur, hauteur

    def facteur_echelle(self) -> float:
        """Facteur appliqué à l'image source pour l'afficher. Jamais nul."""
        self._synchroniser()
        return self._facteur

    def decalage(self) -> tuple[int, int]:
        """Position du coin haut-gauche de l'image, en coordonnées widget."""
        self._synchroniser()
        return self._decalage

    def vers_pixels(self, x: float, y: float) -> tuple[int, int] | None:
        """Convertit un point WIDGET en pixel source, ou ``None`` si hors image.

        Le centrage impose de soustraire le décalage AVANT de diviser : sans
        ça, dans une fenêtre plus large que l'image, un clic au bord gauche de
        la fenêtre donnerait un x négatif alors que le pixel 0 est précisément
        à 160 px de ce bord.

        L'arrondi est « au pixel le plus proche », d'où une zone grise
        d'un demi-pixel d'affichage autour de chaque bord : un clic posé à
        exactement -0.5 px hors image est rattaché au pixel 0. L'écart est
        d'un demi-pixel à l'écran, sans effet sur le comptage, et l'autre
        choix (rejeter) ferait perdre un point que l'opérateur a visiblement
        posé sur le bord.
        """
        if self._image is None:
            return None
        self._synchroniser()
        if self._facteur <= 0:
            return None
        dx = x - self._decalage[0]
        dy = y - self._decalage[1]
        px = int(round(dx / self._facteur))
        py = int(round(dy / self._facteur))
        hauteur, largeur = self._image.shape[:2]
        if not (0 <= px < largeur and 0 <= py < hauteur):
            return None
        return px, py

    def vers_widget(self, px: float, py: float) -> tuple[int, int]:
        """Inverse de `vers_pixels` : pixel source -> point widget.

        Utilisé pour redessiner un aperçu de la ligne en coordonnées image.
        """
        self._synchroniser()
        return (
            int(round(px * self._facteur)) + self._decalage[0],
            int(round(py * self._facteur)) + self._decalage[1],
        )

    # -- Interne ---------------------------------------------------------

    def _zone_affichage(self) -> QRect:
        """Zone de contenu du widget : celle où Qt dessine réellement.

        `width()` inclut la bordure du style (1 px de chaque côté), alors que
        le pixmap est dessiné dans `contentsRect()`. Utiliser `width()` pour
        calculer la place disponible décalerait l'image d'un pixel par rapport
        à la zone peinte — et le clic suivrait ce décalage, donc la ligne
        serait posée un pixel trop loin.
        """
        return self.contentsRect()

    def _synchroniser(self) -> None:
        """Recalcule la géométrie si la zone d'affichage a bougé.

        `resizeEvent` ne suffit PAS comme unique source de vérité : Qt ne
        livre pas cet événement à un widget caché, et il peut être coalescé
        quand plusieurs redimensionnements arrivent dans la même boucle
        (un plein écran, un maximize). Mesuré ici : après deux `resize()` sur
        un widget non affiché, le facteur et le décalage restaient ceux de la
        taille précédente — donc les clics étaient convertis avec une échelle
        périmée et la ligne partait ailleurs que le pixel visé, sans que rien
        ne le signale à l'écran.

        On recalcule donc à la demande, dès qu'une coordonnée est lue, si la
        zone de contenu diffère de celle pour laquelle la géométrie a été
        calculée. Coût : un `contentsRect()` et une comparaison, négligeable.
        """
        if self._image is None:
            return
        contenu = self._zone_affichage()
        if contenu == self._zone_connue:
            return
        self._recalculer_echelle()

    def _recalculer_echelle(self) -> None:
        if self._image is None:
            self._facteur = 1.0
            self._decalage = (0, 0)
            self._zone_connue = QRect()
            return
        hauteur_img, largeur_img = self._image.shape[:2]
        contenu = self._zone_affichage()
        largeur_ui = max(1, contenu.width())
        hauteur_ui = max(1, contenu.height())
        # Un seul facteur pour les deux axes, le plus contraignant des deux :
        # c'est ce qui garantit que l'image n'est pas déformée.
        self._facteur = min(largeur_ui / largeur_img, hauteur_ui / hauteur_img)
        largeur_aff = int(round(largeur_img * self._facteur))
        hauteur_aff = int(round(hauteur_img * self._facteur))
        # Centrage dans le reste de l'espace.
        dec_x = (largeur_ui - largeur_aff) // 2
        dec_y = (hauteur_ui - hauteur_aff) // 2
        # QLabel dessine le pixmap dans contentsRect(), dont l'origine n'est
        # pas (0, 0) quand une bordure est posée : on y ajoute le coin.
        self._decalage = (dec_x + contenu.left(), dec_y + contenu.top())
        self._zone_connue = contenu

    def resizeEvent(self, event) -> None:  # noqa: N802 (API Qt)
        super().resizeEvent(event)
        self._recalculer_echelle()
        self._peindre()

    def _peindre(self) -> None:
        if self._image is None:
            return
        # Le pixmap est reconstruit ICI, à la taille d'affichage courante :
        # une fois par nouvelle image ET une fois par redimensionnement de
        # fenêtre (resizeEvent passe ici aussi). Qt peint ensuite 1:1, plus
        # aucune mise à l'échelle dans la boucle d'affichage.
        self._pixmap = self._construire_pixmap()
        self.setPixmap(self._pixmap)

    def _construire_pixmap(self) -> QPixmap:
        """Pixmap à la TAILLE D'AFFICHAGE, reconstruit depuis `self._image`.

        Appelé à chaque nouvelle image et à chaque redimensionnement. Le
        facteur et le décalage viennent de `_recalculer_echelle` : la taille
        cible est donc exactement celle que `vers_pixels` suppose côté clic —
        la géométrie des interactions est inchangée.
        """
        if self._image is None:
            return QPixmap()
        self._recalculer_echelle()
        hauteur_img, largeur_img = self._image.shape[:2]
        cible = QSize(
            max(1, int(round(largeur_img * self._facteur))),
            max(1, int(round(hauteur_img * self._facteur))),
        )
        affichee = cv2.resize(
            self._image,
            (cible.width(), cible.height()),
            # INTER_AREA pour la réduction (le cas courant : l'écran est plus
            # petit que la vidéo), INTER_LINEAR pour l'agrandissement.
            interpolation=cv2.INTER_AREA
            if self._facteur < 1.0
            else cv2.INTER_LINEAR,
        )
        # La conversion BGR -> RGB et le `.copy()` de la QImage sont
        # indispensables : QImage ne copie pas le buffer numpy, il le pointe.
        # Sans le `.copy()`, le buffer temporaire serait libéré à la sortie de
        # cette fonction et l'affichage deviendrait un dangling pointer — la
        # vidéo se gèle ou affiche n'importe quoi.
        rgb = _vers_rgb(affichee)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (API Qt)
        """Remonte le clic gauche en coordonnées de l'image source.

        La position de l'événement est relative au widget, pas au pixmap :
        c'est exactement ce que `vers_pixels` sait corriger grâce au décalage.
        Le bouton droit est laissé passer : un clic droit ouvre le menu
        contextuel, il ne doit pas poser un point de ligne.
        """
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = event.position().x(), event.position().y()
        point = self.vers_pixels(x, y)
        if point is None:
            return
        # Un clic DANS la bande, une ligne existant : on arme le déplacement,
        # pas un tracé. Le clic reste un tracé dans tous les autres cas — pas
        # de bande du tout, ou clic hors bande — parce que c'est ainsi qu'on
        # trace la première ligne, alors qu'aucune bande n'existe encore.
        if self._dans_zone_deplacement(point):
            self._deplacement_arme = True
            # Ancrage en pixels IMAGE : le geste se mesure donc dans l'unité
            # qui compte pour la ligne, pas dans celle du widget.
            self._ancre_widget = point
            self._decalage_arme = (0, 0)
            return
        self.clic.emit(point[0], point[1])

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (API Qt)
        """Déplace la ligne ET sa bande, en poussant la même quantité.

        La bande n'est pas déplacée séparément : elle est DÉRIVÉE de la ligne
        (`Ligne.rect_bande_detection`), donc `deplacement` ne porte que le
        décalage et la bande suit par construction — aucun second état ne peut
        diverger.

        Le décalage est mesuré depuis l'ANCRAGE du geste, pas depuis le
        mouvement précédent : un `mouseMoveEvent` peut être coalescé par Qt
        quand la souris bouge vite, et un incrément par événement perdrait alors
        des pixels.
        """
        super().mouseMoveEvent(event)
        if not self._deplacement_arme or self._ancre_widget is None:
            return
        point = self.vers_pixels(event.position().x(), event.position().y())
        if point is None:
            return
        # Le curseur peut sortir de l'image pendant le geste : on borne le
        # décalage aux dimensions de la source plutôt que de le perdre, sinon
        # la boîte réapparaîtrait en sautant quand le curseur rentre.
        hauteur, largeur = self._image.shape[:2]
        ax, ay = self._ancre_widget
        dx = min(max(point[0] - ax, -largeur), largeur)
        dy = min(max(point[1] - ay, -hauteur), hauteur)
        if (dx, dy) == self._decalage_arme:
            return
        self._decalage_arme = (dx, dy)
        self.deplacement.emit(dx, dy)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (API Qt)
        """Désarme le déplacement, bouton gauche ou non."""
        super().mouseReleaseEvent(event)
        if not self._deplacement_arme:
            return
        self._deplacement_arme = False
        self._ancre_widget = None
        self._decalage_arme = (0, 0)
        self.deplacement_fini.emit()

    def _dans_zone_deplacement(self, point: tuple[int, int]) -> bool:
        """Le point pixel est-il dans la bande déplaçable ?

        Les bords sont INCLUSIFS des deux côtés : une bande de rognage va
        souvent jusqu'au bord de l'image, et un clic exactement dessus est un
        clic sur la bande, pas à côté.
        """
        if self._zone_deplacement is None or self._image is None:
            return False
        x1, y1, x2, y2 = self._zone_deplacement
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2


def _vers_rgb(img: np.ndarray) -> np.ndarray:
    """Image numpy BGR (ou grise) -> tableau RGB contigu, prêt pour QImage."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # QImage lit le buffer via son pointeur : il lui faut un tableau contigu
    # en uint8, sinon les lignes s'entrelacent et l'image est en biais.
    return np.ascontiguousarray(rgb)