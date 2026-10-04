"""Tests du widget vidéo et de la conversion écran -> pixel source.

C'est le point le plus sensible de la tâche : l'opérateur trace la ligne à la
souris, et il le fait en regardant la scène. Si la conversion coordonnées
widget -> coordonnées de l'image source est fausse, la ligne est tracée au
mauvais endroit *et* l'erreur est invisible à l'écran : la ligne affichée
paraît bien posée, c'est le comptage qui sera faux. Ces tests vérifient donc la
conversion dans les trois régimes : image à la taille du widget (facteur 1),
image plus grande que le widget (réduction), fenêtre plus grande que l'image
(lettres + centrage), plus le clic dans les marges.

Aucun affichage réel n'est nécessaire : le widget est instancié sans `show()`,
sous la plateforme Qt « offscreen », imposée ci-dessous AVANT toute création de
QApplication — Qt choisit son moteur de rendu au premier QApplication, pas à
l'import. On ne touche pas au `conftest.py` partagé pour cela : un autre agent
travaille peut-être dessus en parallèle.
"""

import os

# Doit précéder la création de QApplication (voir ci-dessus). `setdefault`
# respecte un choix déjà pris dans l'environnement de lancement.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from interface.widgets_video import WidgetVideo  # noqa: E402


@pytest.fixture(scope="module")
def application():
    """Une QApplication partagée. Qt n'en autorise qu'une seule par process,
    d'où la portée de module et le réutilisation de l'instance existante."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def image(largeur=320, hauteur=240):
    return np.zeros((hauteur, largeur, 3), dtype=np.uint8)


def creer(largeur=None, hauteur=None) -> WidgetVideo:
    """Widget de `largeur` x `hauteur` pixels, taille minimale levée.

    `setMinimumSize(640, 360)` fait partie de l'IHM (une vidéo trop petite
    n'est pas lisible), mais Qt l'applique comme une contrainte de NATURE :
    `resize(320, 240)` sur un widget caché est silencieusement ignoré et la
    taille reste 640x360. Les tests de conversion de coordonnées ont besoin
    de tailles exactes, on lève donc le plancher explicitement — c'est le
    contrat qu'ils testent, pas l'habillage.
    """
    w = WidgetVideo()
    w.setMinimumSize(0, 0)
    if largeur is not None and hauteur is not None:
        w.resize(largeur, hauteur)
    return w


def cliquer(widget, x, y, bouton=None):
    """Simule un clic en coordonnées WIDGET et renvoie ce que le widget émet."""
    if bouton is None:
        bouton = Qt.MouseButton.LeftButton
    captures = []
    widget.clic.connect(lambda px, py: captures.append((px, py)))
    evenement = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, y),
        QPointF(x, y),
        bouton,
        bouton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(evenement)
    return captures


# -- Facteur d'échelle et centrage ---------------------------------------


def test_facteur_1_quand_l_image_occupe_le_widget(application):
    w = creer(320, 240)
    w.definir_image(image(320, 240))
    assert w.facteur_echelle() == pytest.approx(1.0)
    assert w.decalage() == (0, 0)


def test_image_plus_grande_que_widget_reduit(application):
    """Image 640x480 dans un widget 320x240 : facteur 0.5, pas de marge."""
    w = creer(320, 240)
    w.definir_image(image(640, 480))
    assert w.facteur_echelle() == pytest.approx(0.5)
    assert w.decalage() == (0, 0)


def test_widget_plus_grand_que_l_image_centre_horizontalement(application):
    """Image 320x240 dans un widget 640x240 : facteur 1, 160 px de marge de
    chaque côté. Sans ce centrage, la ligne serait tracée à l'envers."""
    w = creer(640, 240)
    w.definir_image(image(320, 240))
    assert w.facteur_echelle() == pytest.approx(1.0)
    assert w.decalage() == (160, 0)


def test_widget_plus_grand_que_l_image_centre_verticalement(application):
    w = creer(320, 480)
    w.definir_image(image(320, 240))
    assert w.facteur_echelle() == pytest.approx(1.0)
    assert w.decalage() == (0, 120)


def test_les_deux_axes_choisissent_le_meme_facteur(application):
    """Le facteur est unique et contrainte les DEUX axes : sinon l'image serait
    déformée et le point que l'opérateur vise ne correspondrait plus à la
    personne qu'il voit."""
    w = creer(800, 200)  # bien plus large que haut
    w.definir_image(image(400, 400))
    assert w.facteur_echelle() == pytest.approx(0.5)
    # 400x400 -> 200x200, centré verticalement dans 200 px de haut : 0.
    assert w.decalage() == (300, 0)


def test_redimensionner_recalcule_le_facteur(application):
    w = creer(320, 240)
    w.definir_image(image(640, 480))
    assert w.facteur_echelle() == pytest.approx(0.5)
    w.resize(640, 480)
    assert w.facteur_echelle() == pytest.approx(1.0)
    assert w.decalage() == (0, 0)


def test_facteur_jamais_nul_sans_image(application):
    """Sans image, toute division par le facteur planterait au premier clic."""
    w = creer()
    w.definir_image(None)
    assert w.facteur_echelle() > 0


# -- Clic -> coordonnées pixel de l'image source -------------------------


def test_clic_au_coin_haut_gauche_donne_0_0(application):
    w = creer(320, 240)
    w.definir_image(image(320, 240))
    assert cliquer(w, 0, 0) == [(0, 0)]


def test_clic_au_coin_bas_droit_donne_les_dimensions_source(application):
    w = creer(320, 240)
    w.definir_image(image(320, 240))
    px, py = cliquer(w, 319, 239)[0]
    assert (px, py) == (319, 239)


def test_clic_au_centre_donne_le_centre_de_l_image(application):
    w = creer(640, 480)
    w.definir_image(image(640, 480))
    assert cliquer(w, 320, 240) == [(320, 240)]


def test_clic_avec_reduction_retourne_le_pixel_source(application):
    """Image 640x480 réduite de moitié dans un widget 320x240 : un clic au
    centre du widget doit viser le centre de l'image SOURCE, pas 160,240."""
    w = creer(320, 240)
    w.definir_image(image(640, 480))
    assert cliquer(w, 160, 120) == [(320, 240)]


def test_clic_tient_compte_du_centrage(application):
    """Image 320x240 centrée dans un widget 640x240 : cliquer sur le pixel
    source (0,0) impose de cliquer à x=160 dans le widget, pas à x=0."""
    w = creer(640, 240)
    w.definir_image(image(320, 240))
    assert cliquer(w, 160, 0) == [(0, 0)]
    assert cliquer(w, 160 + 159, 239) == [(159, 239)]


def test_clic_hors_de_l_image_est_ignore(application):
    """Un clic dans la marge ne doit pas produire de pixel fantôme en dehors de
    l'image : la ligne serait alors tracée hors cadre et jamais vue."""
    w = creer(640, 240)
    w.definir_image(image(320, 240))
    assert cliquer(w, 10, 10) == []
    assert cliquer(w, 600, 200) == []


def test_bord_exterieur_gauche_ignore_bord_interieur_accepte(application):
    """Le centrage est testé bord par bord : il doit être EXACT, pas
    approximatif — 1 px d'erreur se voit quand l'opérateur vise quelqu'un
    collé au bord du cadre.

    Géométrie : image 320x240 (4:3) dans un widget 800x480 (5:3). Le facteur
    est limité par la hauteur (480/240 = 2), donc l'image est affichée sur
    640x480, centrée : 80 px de marge à gauche et à droite, 0 en haut et en
    bas. Sur les quatre bords à la fois, c'est impossible : avec un facteur
    unique, une marge existe sur un axe ou sur l'autre, jamais sur les deux.
    Les deux configurations sont donc testées séparément.

    Les positions testées évitent les points à exactement un demi-pixel de
    l'image, où l'arrondi au pixel le plus proche est un arbitrage légitime
    (voir `vers_pixels`). Ce qui est vérifié ici, c'est qu'on ne déborde pas
    dans le vide.
    """
    w = creer(800, 480)
    w.definir_image(image(320, 240))
    assert w.decalage() == (80, 0)
    assert cliquer(w, 40, 240) == []  # franchement dans la marge de gauche
    assert cliquer(w, 80, 240) == [(0, 120)]  # premier pixel de l'image
    assert cliquer(w, 710, 240) == [(315, 120)]
    assert cliquer(w, 750, 240) == []  # franchement dans la marge de droite


def test_bord_exterieur_haut_ignore_bord_interieur_accepte(application):
    """L'autre configuration : cette fois c'est la largeur qui est limitante
    (480/320 = 1.5 contre 800/240 = 3.3), donc la marge est verticale et
    l'image est affichée sur 480x360 au lieu de 320x240. Le même code doit
    donner la bonne réponse."""
    w = creer(480, 800)
    w.definir_image(image(320, 240))
    assert w.facteur_echelle() == pytest.approx(1.5)
    assert w.decalage() == (0, 220)
    assert cliquer(w, 240, 100) == []  # franchement dans la marge du haut
    assert cliquer(w, 240, 220) == [(160, 0)]  # premier pixel de l'image
    assert cliquer(w, 240, 550) == [(160, 220)]  # avant-dernière ligne utile
    assert cliquer(w, 240, 579) == [(160, 239)]  # dernière ligne de l'image
    assert cliquer(w, 240, 700) == []  # franchement dans la marge du bas


def test_clic_a_la_demi_pixel_du_bord_reste_dans_l_image(application):
    """Le contrat d'arrondi, épinglé parce qu'il est ambigu sinon.

    À exactement un demi-pixel hors de l'image, l'arrondi au pixel le plus
    proche bascule dans l'image (Python arrondit 0.5 à l'entier pair, donc
    -0.5 -> 0). Le clic est alors accepté sur le pixel 0. C'est un choix
    arbitraire mais sans danger : l'écart est d'un demi-pixel d'affichage,
    et l'alternative — rejeter — ferait perdre un point posé à l'opérateur
    alors qu'il visait visiblement le bord.
    """
    w = creer(800, 480)
    w.definir_image(image(320, 240))
    assert w.decalage() == (80, 0)
    assert cliquer(w, 79, 240) == [(0, 120)]  # -0.5 px -> pixel 0


def test_clic_bord_exterieur_de_l_image_est_accepte(application):
    """Le tout dernier pixel de l'image est affichable : la bordure fait
    partie de l'image source."""
    w = creer(320, 240)
    w.definir_image(image(320, 240))
    assert cliquer(w, 319, 239) == [(319, 239)]


def test_sans_image_aucun_clic_n_est_emise(application):
    w = creer(320, 240)
    w.definir_image(None)
    assert cliquer(w, 100, 100) == []


def test_bouton_droit_n_emet_pas(application):
    """Seul le bouton gauche sert à poser la ligne ; un clic droit ouvrirait
    sinon un menu contextuel en plein comptage."""
    w = creer(320, 240)
    w.definir_image(image(320, 240))
    assert cliquer(w, 160, 120, Qt.MouseButton.RightButton) == []


def test_aller_retour_widget_vers_pixel_et_inverse(application):
    """Le contrat central : ce que l'IHM lit au clic doit être le pixel que
    l'opérateur a visé. On vérifie l'aller-retour sur une grille, dans une
    image réduite ET centrée, le régime le plus facile à se tromper."""
    w = creer(500, 300)
    w.definir_image(image(640, 480))
    facteur = w.facteur_echelle()
    dx, dy = w.decalage()
    for px in (0, 1, 100, 320, 638, 639):
        for py in (0, 1, 240, 479):
            x = int(round(px * facteur)) + dx
            y = int(round(py * facteur)) + dy
            obtenu = cliquer(w, x, y)[0]
            assert obtenu[0] == pytest.approx(px, abs=1)
            assert obtenu[1] == pytest.approx(py, abs=1)


def test_deux_clics_donnent_une_ligne_placee_sur_les_bons_pixels(application):
    """Le geste complet de l'opérateur : deux clics, une ligne verticale. On
    vérifie la ligne *source* obtenue, celle que le moteur comptabilisera."""
    from compteur.ligne import Ligne

    w = creer(640, 240)
    w.definir_image(image(320, 240))
    w.definir_image(image(320, 240))
    p1 = cliquer(w, 160 + 40, 0)[0]  # pixel source (40, 0)
    p2 = cliquer(w, 160 + 40, 239)[0]  # pixel source (40, 239)
    ligne = Ligne(p1, p2)
    assert ligne.p1 == (40.0, 0.0)
    assert ligne.p2 == (40.0, 239.0)


# -- Signaux et affichage -------------------------------------------------


def test_image_changee_est_emise_avec_l_image(application):
    w = creer()
    captures = []
    w.image_changee.connect(captures.append)
    img = image(64, 48)
    w.definir_image(img)
    assert len(captures) == 1
    assert captures[0].shape == (48, 64, 3)


def test_image_changee_pas_emise_pour_none(application):
    """Effacer l'image ne doit pas faire descendre un `None` dans les slots
    connectés, qui attendent tous une frame numpy."""
    w = creer()
    w.definir_image(image(64, 48))
    captures = []
    w.image_changee.connect(captures.append)
    w.definir_image(None)
    assert captures == []


def test_afficher_une_image_ne_la_mute_pas(application):
    """L'image affichée est celle du pipeline : si le widget la modifiait, la
    frame suivante hériterait des annotations de la précédente."""
    img = np.full((48, 64, 3), 128, dtype=np.uint8)
    avant = img.copy()
    w = creer()
    w.definir_image(img)
    assert np.array_equal(img, avant)


def test_image_gris_est_acceptee(application):
    """Une caméra en noir et blanc produit une frame 2D : planter ici ferait
    perdre la vidéo entière, pas un simple cadre."""
    w = creer(64, 48)
    w.definir_image(np.zeros((48, 64), dtype=np.uint8))
    assert cliquer(w, 32, 24) == [(32, 24)]


def test_les_couleurs_ne_sont_pas_inversees(application):
    """OpenCV travaille en BGR, Qt en RGB. Sans conversion, un carton rouge
    de manifestation s'affiche en bleu — le défaut passe inaperçu, parce que
    l'image reste « plausible ».

    La vérification lit le pixel AFFICHÉ, pas la donnée d'entrée : c'est le
    seul endroit où une inversion se verrait vraiment.
    """
    rouge_bgr = np.zeros((8, 8, 3), dtype=np.uint8)  # (0, 0, 255) = rouge en BGR
    rouge_bgr[4, 4] = (0, 0, 255)
    bleu_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bleu_bgr[4, 4] = (255, 0, 0)

    w = creer(8, 8)
    w.definir_image(rouge_bgr)
    affiche = w.pixmap().toImage().pixelColor(4, 4)
    assert (affiche.red(), affiche.green(), affiche.blue()) == (255, 0, 0)

    w.definir_image(bleu_bgr)
    affiche = w.pixmap().toImage().pixelColor(4, 4)
    assert (affiche.red(), affiche.green(), affiche.blue()) == (0, 0, 255)


def test_l_image_affichee_eteint_les_annotations_de_la_frame_precedente(application):
    """Deux frames de suite ne doivent pas se superposer à l'affichage.

    Le widget remplace son pixmap, il ne compose pas : si l'ancien pixmap
    restait, la première frame conserverait les boîtes de la deuxième et
    l'opérateur verrait des détections fantômes.
    """
    w = creer(8, 8)
    w.definir_image(np.zeros((8, 8, 3), dtype=np.uint8))
    w.definir_image(np.zeros((8, 8, 3), dtype=np.uint8))
    image = w.pixmap().toImage()
    # Toute la frame est noire : aucun pixel non nul dans le pixmap affiché.
    assert all(image.pixelColor(x, y).blue() == 0 for x in range(8) for y in range(8))


def test_widget_vierge_ne_leve_pas(application):
    """Le cas de départ au lancement : ni image, ni redimensionnement."""
    w = creer(320, 240)
    assert w.facteur_echelle() > 0
    assert cliquer(w, 10, 10) == []