"""Tests de l'overlay d'annotations.

L'overlay est le seul endroit qui dessine, mais il ne décide rien : il reçoit
une `FrameResult` et une `Ligne`, et rend une image. Ces tests vérifient donc
deux propriétés qui tiennent quelle que soit la scène : l'image source n'est
jamais mutée (l'overlay travaille sur une copie, sinon l'aperçu repeindrait la
frame du pipeline), et rien n'est dessiné quand il n'y a rien à dessiner.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from compteur.types import Detection, FrameResult, Track  # noqa: E402
from interface.overlay import dessiner  # noqa: E402


def image():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_overlay_ne_mute_pas_l_source():
    img = image()
    avant = img.copy()
    dessiner(img, FrameResult(image=img))
    assert np.array_equal(img, avant)


def test_overlay_ne_mute_pas_la_source_quand_il_dessine():
    """Version qui dessine réellement, parce que la précédente ne prouve rien.

    Sur une frame vide, aucun trait n'est tracé, donc même une implémentation
    qui peindrait directement dans `img` passerait le test. Vérifié par
    mutation : retirer la copie protectrice ne fait échouer QUE ce test-ci si
    l'on garde la version vide seule. Ce test est donc le verrou.
    """
    img = image()  # noire
    avant = img.copy()
    r = FrameResult(
        image=img,
        detections=[Detection(50, 50, 90, 120, 0.9, 0)],
        tracks=[Track(1, (70, 85), (50, 50, 90, 120), 3, True)],
        total=7,
    )
    sortie = dessiner(img, r, flash=True)

    assert sortie.any(), "rien n'a été dessiné : le test ne prouve rien"
    assert np.array_equal(img, avant), "la source doit rester vierge"


def test_ligne_ne_mute_pas_la_source_non_plus():
    """`Ligne.dessiner` promise lui aussi de ne pas muter : la ligne tracée
    par l'opérateur ne doit pas salir la frame du pipeline."""
    from compteur.ligne import Ligne

    img = image()
    avant = img.copy()
    ligne = Ligne((160, 20), (160, 220), epaisseur=40, sens=1)
    dessiner(img, FrameResult(image=img), ligne=ligne)
    assert np.array_equal(img, avant)


def test_overlay_meme_taille_entree_sortie():
    img = image()
    assert dessiner(img, FrameResult(image=img)).shape == img.shape


def test_boite_dessinee_change_l_image():
    img = image()
    r = FrameResult(
        image=img,
        detections=[Detection(50, 50, 90, 120, 0.9, 0)],
        tracks=[Track(1, (70, 85), (50, 50, 90, 120), 3, True)],
    )
    assert dessiner(img, r).any(), "les boîtes doivent être visibles"


def test_sans_detection_image_inchangee():
    img = image()
    assert not dessiner(img, FrameResult(image=img)).any()


def test_flash_change_l_image():
    img = image()
    r = FrameResult(image=img, total=5)
    assert dessiner(img, r, flash=True).any(), "le flash doit être visible"


# -- Robustesse : l'overlay encaisse des frames de tailles et de formats
#    various sans planter. Une frame en niveaux de gris est un cas réel dès
#    qu'une caméra est en mode noir et blanc.


def test_image_gris_est_acceptee():
    img = np.zeros((240, 320), dtype=np.uint8)
    sortie = dessiner(img, FrameResult(image=img))
    assert sortie.shape[:2] == (240, 320)


def test_boite_hors_cadre_ne_plante_pas():
    """Une détection partly hors image arrive en bord de cadre, pas au centre."""
    img = image()
    r = FrameResult(
        image=img,
        detections=[Detection(-40.0, -30.0, 25.0, 18.0, 0.5, 0)],
        tracks=[Track(9, (-12.0, -7.0), (-40, -30, 25, 18), 1, False)],
    )
    sortie = dessiner(img, r)
    assert sortie.shape == img.shape
    assert sortie.any()


def test_afficher_ids_false_ne_dessine_pas_le_texte():
    """Sans identifiants, il doit rester des pixels : boîtes et centres."""
    img = image()
    r = FrameResult(
        image=img,
        tracks=[Track(1, (70, 85), (50, 50, 90, 120), 3, True)],
    )
    assert dessiner(img, r, afficher_ids=False).any()


def test_les_centres_de_track_sont_dessines():
    """Les centres sont sur la couleur prévue : c'est ce qui matérialise le
    point suivi, utilisé pour décider des franchissements."""
    from interface.overlay import COULEUR_TRACK

    img = np.full((240, 320, 3), 0, dtype=np.uint8)
    r = FrameResult(image=img, tracks=[Track(1, (160.0, 120.0), (150, 110, 170, 130), 2, True)])
    sortie = dessiner(img, r)
    assert (sortie[120, 160] == np.array(COULEUR_TRACK, dtype=np.uint8)).all()


def test_ligne_est_deleguee_sans_etre_reecrite():
    """La géométrie de la ligne appartient à `compteur.ligne`. L'overlay
    appelle `Ligne.dessiner` et ne redessine ni la bande ni la flèche : une
    réécriture ici divergenceirait de la flèche déjà corrigée en tâche 2."""
    from compteur.ligne import Ligne

    img = image()
    ligne = Ligne((50, 20), (50, 220), epaisseur=40, sens=1)
    attendu = ligne.dessiner(img)
    obtenu = dessiner(img, FrameResult(image=img), ligne=ligne)
    assert np.array_equal(obtenu, attendu)


def test_bande_de_la_ligne_reste_translucide():
    """La bande est un aperçu, pas un voile : sous elle, l'original reste
    lisible. Une bande opaque masquerait exactement les gens comptés."""
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    from compteur.ligne import Ligne

    ligne = Ligne((160, 20), (160, 220), epaisseur=40, sens=1)
    sortie = dessiner(img, FrameResult(image=img), ligne=ligne)
    # Pixel au cœur de la bande : très atténué par rapport au blanc pur,
    # mais pas noirci jusqu'à l'opacité.
    coeur = sortie[120, 160].astype(np.int16)
    assert 0 < coeur.mean() < 255, f"bande opaque ou absente : {coeur}"


def test_les_boites_se_voient_sur_fond_clair():
    """En extérieur, le fond est souvent blanc (ciel, sol surexposé). Une
    boîte de la couleur de l'image ne serait pas visible : on vérifie que le
    contour la modifie vraiment."""
    img = np.full((240, 320, 3), 200, dtype=np.uint8)
    r = FrameResult(image=img, detections=[Detection(100, 100, 140, 160, 0.9, 0)])
    sortie = dessiner(img, r)
    assert not np.array_equal(sortie, img)