import numpy as np
import pytest

from compteur.ligne import Ligne


def ligne_haut_vers_bas(**kw):
    """Ligne verticale x=100, de y=0 à y=1000. Traversée dans le sens +y."""
    return Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), **kw)


def test_coordonnee_projetee_nulle_sur_la_ligne():
    l = ligne_haut_vers_bas()
    assert l.coordonnee_projetee((100.0, 500.0)) == pytest.approx(0.0, abs=1e-6)


def test_coordonnee_signee_change_de_cote():
    l = ligne_haut_vers_bas()
    a = l.coordonnee_projetee((50.0, 500.0))
    b = l.coordonnee_projetee((150.0, 500.0))
    assert a * b < 0, "les deux points sont de côtés opposés"


def test_point_du_cote_avant_et_apres():
    l = ligne_haut_vers_bas(sens=1)
    assert l.point_du_cote((50.0, 500.0)) == 1
    assert l.point_du_cote((150.0, 500.0)) == -1


def test_inverser_sens_inverse_les_cotes():
    l = ligne_haut_vers_bas(sens=-1)
    assert l.point_du_cote((50.0, 500.0)) == -1
    assert l.point_du_cote((150.0, 500.0)) == 1


def test_point_dans_la_bande():
    l = ligne_haut_vers_bas(epaisseur=30)
    assert l.est_dans_la_bande((110.0, 500.0)) is True
    assert l.est_dans_la_bande((120.0, 500.0)) is False


def test_point_hors_segment_malgre_proche_de_la_ligne():
    """Une personne à 1 px de la ligne mais à 5000 px du segment n'est pas dessus."""
    l = ligne_haut_vers_bas(epaisseur=30)
    assert l.contient((105.0, 5000.0)) is False
    assert l.contient((105.0, 500.0)) is True


def test_ligne_degenerate_refusee():
    with pytest.raises(ValueError):
        Ligne(p1=(10.0, 10.0), p2=(10.0, 10.0))


def test_vecteur_normal_unitaire():
    l = ligne_haut_vers_bas()
    nx, ny = l.vecteur_normal()
    assert nx * nx + ny * ny == pytest.approx(1.0, abs=1e-6)


def test_sens_invalide_refuse():
    with pytest.raises(ValueError):
        ligne_haut_vers_bas(sens=0)


def test_dessiner_ne_crash_pas_et_ne_mute_pas():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    l = ligne_haut_vers_bas()
    avant = img.copy()
    sortie = l.dessiner(img)
    assert sortie.shape == img.shape
    assert np.array_equal(img, avant), "dessiner() ne doit pas modifier l'image source"
    assert sortie.any(), "la ligne doit être visible"
