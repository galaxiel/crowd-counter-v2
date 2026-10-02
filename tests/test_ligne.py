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


def test_traversee_complete_est_comptee():
    l = Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), epaisseur=30, sens=1)
    assert l.a_traverse((50.0, 500.0), (150.0, 500.0)) is True


def test_traversee_lente_comptee():
    """Une personne a 5 px/frame traverse la bande en plusieurs frames."""
    l = Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), epaisseur=30, sens=1)
    # 5 px de deplacement, mais un pas qui franchit vraiment x=100.
    assert l.a_traverse((98.0, 500.0), (103.0, 500.0)) is True


def test_pas_de_cote_a_cote_n_est_pas_une_traversee():
    """88 -> 93 : 5 px, mais les deux points restent du meme cote de x=100."""
    l = Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), epaisseur=30, sens=1)
    assert l.a_traverse((88.0, 500.0), (93.0, 500.0)) is False


def test_pied_pose_pile_sur_la_ligne_compte_une_seule_fois():
    """Un pas dont un extremite tombe exactement sur la ligne est retenu."""
    l = Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), epaisseur=30, sens=1)
    assert l.coordonnee_projetee((100.0, 500.0)) == pytest.approx(0.0, abs=1e-6)
    assert l.a_traverse((95.0, 500.0), (100.0, 500.0)) is True
    # le pas suivant repart du meme point, cote oppose : pas de second compte.
    assert l.a_traverse((100.0, 500.0), (105.0, 500.0)) is False


def test_ballon_rebondissant_dans_la_bande_non_compte():
    l = Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), epaisseur=30, sens=1)
    assert l.a_traverse((88.0, 500.0), (85.0, 500.0)) is False


def test_meme_cote_non_compte():
    l = Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), epaisseur=30, sens=1)
    assert l.a_traverse((50.0, 500.0), (80.0, 500.0)) is False


def test_croisement_avant_le_debut_du_segment_non_compte():
    """Le croisement est sur la meme verticale mais hors du segment dessine."""
    l = Ligne(p1=(500.0, 500.0), p2=(500.0, 600.0), epaisseur=30, sens=1)
    assert l.a_traverse((450.0, 100.0), (550.0, 100.0)) is False


def test_croisement_apres_la_fin_du_segment_non_compte():
    l = Ligne(p1=(500.0, 500.0), p2=(500.0, 600.0), epaisseur=30, sens=1)
    assert l.a_traverse((450.0, 2000.0), (550.0, 2000.0)) is False


def test_bilan_de_balayage_lent():
    """200 frames de marche a 5 px/frame doivent produire exactement 1 traversee."""
    l = Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), epaisseur=30, sens=1)
    x, total = 40.0, 0
    for _ in range(200):
        if l.a_traverse((x, 500.0), (x + 5.0, 500.0)):
            total += 1
        x += 5.0
    assert total == 1, "une marche continue ne doit produire qu'un seul comptage"


def test_sens_inverse_inverse_la_traversee():
    """Le meme passage est compte avec sens=+1 et refuse avec sens=-1."""
    avant, apres = (50.0, 500.0), (150.0, 500.0)
    assert Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), sens=1).a_traverse(avant, apres) is True
    assert Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), sens=-1).a_traverse(avant, apres) is False


def test_hysteresis_est_expose():
    """La tache 5 lit ligne.hysteresis : l'attribut doit rester disponible."""
    l = Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), hysteresis=4)
    assert l.hysteresis == 4


def test_traversee_diagonale_comptee_au_croisement_interpole():
    """Le croisement est juge a sa position interpolee, pas a celle de ``avant``.

    Ligne courte y=500..600, marche diagonale de (90,400) a (110,700) : le
    croisement reel est en (100, 550), sur le segment. Prendre ``avant``
    donnerait s = -100, hors segment, et la traversee serait refusee a tort.
    """
    l = Ligne(p1=(100.0, 500.0), p2=(100.0, 600.0), epaisseur=30, sens=1)
    assert l.a_traverse((90.0, 400.0), (110.0, 700.0)) is True


def test_croisement_diagonal_hors_segment_non_compte():
    """Meme diagonale, mais le croisement interpole sort apres la fin du segment.

    Le point de depart est pourtant sur le segment : seule l'interpolation
    permet de le refuser.
    """
    l = Ligne(p1=(100.0, 500.0), p2=(100.0, 600.0), epaisseur=30, sens=1)
    assert l.a_traverse((90.0, 550.0), (110.0, 900.0)) is False


def test_dessiner_ne_crash_pas_et_ne_mute_pas():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    l = ligne_haut_vers_bas()
    avant = img.copy()
    sortie = l.dessiner(img)
    assert sortie.shape == img.shape
    assert np.array_equal(img, avant), "dessiner() ne doit pas modifier l'image source"
    assert sortie.any(), "la ligne doit être visible"
