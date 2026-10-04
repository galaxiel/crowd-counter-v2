"""Libellés de sens explicites + explications des réglages (tâche 13).

Trois exigences, trois blocs de tests.

1. **« Ligne » connaît son orientation.** Une verticale se nomme gauche /
   droite, une horizontale haut / bas. Le libellé est déduit de la géométrie
   tracée, jamais d'un sens abstrait : sur la même ligne verticale tracée dans
   les deux sens, `sens=+1` ne compte pas la même chose.

2. **Le libellé dit ce que fait le compteur.** C'est la seule propriété qui
   vaille : un menu peut afficher n'importe quoi, tant que le mouvement qu'il
   nomme est celui que `a_traverse` retient. Ces tests font les deux à la
   fois — ils vérifient qu'un marcheur qui traverse DANS le sens du libellé est
   compté, et qu'il ne l'est pas dans l'autre.

3. **Le réglage par défaut reste celui qui compte.** La référence mesurée est
   un cortège allant de DROITE À GAUCHE, 315 personnes ; le même décompte dans
   l'autre sens vaut 1. Perdre ce 315 en changeant un libellé serait le pire
   des régressions, puisque c'est le seul chiffre vérifié de bout en bout.
   Aucun test ne se contente donc de lire la valeur du réglage : tous
   rejouent le passage géométrique.
"""

import pytest

pytest.importorskip("PySide6")

from compteur.ligne import (  # noqa: E402
    LIBELLES_SENS,
    ORIENTATION_HORIZONTALE,
    ORIENTATION_VERTICALE,
    Ligne,
)

#: Géométries de référence, dans les DEUX sens de tracé : c'est ce qui
#: distingue un libellé honnête d'un libellé inversé.
VERTICALE_HAUT_BAS = ((640.0, 0.0), (640.0, 720.0))
VERTICALE_BAS_HAUT = ((640.0, 720.0), (640.0, 0.0))
HORIZONTALE_GAUCHE_DROITE = ((0.0, 400.0), (1280.0, 400.0))
HORIZONTALE_DROITE_GAUCHE = ((1280.0, 400.0), (0.0, 400.0))

#: Marches de référence, en pixels : un couple ``(avant, apres)`` qui traverse
#: le segment de part en part, à 100 px de la ligne de chaque côté (la bande
#: fait 30 px, donc 100 px est franchement « côté départ » et « côté arrivée »).
#: Sur une verticale, la ligne est en x=640 ; sur une horizontale tracée en
#: y=400, les marches verticales utilisent x=600, à l'intérieur du segment.
#:
#: Le passage inverse s'obtient en inversant le couple : c'est la même
#: personne qui revient sur ses pas.
ALLER_DROITE = ((500.0, 300.0), (700.0, 300.0))   # x augmente : vers la droite
ALLER_GAUCHE = ((700.0, 300.0), (500.0, 300.0))   # x diminue  : vers la gauche
ALLER_BAS = ((600.0, 300.0), (600.0, 500.0))      # y augmente : vers le bas
ALLER_HAUT = ((600.0, 500.0), (600.0, 300.0))      # y diminue  : vers le haut


# -- Orientation ---------------------------------------------------------


def test_ligne_verticale_se_nomme_verticale():
    assert Ligne(*VERTICALE_HAUT_BAS).orientation == ORIENTATION_VERTICALE
    assert Ligne(*VERTICALE_BAS_HAUT).orientation == ORIENTATION_VERTICALE


def test_ligne_horizontale_se_nomme_horizontale():
    assert Ligne(*HORIZONTALE_GAUCHE_DROITE).orientation == ORIENTATION_HORIZONTALE
    assert Ligne(*HORIZONTALE_DROITE_GAUCHE).orientation == ORIENTATION_HORIZONTALE


def test_orientation_suit_le_sens_de_trace():
    """L'orientation ne dépend pas de la direction du tracé.

    C'est le test qui casse le plus vite si quelqu'un prend `dy > 0` pour un
    critère d'orientation : la même ligne tracée à l'envers serait alors
    annoncée horizontale.
    """
    assert Ligne(*VERTICALE_HAUT_BAS).orientation == Ligne(
        *VERTICALE_BAS_HAUT
    ).orientation


def test_legere_inclinaison_reste_nommee_dapres_son_axe_dominant():
    """Une ligne penchée de 10 px sur 700 est encore une ligne verticale.

    Le critère est l'axe dominant, pas « dx == 0 » : sinon la même ligne passerait
    de « verticale » à « horizontale » sur un pixel de différence, et les
    libellés du menu changeraient de nature au moindre clic.
    """
    penchee = Ligne(p1=(640.0, 0.0), p2=(650.0, 720.0))
    assert penchee.orientation == ORIENTATION_VERTICALE
    assert penchee.libelle_sens(1) in LIBELLES_SENS[ORIENTATION_VERTICALE]


# -- Libellés ------------------------------------------------------------


def test_libelles_verticaux_sont_gauche_droite_et_droite_gauche():
    assert LIBELLES_SENS[ORIENTATION_VERTICALE] == ("Gauche → droite", "Droite → gauche")


def test_libelles_horizontaux_sont_haut_bas_et_bas_haut():
    assert LIBELLES_SENS[ORIENTATION_HORIZONTALE] == ("Haut → bas", "Bas → haut")


def test_ancien_libelle_ambigu_est_disparu():
    """« Avant → après » ne disait pas avant quoi : il ne doit plus nulle part."""
    for libelles in LIBELLES_SENS.values():
        for libelle in libelles:
            assert "Avant" not in libelle
            assert "Après" not in libelle


def test_choix_sens_donne_les_deux_libelles_dans_un_ordre_stable():
    """Quel que soit le sens de tracé, l'ordre d'affichage est le même.

    L'opérateur doit retrouver « Gauche → droite » en première position d'une
    vidéo à l'autre, sans avoir à se demander si sa ligne a été tracée à
    l'envers.
    """
    for p1, p2 in (VERTICALE_HAUT_BAS, VERTICALE_BAS_HAUT):
        choix = Ligne(p1, p2).choix_sens()
        assert [libelle for libelle, _ in choix] == list(
            LIBELLES_SENS[ORIENTATION_VERTICALE]
        )


def test_libelle_verticale_depend_du_sens_de_trace():
    """Même ligne, tracée en sens inverse : `sens=+1` compte l'inverse.

    C'est la raison d'être des libellés calculés. Un libellé attaché au seul
    `sens` afficherait « Gauche → droite » dans les deux cas, alors que le
    compteur fait deux choses opposées.
    """
    haut_bas = Ligne(*VERTICALE_HAUT_BAS)
    bas_haut = Ligne(*VERTICALE_BAS_HAUT)
    assert haut_bas.libelle_sens(1) != bas_haut.libelle_sens(1)
    assert {haut_bas.libelle_sens(1), haut_bas.libelle_sens(-1)} == {
        "Gauche → droite",
        "Droite → gauche",
    }


def test_libelle_refuse_un_sens_hors_du_domaine():
    with pytest.raises(ValueError, match=r"\+1 ou -1"):
        Ligne(*VERTICALE_HAUT_BAS).libelle_sens(0)


# -- Cohérence libellé / comportement réel -------------------------------


@pytest.mark.parametrize(
    ("p1", "p2"),
    [
        VERTICALE_HAUT_BAS,
        VERTICALE_BAS_HAUT,
        HORIZONTALE_GAUCHE_DROITE,
        HORIZONTALE_DROITE_GAUCHE,
    ],
    ids=["verticale_haut_bas", "verticale_bas_haut", "horiz_gauche_droite", "horiz_droite_gauche"],
)
def test_libelle_et_comportement_sont_daccord(p1, p2):
    """Le mouvement nommé par le libellé est EXACTEMENT celui qui est compté.

    Test central du lot. Pour chacune des quatre géométries et chacun des deux
    sens : un marcheur qui va dans le sens du libellé est compté, le même
    marcheur en sens inverse ne l'est pas. Un libellé qui mentirait ferait
    échouer ce test, quelle que soit sa formulation.
    """
    marches = {
        "Gauche → droite": ALLER_DROITE,
        "Droite → gauche": ALLER_GAUCHE,
        "Haut → bas": ALLER_BAS,
        "Bas → haut": ALLER_HAUT,
    }
    for sens in (1, -1):
        libelle = Ligne(p1, p2).libelle_sens(sens)
        avant, apres = marches[libelle]
        ligne = Ligne(p1, p2, sens=sens)
        assert ligne.a_traverse(avant, apres) is True, (
            f"sens={sens:+d} : le libellé « {libelle} » promet un comptage qui "
            "n'a pas lieu"
        )
        assert ligne.a_traverse(apres, avant) is False, (
            f"sens={sens:+d} : « {libelle} » compte aussi le passage inverse"
        )


def test_libelle_verticale_et_libelle_horizontale_ne_se_confondent_pas():
    """Un libellé d'orientation ne doit jamais apparaître sur l'autre."""
    for (p1, p2), verticale in (
        (VERTICALE_HAUT_BAS, True),
        (HORIZONTALE_GAUCHE_DROITE, False),
    ):
        for sens in (1, -1):
            libelle = Ligne(p1, p2).libelle_sens(sens)
            attendu = (
                LIBELLES_SENS[ORIENTATION_VERTICALE]
                if verticale
                else LIBELLES_SENS[ORIENTATION_HORIZONTALE]
            )
            assert libelle in attendu


# -- Le réglage par défaut reste celui qui compte ------------------------


def test_le_sens_par_defaut_compte_le_cortège_de_droite_a_gauche():
    """LE test de non-régression du 315.

    `config/default.json` porte `sens = +1`. Sur une ligne verticale tracée de
    BAS en HAUT — l'ordre dans lequel la vidéo de référence a été tracée — ce
    `+1` compte de droite à gauche. On le vérifie par la géométrie, pas en
    lisant `Config.sens` : lire la valeur ne prouverait rien, puisque c'est
    précisément la sémantique du libellé qui est en jeu.

    Si ce test échoue, le 315 est cassé : l'opérateur qui relance l'application
    sans toucher aux réglages obtiendrait 1 au lieu de 315.
    """
    from compteur.config import Config, chemin_defaut_config

    config = Config.depuis_fichier(chemin_defaut_config())
    ligne = Ligne(*VERTICALE_BAS_HAUT, sens=config.sens)
    assert ligne.orientation == ORIENTATION_VERTICALE
    assert ligne.libelle_sens(config.sens) == "Droite → gauche"
    assert ligne.a_traverse(*ALLER_GAUCHE) is True, (
        "le sens par défaut doit compter le cortège allant de droite à gauche"
    )
    assert ligne.a_traverse(*ALLER_DROITE) is False


def test_une_ligne_verticale_n_admet_qu_un_sens_par_libelle():
    """Deux libellés, deux comportements disjoints.

    Une fois le libellé affiché, l'opérateur doit pouvoir prédire le résultat
    sans lancer l'analyse : il faut donc que les deux entrées se distinguent
    par un comportement observable, pas seulement par un texte.
    """
    for p1, p2 in (VERTICALE_HAUT_BAS, VERTICALE_BAS_HAUT):
        Retenu = {
            Ligne(p1, p2).libelle_sens(sens): Ligne(p1, p2, sens=sens).a_traverse(
                *ALLER_DROITE
            )
            for sens in (1, -1)
        }
        assert Retenu == {"Gauche → droite": True, "Droite → gauche": False} or (
            Retenu == {"Gauche → droite": False, "Droite → gauche": True}
        ), f"{p1}->{p2} : un libellé ne se distingue pas de l'autre ({Retenu})"
