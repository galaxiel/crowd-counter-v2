"""Tests du récapitulatif de fin d'analyse (tâche 20).

Deux tests seulement, parce que cette tâche est de l'AFFICHAGE. Le rendu
pixel par pixel d'une courbe n'est pas un test : il échouerait sur un arrondi
de Qt sans rien dire de faux sur le comportement. Ce qui compte, et que ces
deux tests couvrent, c'est que les chiffres affichés soient ceux du rapport
d'export — pas une deuxième version de la vérité — et que la courbe ait bien
un point par minute de l'analyse, minutes vides comprises.
"""

import os

# Doit précéder la création de QApplication (voir test_widgets_video.py).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from compteur.types import Evenement, Resultat  # noqa: E402
from interface.recap import (  # noqa: E402
    LIBELLE_DEBIT,
    LIBELLE_DUREE,
    LIBELLE_PIC,
    LIBELLE_TOTAL,
    PanneauRecap,
    formater_duree,
    points_de_debit,
)


@pytest.fixture
def application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def resultat_de_test() -> Resultat:
    """3 minutes de vidéo, 13 personnes : 4, 5, puis 4 par tranche de minute.

    Le maximum par TRANCHE est 5, sur la deuxième minute. Le pic du rapport,
    lui, est un maximum en FENÊTRE GLISSANTE de 60 s, et il vaut 6 : la fenêtre
    [10 s, 70 s] attrape les 4 premières personnes de la première minute plus
    la première de la deuxième. C'est 6, pas 5, et la première fenêtre qui
    atteint 6 commence à 70 s.

    Les deux chiffres sont différents par construction, et c'est ce que ce
    test verrouille : afficher 5 parce que c'est le maximum de la courbe, ou
    afficher 6 au bon endroit, sont deux lectures de la même mesure.
    Les mélanger reviendrait à annoncer un chiffre que personne n'a mesuré.
    """
    horodatages = [
        5.0,
        15.0,
        25.0,
        35.0,  # minute 1 : 4 personnes
        70.0,
        85.0,
        95.0,
        105.0,
        115.0,  # minute 2 : 5 personnes, étalées sur toute la tranche
        130.0,
        150.0,
        170.0,
        175.0,  # minute 3 : 4 personnes
    ]
    evenements = [
        Evenement(frame=int(t * 25), timestamp_s=t, x=100.0, y=50.0, track_id=i)
        for i, t in enumerate(horodatages)
    ]
    return Resultat(
        total=len(evenements),
        evenements=evenements,
        modele="yolov8n-head.pt",
        nb_frames=int(180.0 * 25),
        presents_max=40,
        presents_moyen=22.0,
        # 95 s de temps de CALCUL pour 180 s de vidéo : c'est la durée vidéo
        # qui doit diviser le total, sinon le débit est sous-estimé d'environ
        # deux fois.
        secondes=95.0,
        duree_video_s=180.0,
    )


def test_recap_affiche_les_chiffres_du_rapport_et_une_courbe_par_minute(application):
    """Les valeurs affichées et la longueur de la courbe sont justes."""
    panneau = PanneauRecap()
    panneau.afficher(resultat_de_test())

    assert panneau.valeurs[LIBELLE_TOTAL].text() == "13"
    # 180 s de vidéo : c'est cette durée qui divise le total, pas les 95 s de
    # temps de calcul.
    assert panneau.valeurs[LIBELLE_DUREE].text() == "3 min 00 s"
    assert formater_duree(161.0) == "2 min 41 s"
    # 13 personnes en 3 minutes : 4,3 par minute, arrondi à 4.
    assert panneau.valeurs[LIBELLE_DEBIT].text() == "4 pers/min"

    # 6 événements dans la fenêtre de 60 s qui suit 70 s. L'instant affiché
    # est celui du DÉBUT de la fenêtre.
    assert panneau.valeurs[LIBELLE_PIC].text() == "6 pers/min à 1 min 10 s"

    points = panneau.courbe.points()
    assert [t for t, _ in points] == [0.0, 60.0, 120.0]
    assert [v for _, v in points] == [4, 5, 4]


def test_points_de_debit_garde_les_minutes_vides(application):
    """Une minute sans franchissement est un point à zéro, pas un trou.

    Une courbe où les minutes vides disparaîtraient se lirait « personne ne
    passe » entre deux points collés, alors que la réalité est « personne ne
    passe PENDANT une minute entière » — ce n'est pas la même information pour
    un opérateur qui cherche le moment du pic.
    """
    resultat = Resultat(
        total=1,
        evenements=[Evenement(frame=25, timestamp_s=10.0, x=1.0, y=1.0, track_id=1)],
        duree_video_s=180.0,
        secondes=90.0,
    )
    points = points_de_debit(resultat)
    assert points == [(0.0, 1), (60.0, 0), (120.0, 0)]

    # Durée nulle : aucune tranche, aucune courbe — et surtout pas de division
    # par zéro.
    assert points_de_debit(Resultat(total=0, evenements=[], duree_video_s=0.0)) == []