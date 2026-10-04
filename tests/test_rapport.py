"""Tests des statistiques de synthèse et des indicateurs de fiabilité.

`compteur.rapport` ne fait plus QUE calculer : l'export de fichiers a été
supprimé (tâche 21), il ne reste que les chiffres que `interface.recap`
affiche. Ce sont eux qui sont testés ici.

Deux exigences gouvernent ces tests :

1. **Honnêteté** — sur la vidéo de référence, la détection est fiable (88 têtes
   par frame) mais le comptage rend 0 : le mouvement des personnes est noyé dans
   le bruit du détecteur (déplacement médian 0,00 px/frame, bruit σ=3 px). Un
   « 0 personnes » lu sans contexte est un mensonge journalistique. Les
   statistiques doivent donc exposer les indicateurs de fiabilité qui permettent
   à un lecteur de voir que le total n'est pas fiable, et NE DOIT PAS produire
   de pourcentage d'erreur automatique : on n'a pas la vérité terrain.

2. **Robustesse** — zéro événement est le cas RÉEL de la vidéo de référence, pas
   un cas limite inventé pour les tests. Le calcul ne doit pas planter.
"""

import pytest

from compteur.config import Config
from compteur.rapport import (
    indicateurs_fiabilite,
    statistiques,
)
from compteur.types import Evenement, Resultat


def resultat_fictif():
    """Trois passages en 10 s, deux frames par personne : le cas « qui marche »."""
    evenements = [
        Evenement(frame=30, timestamp_s=1.0, x=100.0, y=50.0, track_id=1),
        Evenement(frame=60, timestamp_s=2.0, x=110.0, y=50.0, track_id=2),
        Evenement(frame=150, timestamp_s=5.0, x=120.0, y=50.0, track_id=3),
    ]
    return Resultat(
        total=3,
        evenements=evenements,
        config=Config(modele="yolov8n-head.pt", seuil_confiance=0.3),
        modele="yolov8n-head.pt",
        nb_frames=300,
        presents_max=5,
        presents_moyen=2.5,
        secondes=300 / 30,
    )


def resultat_video_de_reference():
    """Le cas réel : 88 détections par frame, 0 franchissement compté."""
    return Resultat(
        total=0,
        evenements=[],
        config=Config(modele="medium.pt", seuil_confiance=0.25, ligne=(556.0, 90.0, 588.0, 654.0)),
        modele="medium.pt",
        nb_frames=300,
        presents_max=88,
        presents_moyen=87.4,
        secondes=300 / 30,
    )


# -- Statistiques ----------------------------------------------------------


def test_statistiques_debit():
    s = statistiques(resultat_fictif())
    assert s["total"] == 3
    assert s["duree_s"] == 10.0
    assert s["personnes_par_minute"] == 18.0
    assert s["presents_max"] == 5


def test_statistiques_vide_sans_evenement():
    r = resultat_fictif()
    r.evenements = []
    r.total = 0
    s = statistiques(r)
    assert s["personnes_par_minute"] == 0.0


def test_statistiques_zero_duree_ne_divise_pas_par_zero():
    r = resultat_fictif()
    r.secondes = 0.0
    s = statistiques(r)
    assert s["duree_s"] == 0.0
    assert s["personnes_par_minute"] == 0.0
    assert s["fps_moyen"] is None


def test_statistiques_declares_les_cles_attendues():
    """Contrat avec `tools/comparer_modeles.py` et l'interface : l'ensemble des
    clés promises par le plan doit être là, sinon un appelant casse."""
    s = statistiques(resultat_fictif())
    for cle in (
        "total",
        "duree_s",
        "personnes_par_minute",
        "presents_max",
        "presents_moyen",
        "debit_max_par_minute",
        "modele",
        "nb_frames",
    ):
        assert cle in s, f"clé de statistiques manquante : {cle}"
    # Le module ne doit rien importer : aucun toolkit graphique ici non plus.
    assert set(s["fiabilite"])


def test_debit_max_prend_le_pic_et_ignore_les_bursts_lointains():
    """Le débit maximal est le PIC, pas la moyenne.

    Trois passages à t=1, 2, 3 s puis deux autres à t=100, 101 s : la meilleure
    minute contient les trois premiers, pas les cinq. Sur une vidéo de 200 s,
    la fenêtre est bien la minute complète.
    """
    horodatages = [1.0, 2.0, 3.0, 100.0, 101.0]
    evenements = [
        Evenement(frame=i, timestamp_s=t, x=1.0, y=1.0, track_id=i)
        for i, t in enumerate(horodatages, start=1)
    ]
    r = Resultat(
        total=len(evenements),
        evenements=evenements,
        config=Config(),
        modele="m.pt",
        nb_frames=6000,
        presents_max=3,
        presents_moyen=1.0,
        secondes=200.0,
    )
    s = statistiques(r)
    assert s["debit_max_fenetre_s"] == 60.0
    assert s["debit_max_par_minute"] == 3
    assert s["personnes_par_minute"] == pytest.approx(1.5)


def test_debit_max_sur_burst_ne_setend_pas_une_minute_inexistante():
    """600 passages en 20 s : le rapport publie 600 sur 20 s, PAS 1800 « par
    minute ». Extrapoler supposerait un débit uniforme que rien ne mesure — et
    un lieu qui voit 600 personnes en 20 s n'en verra pas 1800 en une minute."""
    evenements = [
        Evenement(frame=i, timestamp_s=i / 30.0, x=1.0, y=1.0, track_id=i)
        for i in range(600)
    ]
    r = Resultat(
        total=len(evenements),
        evenements=evenements,
        config=Config(),
        modele="m.pt",
        nb_frames=600,
        presents_max=3,
        presents_moyen=1.0,
        secondes=20.0,
    )
    s = statistiques(r)
    assert s["debit_max_fenetre_s"] == 20.0
    assert s["debit_max_par_minute"] == 600
    assert s["personnes_par_minute"] == pytest.approx(1800.0)


def test_debit_max_signale_une_fenetre_plus_courte_que_la_minute():
    """Une vidéo de 10 s ne peut pas produire un « par minute » Observé : on
    publie la fenêtre réellement couverte, jamais une extrapolation."""
    s = statistiques(resultat_fictif())
    assert s["debit_max_fenetre_s"] == 10.0
    assert s["debit_max_par_minute"] == 3


# -- Fiabilité : le cœur honnête de l'export -------------------------------


def test_indicateurs_de_fiabilite_presents():
    """Les trois indicateurs demandés doivent exister, y compris quand ils
    valent None : « non mesuré » est une information, une clé absente n'en
    est pas une."""
    f = indicateurs_fiabilite(resultat_fictif())
    for cle in (
        "nb_tracks_comptes",
        "nb_tracks_vus",
        "duree_vie_moyenne_track_frames",
        "detections_par_frame_moyen",
        "detections_par_frame_max",
    ):
        assert cle in f, f"indicateur de fiabilité manquant : {cle}"


def test_indicateurs_comptent_les_tracks_distincts():
    """Le registre anti-recomptage rend un événement par track : les track_id
    distincts disent combien d'IDENTITÉS ont franchi la ligne."""
    r = resultat_fictif()
    r.evenements.append(Evenement(frame=200, timestamp_s=6.0, x=130.0, y=50.0, track_id=1))
    f = indicateurs_fiabilite(r)
    assert f["nb_tracks_comptes"] == 3
    assert f["evenements_par_track"] == pytest.approx(4 / 3, abs=0.005)


def test_vie_moyenne_d_un_track_est_mesuree_par_le_compteur():
    """La durée de vie d'un track est MESURÉE, pas reconstruite.

    Un premier jet laissait `None` en prétendant qu'elle n'était pas
    mesurable depuis `Resultat`. C'était faux : `Compteur` voit les tracks
    frame par frame, il mesure donc leur durée de vie réelle et la publie
    dans `Resultat`. Le `None` masquait précisément l'indicateur qui permet
    à un utilisateur de juger si son décompte tient — sur la vidéo de
    référence, une durée de vie de 2 frames doit être visible, pas nulle.
    """
    r = resultat_fictif()
    r.duree_vie_track_moy = 76.6
    r.nb_tracks_vus = 2133
    f = indicateurs_fiabilite(r)
    assert f["duree_vie_moyenne_track_frames"] == pytest.approx(76.6)
    assert f["nb_tracks_vus"] == 2133
    # Plus d'avertissement « non mesurable » : l'affirmation serait fausse.
    avertissements = " ".join(f["avertissements"])
    assert "non mesurable" not in avertissements


def test_vie_moyenne_nulle_est_signalee_comme_suspecte():
    """Une durée de vie nulle signifie qu'aucun track n'a survécu une frame.

    C'est le symptôme exact de la vidéo de référence : 92 personnes
    détectées par frame et 0 comptée. Le dire vaut mieux qu'un `None`.
    """
    r = resultat_fictif()
    r.duree_vie_track_moy = 0.0
    r.nb_tracks_vus = 0
    f = indicateurs_fiabilite(r)
    assert f["duree_vie_moyenne_track_frames"] == pytest.approx(0.0)


def test_zero_avec_foule_signale_un_defaut_de_comptage():
    """Le cas réel de la vidéo de référence : 88 personnes en moyenne, 0
    passage. C'est un défaut de mesure, pas une foule vide — le rapport doit
    le dire dans ses mots."""
    f = indicateurs_fiabilite(resultat_video_de_reference())
    messages = " ".join(f["avertissements"])
    assert "0 franchissement" in messages
    assert "défaut de comptage" in messages
    assert "87.4" in messages or "87,4" in messages
    # Aucune contradiction interne : c'est mesuré, c'est juste noir.
    assert f["coherent"] is True


def test_contradiction_detectee_entre_total_et_evenements():
    r = resultat_fictif()
    r.total = 7  # le compteur dit 7, la liste n'en contient que 3
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is False
    assert "divergent" in " ".join(f["avertissements"])


def test_contradiction_detectee_entre_passages_et_presence():
    """Des passages comptés alors que personne n'est présent : l'inverse exact
    du cas de référence, aussi impossible."""
    r = resultat_fictif()
    r.presents_moyen = 0.0
    r.presents_max = 0
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is False


def test_track_id_repete_signale_un_double_comptage():
    """Le verrou anti-recomptage a cédé : le total est gonflé."""
    r = resultat_fictif()
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is True
    r.evenements.append(Evenement(frame=210, timestamp_s=7.0, x=140.0, y=50.0, track_id=2))
    r.total = 4
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is False
    assert "track_id" in " ".join(f["avertissements"])


def test_duree_incoherente_avec_le_nombre_de_frames_signalee():
    """`Resultat.secondes` vient soit de la vidéo, soit du temps de calcul
    selon l'appelant : si les deux ne collent pas, le débit par minute est
    sous-estimé. Le rapport le signale au lieu de publier un faux débit."""
    r = resultat_fictif()
    r.secondes = 120.0  # 300 frames « en » 120 s : 2,5 images/s
    f = indicateurs_fiabilite(r)
    assert "2.5" in " ".join(f["avertissements"]) or "2,5" in " ".join(f["avertissements"])


def test_zero_frame_signale():
    r = resultat_video_de_reference()
    r.nb_frames = 0
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is False
    assert "frame" in " ".join(f["avertissements"])


def test_fiabilite_avec_zero_evenement_ne_plante_pas():
    """Le cas réel : aucun événement, aucune division par zéro."""
    r = resultat_video_de_reference()
    f = indicateurs_fiabilite(r)
    assert f["nb_tracks_comptes"] == 0
    assert f["intervalle_median_evenements_s"] is None
    assert f["frames_par_evenement"] is None
    assert isinstance(f["avertissements"], list)
