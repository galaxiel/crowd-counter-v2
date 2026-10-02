import dataclasses

import numpy as np
import pytest

from compteur.types import Detection, Evenement, FrameResult


def test_center_est_le_milieu_de_la_boite():
    d = Detection(x1=10, y1=20, x2=30, y2=60, score=0.9, class_id=0)
    assert d.center == (20.0, 40.0)


def test_aire_calculee():
    d = Detection(x1=0, y1=0, x2=10, y2=5, score=0.5, class_id=0)
    assert d.aire == pytest.approx(50.0)


def test_evenement_vers_ligne_csv():
    e = Evenement(frame=42, timestamp_s=1.4, x=100.0, y=200.0, track_id=7)
    assert e.vers_ligne_csv() == "42,1.400,100.0,200.0,7"


def test_frame_result_na_pas_de_champ_redondant():
    """`indice` doublait `frame_index` et n'était consommé par aucune tâche.

    Le test verrouille l'ensemble des champs pour qu'une tâche ultérieure
    n'introduise pas un second index de frame qui divergerait du premier.
    """
    noms = [f.name for f in dataclasses.fields(FrameResult)]
    assert noms == [
        "image",
        "detections",
        "tracks",
        "total",
        "presents",
        "frame_index",
        "timestamp_s",
        "evenements",
    ]
    assert "indice" not in noms


def test_frame_result_se_construit_par_mots_cles():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    r = FrameResult(image=img, frame_index=7, timestamp_s=0.35)
    assert r.frame_index == 7
    assert r.timestamp_s == 0.35
    assert r.detections == [] and r.tracks == [] and r.evenements == []
    assert r.total == 0 and r.presents == 0
