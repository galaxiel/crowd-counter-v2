import pytest

from compteur.types import Detection, Evenement


def test_center_est_le_milieu_de_la_boite():
    d = Detection(x1=10, y1=20, x2=30, y2=60, score=0.9, class_id=0)
    assert d.center == (20.0, 40.0)


def test_aire_calculee():
    d = Detection(x1=0, y1=0, x2=10, y2=5, score=0.5, class_id=0)
    assert d.aire == pytest.approx(50.0)


def test_evenement_vers_ligne_csv():
    e = Evenement(frame=42, timestamp_s=1.4, x=100.0, y=200.0, track_id=7)
    assert e.vers_ligne_csv() == "42,1.400,100.0,200.0,7"
