import json

from compteur.config import Config, chemin_defaut_config


def test_aller_retour_dict():
    c = Config(modele="yolov8n-head.pt", seuil_confiance=0.4, frames_confirmation=5)
    retour = Config.depuis_dict(c.vers_dict())
    assert retour == c


def test_aller_retour_fichier(tmp_path):
    c = Config(modele="x.pt", epaisseur_bande=55, sens=-1)
    p = tmp_path / "c.json"
    c.vers_fichier(p)
    assert Config.depuis_fichier(p) == c


def test_ligne_none_est_acceptee():
    c = Config(modele="x.pt", ligne=None)
    assert c.depuis_dict(c.vers_dict()).ligne is None


def test_config_par_defaut_existe():
    assert chemin_defaut_config().exists(), "config/default.json doit être versionné"


def test_config_par_defaut_est_valide():
    d = json.loads(chemin_defaut_config().read_text(encoding="utf-8"))
    Config.depuis_dict(d)  # lève si une clé est inconnue ou mal typée
