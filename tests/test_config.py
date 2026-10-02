import json

import pytest

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
    """La config par défaut est chargeable sans erreur.

    `depuis_dict` rejette une clé inconnue ; le module ne valide pas les types
    au-delà des conversions tuple/liste et str/int.
    """
    d = json.loads(chemin_defaut_config().read_text(encoding="utf-8"))
    Config.depuis_dict(d)


def test_ligne_non_nulle_est_serialisee_en_liste():
    """vers_dict doit produire du JSON valide : un tuple n'est pas sérialisable."""
    c = Config(modele="x.pt", ligne=(10.0, 20.0, 30.0, 40.0))
    d = c.vers_dict()
    assert isinstance(d["ligne"], list), "JSON n'a pas de tuple"
    assert d["ligne"] == [10.0, 20.0, 30.0, 40.0]
    assert all(isinstance(v, float) for v in d["ligne"])
    # ... et le retour redonne bien un tuple de 4 floats.
    assert Config.depuis_dict(d).ligne == (10.0, 20.0, 30.0, 40.0)


def test_ligne_depuis_json_reste_un_tuple():
    c = Config.depuis_dict({"modele": "x.pt", "ligne": [1, 2, 3, 4]})
    assert c.ligne == (1.0, 2.0, 3.0, 4.0)
    assert isinstance(c.ligne, tuple)


def test_classes_retenues_non_nulles_font_l_aller_retour():
    c = Config(modele="x.pt", classes_retenues=[0, 2])
    d = c.vers_dict()
    assert d["classes_retenues"] == [0, 2]
    assert Config.depuis_dict(d).classes_retenues == [0, 2]


def test_classes_retenues_sont_converties_en_entiers():
    c = Config.depuis_dict({"modele": "x.pt", "classes_retenues": ["0", 2.0]})
    assert c.classes_retenues == [0, 2]
    assert all(isinstance(v, int) for v in c.classes_retenues)


def test_cle_inconnue_rejetee():
    with pytest.raises(ValueError, match="inconnues"):
        Config.depuis_dict({"modele": "x.pt", "parametre_inexistant": 1})


def test_defauts_lit_le_fichier_versionne():
    """defauts() doit renvoyer ce que contient config/default.json."""
    assert Config.defauts() == Config.depuis_fichier(chemin_defaut_config())


def test_defauts_et_valeurs_du_dataclass_ne_divergent_pas():
    """Le fichier versionné et les valeurs par défaut du dataclass coincident.

    Toute valeur par défaut doit lire config/default.json : si le fichier et le
    dataclass divergent, Config() (utilisé quand le fichier manque) et
    Config.defauts() ne décrivent plus la même application.
    """
    assert Config() == Config.depuis_fichier(chemin_defaut_config())
