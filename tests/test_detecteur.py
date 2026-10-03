"""Détection : image numpy -> liste de boîtes.

Le faux modèle reproduit l'API réelle d'ultralytics, à laquelle le détecteur
est couplé :

- ``modele(img, **kw)`` renvoie une **liste** de résultats (une image ->
  un ``Results``), pas l'objet lui-même ;
- ``resultat.boxes`` est un **attribut** (``Boxes`` ou ``None``) ;
- ``boxes.xyxy``, ``boxes.conf`` et ``boxes.cls`` sont des **propriétés**
  renvoyant des tenseurs, pas des méthodes ;
- ``modele.names`` est un **dict** ``{id: nom}``.

Un faux qui s'écarterait de cette forme testerait le faux, pas le code réel :
les erreurs d'API ci-dessus ont été constatées sur le premier jet de ce plan.
"""

import numpy as np
import pytest

from compteur.config import Config
from compteur.detecteur import Detecteur, charger_modele
from compteur.types import Detection


class FauxBoites:
    """Émulat d'`ultralytics.engine.results.Boxes` : attributs tenseurs."""

    def __init__(self, brut):
        self.brut = list(brut)

    def __len__(self):
        return len(self.brut)

    @property
    def xyxy(self):
        return np.array([b[:4] for b in self.brut], dtype=float).reshape(-1, 4)

    @property
    def conf(self):
        return np.array([b[4] for b in self.brut], dtype=float)

    @property
    def cls(self):
        return np.array([b[5] for b in self.brut], dtype=float)


class FauxResultat:
    """Émulat d'`ultralytics.engine.results.Results`."""

    def __init__(self, brut):
        self.boxes = None if brut is None else FauxBoites(brut)


class FauxYOLO:
    """Substitut de YOLO : renvoie toujours la même liste de détections brutes."""

    def __init__(self, brut, noms=None):
        self.brut = None if brut is None else list(brut)
        self.names = {0: "head", 1: "person"} if noms is None else noms
        self.appels = []

    def __call__(self, img, **kwargs):
        self.appels.append(kwargs)
        self.derniere_img = img
        return [FauxResultat(self.brut)]


def image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# --- filtrage -----------------------------------------------------------


def test_filtre_par_seuil_de_confiance():
    """0.9 passe, 0.1 tombe sous le seuil de 0.5."""
    brut = [(0, 0, 100, 100, 0.9, 0), (200, 200, 300, 300, 0.1, 0)]
    d = Detecteur(FauxYOLO(brut), Config(seuil_confiance=0.5))
    dets = d.detecter(image())
    assert len(dets) == 1
    assert dets[0].score == pytest.approx(0.9)


def test_seuil_passe_au_modele():
    """Le seuil est transmis à ultralytics, qui élague en amont."""
    f = FauxYOLO([])
    Detecteur(f, Config(seuil_confiance=0.42)).detecter(image())
    assert f.appels[0]["conf"] == 0.42


def test_filtre_par_taille_minimale():
    """5x5 sous le minimum de 30 px, 100x100 le dépasse."""
    petit = (0, 0, 5, 5, 0.9, 0)
    grand = (0, 0, 100, 100, 0.9, 0)
    d = Detecteur(FauxYOLO([petit, grand]), Config(taille_min_px=30))
    dets = d.detecter(image())
    assert len(dets) == 1
    assert (dets[0].x2 - dets[0].x1) == pytest.approx(100)


def test_taille_minimale_appliquee_a_la_hauteur():
    """Une boîte large mais basse est rejetée : la hauteur compte aussi."""
    plate = (0, 0, 100, 10, 0.9, 0)  # 100 de large, 10 de haut
    d = Detecteur(FauxYOLO([plate]), Config(taille_min_px=30))
    assert d.detecter(image()) == []


def test_taille_minimale_appliquee_a_la_largeur():
    """Symétrique : une boîte étroite mais haute est rejetée."""
    etroite = (0, 0, 10, 100, 0.9, 0)  # 10 de large, 100 de haut
    d = Detecteur(FauxYOLO([etroite]), Config(taille_min_px=30))
    assert d.detecter(image()) == []


def test_taille_egale_au_minimum_conservee():
    """Le seuil de taille est inclusif : exactement mini, on garde."""
    pile = (0, 0, 30, 30, 0.9, 0)
    d = Detecteur(FauxYOLO([pile]), Config(taille_min_px=30))
    assert len(d.detecter(image())) == 1


def test_filtre_par_classe():
    brut = [(0, 0, 100, 100, 0.9, 0), (200, 200, 300, 300, 0.9, 1)]
    d = Detecteur(FauxYOLO(brut), Config(classes_retenues=[1]))
    dets = d.detecter(image())
    assert len(dets) == 1
    assert dets[0].class_id == 1


def test_toutes_classes_retenues_si_aucune_liste():
    """`classes_retenues=None` ne filtre rien."""
    brut = [(0, 0, 100, 100, 0.9, 0), (200, 200, 300, 300, 0.9, 7)]
    d = Detecteur(FauxYOLO(brut), Config(classes_retenues=None))
    assert len(d.detecter(image())) == 2


def test_liste_de_classes_vide_ne_retenue_rien():
    """Liste vide = aucune classe voulue, ce n'est pas « pas de filtre »."""
    d = Detecteur(FauxYOLO([(0, 0, 100, 100, 0.9, 0)]), Config(classes_retenues=[]))
    assert d.detecter(image()) == []


# --- conversion en Detection -------------------------------------------


def test_coordonnes_bien_lues():
    d = Detecteur(FauxYOLO([(10, 20, 110, 220, 0.9, 0)]), Config())
    det = d.detecter(image())[0]
    assert (det.x1, det.y1, det.x2, det.y2) == (10, 20, 110, 220)
    assert det.score == pytest.approx(0.9)
    assert det.class_id == 0


def test_valeurs_converties_en_types_python():
    """Pas de np.float64 qui traîne : les champs sont annotés float/int.

    `type(x) is float` et non `isinstance` : np.float64 hérite de float, donc
    isinstance(dirait vert alors que l'annotation ment. (Vérifié par mutation.)
    """
    d = Detecteur(FauxYOLO([(10, 20, 110, 220, 0.9, 0)]), Config())
    det = d.detecter(image())[0]
    assert type(det.x1) is float
    assert type(det.y1) is float
    assert type(det.x2) is float
    assert type(det.y2) is float
    assert type(det.score) is float
    assert type(det.class_id) is int
    assert isinstance(det, Detection)


def test_coordonnes_decimales_preservees():
    """Un tenseur float32 ne doit pas être arrondi au pixel entier."""
    d = Detecteur(FauxYOLO([(10.5, 20.25, 110.75, 220.5, 0.93, 1)]), Config())
    det = d.detecter(image())[0]
    assert det.x1 == pytest.approx(10.5)
    assert det.y1 == pytest.approx(20.25)
    assert det.x2 == pytest.approx(110.75)
    assert det.y2 == pytest.approx(220.5)
    assert det.score == pytest.approx(0.93)


def test_vide_si_aucune_detection():
    d = Detecteur(FauxYOLO([]), Config())
    assert d.detecter(image()) == []


def test_vide_si_modele_ne_detecte_rien_dans_une_scene():
    """Une scène sans personne ne doit pas faire planter la boucle."""
    d = Detecteur(FauxYOLO([]), Config())
    assert all(d.detecter(np.zeros((720, 1280, 3), dtype=np.uint8)) == [] for _ in range(3))


def test_vide_si_boxes_est_none():
    """Ultralytics renvoie `boxes=None` pour un modèle de classification."""
    d = Detecteur(FauxYOLO(None), Config())
    assert d.detecter(image()) == []


# --- classes et taille d'entrée ----------------------------------------


def test_noms_classes_exposes():
    d = Detecteur(FauxYOLO([]), Config())
    assert d.noms_classes() == {0: "head", 1: "person"}


def test_noms_classes_estune_copie():
    """Le dict renvoyé ne doit pas être l'état interne du modèle.

    Sans copie, l'interface qui ajoute un libellé en français ecraserait la
    table du modèle pour toutes les frames suivantes.
    """
    f = FauxYOLO([])
    noms = Detecteur(f, Config()).noms_classes()
    noms[0] = "tete"
    noms[99] = "ajout"
    assert f.names == {0: "head", 1: "person"}


def test_noms_classes_depuis_une_liste():
    """Certaines versions exposent `names` en liste : on la convertit."""
    d = Detecteur(FauxYOLO([], noms=["head", "person"]), Config())
    assert d.noms_classes() == {0: "head", 1: "person"}


def test_noms_classes_vides_si_absents():
    d = Detecteur(FauxYOLO([]), Config())
    d.modele.names = None
    assert d.noms_classes() == {}


def test_taille_entree_modifiable():
    """Le changement se voit au appel suivant, via `imgsz`."""
    f = FauxYOLO([])
    d = Detecteur(f, Config())
    d.detecter(image())
    assert f.appels[0]["imgsz"] == 640
    d.changer_taille_entree(1280)
    d.detecter(image())
    assert f.appels[1]["imgsz"] == 1280
    assert d.config.taille_entree == 1280


def test_appel_silencieux_au_modele():
    """`verbose=False` : la console ne doit pas être noyée de logs ultralytics."""
    f = FauxYOLO([])
    Detecteur(f, Config()).detecter(image())
    assert f.appels[0]["verbose"] is False


def test_image_transmise_telle_quelle():
    """Le détecteur ne réencode pas l'image avant de l'envoyer au modèle."""
    f = FauxYOLO([])
    img = image()
    Detecteur(f, Config()).detecter(img)
    assert f.appels[0] is not None
    assert f.derniere_img is img


# --- chargement du modèle ----------------------------------------------


def test_charger_modele_echoue_si_fichier_absent(tmp_path):
    """Le message doit nommer le fichier, pour être actionnable."""
    manquant = tmp_path / "absent.pt"
    with pytest.raises(FileNotFoundError) as e:
        charger_modele(str(manquant))
    assert "absent.pt" in str(e.value)


def test_charger_modele_refuse_un_format_inconnu(tmp_path):
    fichier = tmp_path / "modele.txt"
    fichier.write_text("pas un modele", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        charger_modele(str(fichier))
    assert ".txt" in str(e.value)


def test_charger_modele_ne_depend_pas_de_torch_pour_valider(tmp_path):
    """La validation du chemin précède l'import de torch/ultralytics.

    Sans cela, `charger_modele` lèverait ModuleNotFoundError sur une machine
    sans torch au lieu du FileNotFoundError promis par l'API.
    """
    import inspect

    source = inspect.getsource(charger_modele)
    assert "import torch" in source, "torch doit être importé (paresseusement)"
    assert source.index("FileNotFoundError") < source.index("import torch")
