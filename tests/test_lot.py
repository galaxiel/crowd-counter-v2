"""Le chemin groupé (`traiter_lot`) doit rendre EXACTEMENT le chemin unitaire.

Le batching ne touche qu'à la DÉTECTION (un appel modèle par lot au lieu d'un
appel par image). Le suivi et la machine à états du comptage restent
séquentiels : un lot ne doit donc JAMAIS changer un décompte. Ce fichier
verrouille cette propriété avec le vrai `Tracker` et la vraie `Ligne` — pas
des bouchons — sur un scénario de traversées réelles.

Aucun torch ni cv2 ici : le détecteur est un bouchon jouant le même scénario
dans les deux voies.
"""

from __future__ import annotations

import numpy as np
import pytest

from compteur.compteur import Compteur
from compteur.config import Config
from compteur.ligne import Ligne
from compteur.tracker import Tracker
from compteur.types import Detection

LIGNE_X = 400.0
NB_FRAMES = 40


def _boite_tete(x: float, y: float = 200.0) -> Detection:
    """Une tête 30×30 centrée en (x, y)."""
    return Detection(x1=x - 15, y1=y - 15, x2=x + 15, y2=y + 15, score=0.9, class_id=0)


def _position(index: int, depart: float) -> float:
    """Déplacement régulier de 8 px/frame, calé pour traverser `LIGNE_X`."""
    return depart + 8.0 * index


def _scenario(index: int) -> list[Detection]:
    """Deux têtes : l'une traverse tôt, l'autre part plus tard et plus bas.

    Les deux APPARAISSANT du côté de départ (x < LIGNE_X) : une personne qui
    apparaît déjà de l'autre côté de la ligne n'est jamais comptée — c'est le
    verrou anti-recomptage, et ce test ne doit pas le contourner.
    """
    detections = []
    if 0 <= index < 20:
        detections.append(_boite_tete(_position(index, 300.0)))
    if 6 <= index < 34:
        detections.append(_boite_tete(_position(index - 6, 260.0), y=320.0))
    return detections


class DetecteurScenario:
    """Bouchon jouant le même scénario dans les DEUX voies (unitaire et lot)."""

    def __init__(self) -> None:
        self.bandes_recues = 0

    def detecter(self, img, bande=None):
        self.bandes_recues += 1
        return _scenario(int(img[0, 0, 0]))

    def detecter_lot(self, imgs, bande=None):
        self.bandes_recues += 1
        return [_scenario(int(img[0, 0, 0])) for img in imgs]


class DetecteurSansLot:
    """Bouchon SANS `detecter_lot` : le chemin de repli doit servir."""

    def detecter(self, img, bande=None):
        return _scenario(int(img[0, 0, 0]))


def _image(index: int) -> np.ndarray:
    """Image factice dont le coin code l'index : le bouchon lit son scénario."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[0, 0, 0] = index
    return img


def _config() -> Config:
    return Config(
        frames_confirmation=1,
        frames_hysteresis=1,
        fenetre_lissage=1,
        ligne=(LIGNE_X, 0.0, LIGNE_X, 480.0),
        epaisseur_bande=20,
    )


def _ligne(config: Config) -> Ligne:
    return Ligne(
        (config.ligne[0], config.ligne[1]),
        (config.ligne[2], config.ligne[3]),
        epaisseur=config.epaisseur_bande,
        sens=config.sens,
        hysteresis=config.frames_hysteresis,
    )


def _frames(nb: int) -> list[tuple[np.ndarray, int, float]]:
    return [(_image(i), i, i / 25.0) for i in range(nb)]


def _compter_pas_a_pas(frames, detecteur):
    compteur = Compteur(_config(), detecteur, Tracker(_config()))
    compteur.ajuster_ligne(_ligne(_config()))
    return [compteur.traiter_frame(img, index, ts) for img, index, ts in frames]


def _compter_par_lot(frames, detecteur, taille_lot):
    compteur = Compteur(_config(), detecteur, Tracker(_config()))
    compteur.ajuster_ligne(_ligne(_config()))
    resultats = []
    for debut in range(0, len(frames), taille_lot):
        resultats.extend(compteur.traiter_lot(frames[debut : debut + taille_lot]))
    return resultats


def test_le_scenario_traverse_vraiment():
    """Garde-fou : si le scénario ne produit aucun compte, le test ne verrouille rien."""
    resultats = _compter_pas_a_pas(_frames(NB_FRAMES), DetecteurScenario())
    assert resultats[-1].total >= 2, "le scénario doit compter les deux têtes"


def test_lot_rend_le_meme_decompte_que_le_pas_a_pas():
    frames = _frames(NB_FRAMES)
    pas_a_pas = _compter_pas_a_pas(frames, DetecteurScenario())
    # Tailles de lot variées, dont une qui ne tombe pas juste sur NB_FRAMES :
    # le lot partiel de fin de vidéo est un cas à part entière.
    for taille_lot in (2, 3, 4, 7, NB_FRAMES):
        par_lot = _compter_par_lot(frames, DetecteurScenario(), taille_lot)
        assert [r.total for r in par_lot] == [r.total for r in pas_a_pas], (
            f"taille de lot {taille_lot} : le total par frame diverge"
        )
        assert [r.frame_index for r in par_lot] == [r.frame_index for r in pas_a_pas]
        evenements_avant = [
            (r.frame_index, e.track_id) for r in pas_a_pas for e in r.evenements
        ]
        evenements_lot = [
            (r.frame_index, e.track_id) for r in par_lot for e in r.evenements
        ]
        assert evenements_lot == evenements_avant, (
            f"taille de lot {taille_lot} : événements déplacés ou recomptés"
        )


def test_lot_sans_detecter_lot_replie_sur_l_unite():
    """Un détecteur sans `detecter_lot` (tests, API ancienne) doit marcher."""
    frames = _frames(NB_FRAMES)
    pas_a_pas = _compter_pas_a_pas(frames, DetecteurScenario())
    par_lot = _compter_par_lot(frames, DetecteurSansLot(), 4)
    assert [r.total for r in par_lot] == [r.total for r in pas_a_pas]


def test_un_lot_unique_rend_le_meme_resultat_que_traiter_frame():
    """Lot de 1 : même détection, même intégration, même résultat."""
    frames = _frames(1)
    pas_a_pas = _compter_pas_a_pas(frames, DetecteurScenario())
    par_lot = _compter_par_lot(frames, DetecteurScenario(), 1)
    assert par_lot[0].total == pas_a_pas[0].total
    assert par_lot[0].frame_index == pas_a_pas[0].frame_index


def test_le_compteur_est_immune_a_la_reutilisation_d_image():
    """`traiter_lot` ne doit pas dépendre d'une mutation de l'image entre deux lots.

    La frame passée au compteur est la même numpy array qui ira à l'affichage :
    si un maillon la peignait, le comptage d'une seconde analyse divergerait.
    """
    frames = _frames(8)
    par_lot = _compter_par_lot(frames, DetecteurScenario(), 4)
    # On RECOMpte la même séquence : le total ne doit pas bouger.
    par_lot_bis = _compter_par_lot(frames, DetecteurScenario(), 4)
    assert [r.total for r in par_lot] == [r.total for r in par_lot_bis]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
