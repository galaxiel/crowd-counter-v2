"""Tests du comptage — le moteur, pas l'interface.

Harnais de test : la géométrie réelle de `Ligne` est utilisée, le tracker est
remplacé par un script de positions, le détecteur par un bouchon qui ne rend
rien (le faux tracker joue un script et ignore les détections). Ce qui est
éprouvé ici est la machine à états de `Compteur`, pas YOLO ni ByteTrack.
"""

import numpy as np
import pytest

from compteur.compteur import Compteur, analyser_video
from compteur.config import Config
from compteur.ligne import Ligne
from compteur.types import Detection, Track


def image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def t(identifiant, x, y=50.0):
    """Un track confirmé, positionné en x le long d'une ligne verticale."""
    return Track(
        track_id=identifiant,
        center=(x, y),
        bbox=(x - 5.0, y - 5.0, x + 5.0, y + 5.0),
        age=5,
        confirmed=True,
    )


def marche(identifiant, xs, y=50.0):
    """Un scénario d'un seul track : une position x par frame."""
    return [[t(identifiant, x, y)] for x in xs]


def pas_a_pas(x_depart, x_fin, pas):
    """Positions successives d'une marche régulière, extrémités incluses."""
    nombre = int(round((x_fin - x_depart) / pas))
    return [x_depart + i * pas for i in range(nombre + 1)]


class FauxTracker:
    """Joue un scénario : une liste de tracks par frame, puis plus rien."""

    def __init__(self, scenario):
        self.scenario = list(scenario)
        self.index = 0
        self.config = None

    def mettre_a_jour(self, detections, frame_index):
        sortie = self.scenario[self.index] if self.index < len(self.scenario) else []
        self.index += 1
        return list(sortie)

    def reinitialiser(self):
        self.index = 0

    def nb_tracks_vus(self):
        return self.index


class FauxDetecteur:
    """Bouchon : le faux tracker ignore les détections et joue son script."""

    def __init__(self):
        self.config = None

    def detecter(self, img):
        return []


def ligne_de(config, sens=1):
    """Ligne verticale x=320, avec les réglages géométriques de `config`.

    Comme dans `analyser_video` : `epaisseur_bande` et `frames_hysteresis`
    viennent de la configuration, pas d'une constante du test. Sans cela,
    `frames_hysteresis` serait silencieusement ignoré par les tests.
    """
    return Ligne(
        p1=(320.0, 0.0),
        p2=(320.0, 480.0),
        epaisseur=config.epaisseur_bande,
        sens=sens,
        hysteresis=config.frames_hysteresis,
    )


def compteur(scenario, sens=1, avec_ligne=True, **kw):
    config = Config(frames_confirmation=1, **kw)
    c = Compteur(config, FauxDetecteur(), FauxTracker(scenario))
    if avec_ligne:
        c.ajuster_ligne(ligne_de(config, sens))
    return c


def jouer(c, nb_frames, premier_index=0, pas_s=0.04):
    for i in range(nb_frames):
        c.traiter_frame(image(), premier_index + i, (premier_index + i) * pas_s)
    return c


# Géométrie de référence, avec epaisseur_bande=20 et sens=1 :
#   ligne verticale x=320, donc coordonnee_projetee(x) = 320 - x
#   point_du_cote : +1 si x < 310,  -1 si x > 330,  0 entre les deux (bande)
#   a_traverse est vrai sur la SEULE frame où la coordonnée passe de > 0 à
#   <= 0, c'est-à-dire quand le centre saute de x < 320 à x >= 320.
# Marche de référence : 100 -> 200 -> 300 -> 340. Le franchissement est
# détecté sur la frame où le track est en x=340, soit l'index 3.
REFERENCE = [100.0, 200.0, 300.0, 340.0, 400.0, 460.0, 520.0]


def test_traversal_dans_le_sens_compte_une_fois():
    """Un ID stable traversant la ligne de gauche à droite compte exactement 1."""
    c = compteur(marche(1, REFERENCE), epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(REFERENCE))
    assert c.total == 1
    assert len(c.evenements) == 1
    assert c.evenements[0].track_id == 1
    assert c.evenements[0].frame == 3


@pytest.mark.parametrize("vitesse", [5, 20, 80], ids=lambda v: f"{v}px/frame")
def test_marche_continue_compte_exactement_une_fois(vitesse):
    """La vitesse de marche ne change rien au nombre de comptages.

    Paramétré sur la vitesse ET maintenu à `frames_hysteresis=2` : le seuil ne
    doit pas devenir inatteignable quand la personne franchit en une seule
    frame. C'était le défaut de la version précédente, où le compteur de
    frames consécutives ne pouvait dépasser 1 et rendait tout seuil > 1
    impossible à satisfaire — donc, avec le défaut `frames_hysteresis=2` de
    config/default.json, personne n'était jamais compté.
    """
    xs = pas_a_pas(40.0, 600.0, vitesse)
    c = compteur(marche(1, xs), epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(xs))
    assert c.total == 1, f"{vitesse} px/frame doit compter 1 fois, pas {c.total}"


@pytest.mark.parametrize("hysteresis", [0, 2], ids=lambda h: f"hysteresis={h}")
def test_bruit_de_detection_autour_de_la_ligne_ne_compte_rien(hysteresis):
    """Une personne qui oscille d'un pixel autour de la ligne ne franchit pas.

    Test verrou de l'algorithme. À ±1 px de la ligne, `point_du_cote` renvoie 0
    à chaque frame : aucun côté de départ n'est donc jamais mémorisé, et le
    franchissement que `a_traverse` signale à chaque passage 319 -> 320 est
    rejeté faute d'état. Le paramétrage sur `hysteresis=0` prouve que le
    garde-fou est l'état mémorisé, et pas le délai.
    """
    scenario = marche(1, [319.0 if f % 2 else 320.0 for f in range(40)])
    c = compteur(scenario, epaisseur_bande=20, frames_hysteresis=hysteresis)
    jouer(c, len(scenario))
    assert c.total == 0
    assert c.evenements == []


def test_vingt_personnes_ayent_des_identifiants_distincts_donnent_vingt():
    """Vingt personnes distinctes franchissant la ligne = vingt comptages.

    Décalage d'une frame entre elles : les 20 identifiants sont vus dans des
    frames différentes, et aucun registre n'est écrasé par un autre.
    """
    n = 20
    scenario = []
    for f in range(n + 8):
        tracks = []
        for k in range(n):
            if f < k:
                continue
            pas = min(f - k, len(REFERENCE) - 1)
            tracks.append(t(k + 1, REFERENCE[pas], 12.0 * (k + 1)))
        scenario.append(tracks)

    c = compteur(scenario, epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(scenario))
    assert c.total == 20
    assert len(c.evenements) == 20
    assert len({ev.track_id for ev in c.evenements}) == 20


def test_aller_retour_ne_compte_qu_une_seule_fois():
    """Un aller-retour sur la ligne compte 1, pas 2."""
    xs = [100.0, 200.0, 300.0, 340.0, 400.0, 300.0, 200.0, 100.0, 200.0, 340.0]
    c = compteur(marche(7, xs), epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(xs))
    assert c.total == 1
    assert c.evenements[0].frame == 3


@pytest.mark.parametrize(
    "frames_depart, attendu",
    [(1, 0), (2, 1), (3, 1)],
    ids=["1 frame -> 0", "2 frames -> 1", "3 frames -> 1"],
)
def test_hysteresis_exige_n_frames_du_cote_depart(frames_depart, attendu):
    """Nouvelle sémantique de `frames_hysteresis` : un délai AVANT le comptage.

    `frames_hysteresis=2` veut dire « la personne doit avoir été vue du côté
    de départ sur au moins 2 frames avant que je compte ». Voir ce côté une
    seule fois ne suffit pas ; en voir trois suffit largement. L'ancien sens
    (« nombre de frames consécutives pendant lesquelles a_traverse est vrai »)
    était inatteignable, `a_traverse` n'étant vrai que sur une frame.
    """
    xs = [100.0] * frames_depart + [500.0]
    c = compteur(marche(7, xs), epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(xs))
    assert c.total == attendu


def test_presents_moyen_est_une_moyenne_sur_toutes_les_frames():
    """`presents_moyen` est la moyenne des présents sur TOUTES les frames."""
    scenario = [
        [t(k, 50.0 + 20.0 * k) for k in range(2)],
        [t(k, 50.0 + 20.0 * k) for k in range(4)],
        [t(k, 50.0 + 20.0 * k) for k in range(6)],
    ]
    c = compteur(scenario, epaisseur_bande=20)
    jouer(c, len(scenario))
    r = c.resultat("x.pt", 3, 0.12)
    assert r.presents_moyen == pytest.approx(4.0)
    assert r.presents_max == 6
    assert r.total == 0
    assert r.nb_frames == 3


def test_presents_refletent_le_nombre_de_tracks_confirmes():
    c = compteur([[t(1, 10.0, 10.0), t(2, 180.0, 90.0)]], epaisseur_bande=20)
    r = c.traiter_frame(image(), 0, 0.0)
    assert r.presents == 2
    assert r.total == 0


def test_evenement_horodate_et_localise_le_comptage():
    # Le faux tracker rend les positions dans l'ordre : il faut que la
    # troisième frame soit celle du franchissement, d'où 340 et non 300.
    c = compteur(marche(7, [100.0, 200.0, 340.0]), epaisseur_bande=20, frames_hysteresis=2)
    c.traiter_frame(image(), 0, 0.0)
    c.traiter_frame(image(), 1, 0.04)
    r = c.traiter_frame(image(), 42, 1.68)
    ev = c.evenements[0]
    assert (ev.frame, ev.track_id) == (42, 7)
    assert ev.timestamp_s == pytest.approx(1.68)
    assert (ev.x, ev.y) == (340.0, 50.0)
    assert r.evenements == [ev], "la frame ne publie que ses propres evenements"
    assert r.total == 1


def test_purge_des_registres_quand_le_tracker_perd_un_track():
    """Un track disparu du tracker ne doit pas laisser d'entrée derrière lui.

    Sans purge, `_cotes` et `_stabilite` grossiraient sans borne sur une vidéo
    longue : une entrée par personne jamais revue.
    """
    scenario = marche(1, REFERENCE) + [[]]
    c = compteur(scenario, epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(REFERENCE))
    assert c.total == 1
    assert c._derniers_centers and c._cotes and c._stabilite

    r = c.traiter_frame(image(), len(REFERENCE), 0.28)
    assert r.presents == 0
    assert c._derniers_centers == {}
    assert c._cotes == {}
    assert c._stabilite == {}


def test_identifiant_deja_compte_survit_a_la_perte_du_track():
    """Le registre anti-recomptage n'est PAS purgé avec les autres.

    Un identifiant réapparu après une occlusion, qui recroise la ligne, ne doit
    pas être compté une seconde fois.
    """
    scenario = marche(1, REFERENCE) + [[]] + marche(1, REFERENCE)
    c = compteur(scenario, epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(scenario))
    assert c.total == 1
    assert c._deja_comptes == {1}


def test_traversee_dans_le_mauvais_sens_ne_compte_pas():
    """Marcher dans le sens opposé à la normale ne compte personne."""
    xs = list(reversed(REFERENCE))
    c = compteur(marche(1, xs), epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(xs))
    assert c.total == 0


def test_sens_inverse_retient_le_passage_oppose():
    """`sens = -1` inverse la normale : c'est l'autre passage qui compte."""
    c = compteur(
        marche(1, REFERENCE), sens=-1, epaisseur_bande=20, frames_hysteresis=2
    )
    jouer(c, len(REFERENCE))
    assert c.total == 0

    c2 = compteur(
        marche(1, list(reversed(REFERENCE))),
        sens=-1,
        epaisseur_bande=20,
        frames_hysteresis=2,
    )
    jouer(c2, len(REFERENCE))
    assert c2.total == 1
    # Le passage retenu n'est pas le miroir exact de l'autre : l'intervalle
    # semi-ouvert de `a_traverse` (> 0 puis <= 0) n'est pas symétrique, donc
    # sur la marche inversée le franchissement tombe sur l'index 4 (300 -> 340
    # vues dans le repère inversé) et non sur 3.
    assert c2.evenements[0].frame == 4


def test_sans_ligne_rien_n_est_comptte():
    c = compteur(marche(1, REFERENCE), avec_ligne=False, epaisseur_bande=20)
    jouer(c, len(REFERENCE))
    assert c.total == 0
    assert c.evenements == []
    assert c.presents == 1


def test_track_apparu_apres_la_ligne_n_est_pas_compte():
    """Une personne qui apparaît déjà passée n'est pas un franchissement."""
    c = compteur(marche(1, [500.0, 560.0, 620.0]), epaisseur_bande=20)
    jouer(c, 3)
    assert c.total == 0

    # Même chose au milieu de la bande : aucun côté de départ n'est connu.
    c2 = compteur(marche(2, [320.0, 400.0]), epaisseur_bande=20)
    jouer(c2, 2)
    assert c2.total == 0


def test_reinitialiser_oublie_tout():
    c = compteur(marche(1, REFERENCE), epaisseur_bande=20, frames_hysteresis=2)
    jouer(c, len(REFERENCE))
    assert c.total == 1
    reference_evenements = c.evenements

    c.reinitialiser()
    assert c.total == 0
    assert c.evenements == []
    assert c._derniers_centers == {}
    assert c._cotes == {}
    assert c._stabilite == {}
    assert c._deja_comptes == set()
    # La liste est vidée EN PLACE : l'IHM (tâche 11) en garde une référence
    # pour son panneau d'audit, une réassignation la laisserait sur l'ancienne.
    assert reference_evenements is c.evenements

    # Le faux tracker est remis à zéro : le même scénario rejoué recompte.
    c.ajuster_ligne(ligne_de(c.config))
    jouer(c, len(REFERENCE))
    assert c.total == 1


class FauxCapture:
    """Remplace `cv2.VideoCapture` : rend des frames puis signale la fin.

    Le codec vidéo de la machine de test n'est pas fiable (OpenCV 5.0 écrit
    un AVI que son propre lecteur refuse ensuite), et un test d'orchestration
    n'a rien à voir avec un encodeur. On fixe donc la source.
    """

    def __init__(self, chemin, nb_frames=6, fps=25.0):
        import cv2

        self.chemin = chemin
        self.nb_frames = nb_frames
        self.fps = fps
        self.ouvert = True
        self.index = 0
        self.relache = False
        self._prop_fps = cv2.CAP_PROP_FPS
        self._prop_nb = cv2.CAP_PROP_FRAME_COUNT

    def isOpened(self):
        return self.ouvert

    def get(self, propriete):
        if propriete == self._prop_fps:
            return self.fps
        if propriete == self._prop_nb:
            return float(self.nb_frames)
        return 0.0

    def read(self):
        if self.index >= self.nb_frames:
            return False, None
        img = image()
        img[0, 0] = 10 + self.index  # marqueur : chaque frame est identifiable
        self.index += 1
        return True, img

    def release(self):
        self.relache = True


def _installer_capture(monkeypatch, capture):
    """Branche le faux capturage sur `analyser_video`, qui importe cv2 sur place."""
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(cv2, "VideoCapture", lambda chemin, *a, **kw: capture)
    return capture


def test_video_illisible_signalee(monkeypatch):
    capture = _installer_capture(monkeypatch, FauxCapture("absent.avi"))
    capture.ouvert = False
    with pytest.raises(FileNotFoundError, match="illisible"):
        analyser_video(
            "absent.avi",
            Config(ligne=(320.0, 0.0, 320.0, 480.0)),
            detecteur=FauxDetecteur(),
            tracker=FauxTracker([]),
        )


def test_analyser_video_boucle_jusqu_a_la_fin(monkeypatch):
    """Le chef d'orchestration lit la vidéo, publie chaque frame, rend un bilan."""
    capture = _installer_capture(monkeypatch, FauxCapture("synthetique.avi"))
    vues = []
    config = Config(
        frames_confirmation=1,
        ligne=(320.0, 0.0, 320.0, 480.0),
        epaisseur_bande=20,
        frames_hysteresis=2,
    )
    r = analyser_video(
        "synthetique.avi",
        config,
        callback_frame=vues.append,
        detecteur=FauxDetecteur(),
        tracker=FauxTracker(marche(1, REFERENCE)),
    )
    assert r.total == 1
    assert r.nb_frames == 6
    assert r.config is config
    assert r.modele == config.modele
    assert [f.frame_index for f in vues] == [0, 1, 2, 3, 4, 5]
    assert [f.total for f in vues] == [0, 0, 0, 1, 1, 1]
    # Les timestamps viennent de l'index divisé par le fps de la vidéo.
    assert [f.timestamp_s for f in vues] == pytest.approx(
        [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    )
    # Le callback reçoit bien les images lues, dans l'ordre.
    assert [int(f.image[0, 0, 0]) for f in vues] == [10, 11, 12, 13, 14, 15]
    assert r.evenements[0].frame == 3
    assert r.evenements[0].track_id == 1
    assert r.presents_max == 1
    assert r.presents_moyen == pytest.approx(1.0)
    assert capture.relache, "le capturage doit être relâché en fin de lecture"


def test_analyser_video_sans_callback_ne_crashe_pas(monkeypatch):
    _installer_capture(monkeypatch, FauxCapture("synthetique.avi"))
    config = Config(ligne=(320.0, 0.0, 320.0, 480.0), epaisseur_bande=20)
    r = analyser_video(
        "synthetique.avi",
        config,
        detecteur=FauxDetecteur(),
        tracker=FauxTracker(marche(1, REFERENCE)),
    )
    assert r.nb_frames == 6
    assert r.total == 1
    assert r.presents_moyen == pytest.approx(1.0)


def test_analyser_video_construit_la_ligne_depuis_la_config(monkeypatch):
    """`sens` de la config doit atteindre la ligne.

    Sans ce test, un oubli dans la construction du `Ligne` passerait
    inaperçu : la ligne garderait `sens=1` quel que soit le réglage, et
    l'utilisateur compterait les passages dans le mauvais sens.
    """
    def _analyser(sens, scenario):
        # Un capturage neuf par analyse : le précédent est épuisé.
        _installer_capture(monkeypatch, FauxCapture("synthetique.avi"))
        config = Config(
            frames_confirmation=1,
            ligne=(320.0, 0.0, 320.0, 480.0),
            epaisseur_bande=20,
            sens=sens,
            frames_hysteresis=2,
        )
        return analyser_video(
            "synthetique.avi",
            config,
            detecteur=FauxDetecteur(),
            tracker=FauxTracker(marche(1, scenario)),
        ).total

    assert _analyser(1, REFERENCE) == 1
    assert _analyser(1, list(reversed(REFERENCE))) == 0
    assert _analyser(-1, list(reversed(REFERENCE))) == 1
    assert _analyser(-1, REFERENCE) == 0


def test_le_faux_tracker_bien_joue_le_scenario():
    """Non-vacuité du harnais : le faux tracker rend bien les positions prévues.

    Sans ce test, une régression qui ferait ignorer les positions par le
    faux tracker passerait tous les autres tests au vert.
    """
    c = compteur(marche(1, REFERENCE), epaisseur_bande=20)
    vus = [c.traiter_frame(image(), f, f * 0.04).tracks for f in range(3)]
    assert [x[0].center[0] for x in vus] == [100.0, 200.0, 300.0]
    assert [len(x) for x in vus] == [1, 1, 1]


def test_une_detection_bien_construite():
    d = Detection(x1=0.0, y1=0.0, x2=10.0, y2=10.0, score=0.9, class_id=0)
    assert d.center == (5.0, 5.0)
