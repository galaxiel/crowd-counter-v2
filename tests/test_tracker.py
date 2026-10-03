"""Tests du tracking : de la détection isolée à l'identité stable dans le temps.

Les tests de comportement tournent contre le VRAI backend supervision, pas
contre un faux : c'est le seul moyen que « la personne de la frame 12 est la
même qu'en frame 13 » soit une propriété vérifiée et non supposée. Le faux
backend sert au test d'injection, et un test d'égalité de forme vérifie qu'il
parle bien la même langue que le vrai.
"""

from __future__ import annotations

import pytest

from compteur.config import Config
from compteur.tracker import TrackSuivi, Tracker, _BackendSupervision
from compteur.types import Detection


def d(x, y, w=50, h=50, score=0.9, class_id=0):
    return Detection(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h),
                     score=score, class_id=class_id)


def par_id(tracks, track_id):
    """Le track d'un certain identifiant, ou None s'il n'est pas présent."""
    for t in tracks:
        if t.track_id == track_id:
            return t
    return None


# --------------------------------------------------------------------------
# Confirmation : une track n'existe pour le compteur qu'après N frames
# --------------------------------------------------------------------------


def test_detection_isolee_ne_produit_rien_tant_que_non_confirmee():
    t = Tracker(Config(frames_confirmation=3))
    assert t.mettre_a_jour([d(10, 10)], 0) == []
    assert t.mettre_a_jour([d(11, 10)], 1) == []


def test_track_confirme_apres_le_nombre_de_frames_demande():
    t = Tracker(Config(frames_confirmation=3))
    assert t.mettre_a_jour([d(10, 10)], 0) == []
    assert t.mettre_a_jour([d(11, 10)], 1) == []
    tracks = t.mettre_a_jour([d(12, 10)], 2)
    assert len(tracks) == 1
    assert tracks[0].confirmed is True
    assert isinstance(tracks[0].track_id, int)
    assert tracks[0].track_id > 0


def test_confirmation_plus_rapide_si_demande_une_frame():
    """Non-vacuité : le seuil agit vraiment, ce n'est pas un « dès la 1re frame »."""
    lent = Tracker(Config(frames_confirmation=5))
    assert lent.mettre_a_jour([d(10, 10)], 0) == []
    rapide = Tracker(Config(frames_confirmation=1))
    assert len(rapide.mettre_a_jour([d(10, 10)], 0)) == 1


# --------------------------------------------------------------------------
# Identité : c'est le cœur du tracking
# --------------------------------------------------------------------------


def test_meme_personne_conserve_son_identifiant():
    t = Tracker(Config(frames_confirmation=1))
    premier = t.mettre_a_jour([d(10, 10)], 0)[0]
    second = t.mettre_a_jour([d(12, 11)], 1)[0]
    assert premier.track_id == second.track_id
    assert t.nb_tracks_vus() == 1


def test_deux_personnes_distinctes_ont_des_ids_distincts():
    t = Tracker(Config(frames_confirmation=1))
    tracks = t.mettre_a_jour([d(10, 10), d(400, 300)], 0)
    assert len({x.track_id for x in tracks}) == 2
    # Une seule track dupliquée passerait le test précédent : on vérifie aussi
    # que les deux boîtes correspondent bien aux deux personnes.
    centres = {tuple(round(c, 1) for c in tr.center) for tr in tracks}
    assert centres == {(35.0, 35.0), (425.0, 325.0)}


def test_personne_perdue_puis_revue_recompte_comme_nouvelle():
    """Après survie_max frames sans détection, l'ID n'est pas réutilisé.

    Réécrit : le brief appelait reinitialiser() au milieu, ce qui ne testait que
    le reset et jamais la perte. Ici la track disparaît vraiment (survie_max
    frames sans détection) et une nouvelle détection, à l'autre bout de l'image,
    doit obtenir un identifiant différent.
    """
    t = Tracker(Config(frames_confirmation=1, survie_max=2))
    ancien = t.mettre_a_jour([d(10, 10)], 0)[0].track_id

    # La personne est occultée plus longtemps que survie_max : la track meurt.
    # supervision la garde d'abord dans son tampon de survie, l'identifiant ne
    # disparaît qu'une fois le tampon épuisé. On vérifie les deux temps.
    lots = [t.mettre_a_jour([], i) for i in range(1, 9)]
    assert lots[0], "la track doit survivre au moins une frame d'occultation"
    for numero, lot in enumerate(lots[3:], start=4):
        assert lot == [], f"la track aurait dû être abandonnée à la frame {numero}"

    # Quelqu'un d'autre apparaît ailleurs. Deux frames : supervision n'active un
    # tracklateur né après la première frame qu'à la frame suivante.
    neuf = None
    for i in (7, 8):
        tracks = t.mettre_a_jour([d(600, 400)], i)
        if tracks:
            neuf = tracks[0].track_id
            break
    assert neuf is not None, "la nouvelle détection n'a jamais produit de track"
    assert neuf != ancien, "l'identifiant d'une personne disparue a été réutilisé"


def test_track_conserve_son_id_pendant_une_occlusion_plus_courte_que_survie_max():
    """Une occultation brève ne doit pas couper l'identité."""
    t = Tracker(Config(frames_confirmation=1, survie_max=5))
    avant = t.mettre_a_jour([d(10, 10)], 0)[0].track_id
    perdus = [t.mettre_a_jour([], i) for i in (1, 2)]
    apres = t.mettre_a_jour([d(10, 10)], 3)[0].track_id

    assert avant == apres
    # La track survit dans le tampon : elle est toujours annoncée.
    for lot in perdus:
        assert [x.track_id for x in lot] == [avant]


def test_confirmation_ne_se_perd_pas_apres_une_reactivation():
    """Une personne occultée puis retrouvée reste confirmée.

    ByteTrack remet son compteur de hits à zéro quand il réactive un track
    perdu. Filtrer sur ce compteur ferait disparaître de la sortie une personne
    pourtant confirmée depuis 50 frames.
    """
    t = Tracker(Config(frames_confirmation=1, survie_max=5))
    avant = t.mettre_a_jour([d(10, 10)], 0)[0].track_id
    t.mettre_a_jour([], 1)  # occultation : la track passe dans le tampon
    retrouve = t.mettre_a_jour([d(11, 10)], 2)
    assert len(retrouve) == 1, "une personne retrouvée ne doit pas disparaître"
    assert retrouve[0].track_id == avant
    assert t.nb_tracks_vus() == 1, "ce n'est pas une nouvelle personne"


# --------------------------------------------------------------------------
# Géométrie : pixels source, pas des coordonnées internes au backend
# --------------------------------------------------------------------------


def test_bbox_exposee_en_pixels():
    """La bounding box est en pixels source, et non une coordonnée interne."""
    t = Tracker(Config(frames_confirmation=1))
    track = t.mettre_a_jour([d(10, 20)], 0)[0]
    assert track.bbox == pytest.approx((10.0, 20.0, 60.0, 70.0))
    assert track.center == pytest.approx((35.0, 45.0))


def test_bbox_non_carre_exposee_en_pixels():
    """Non-vacuité : largeur et hauteur ne sont pas confondues."""
    t = Tracker(Config(frames_confirmation=1))
    track = t.mettre_a_jour([d(10, 20, 60, 80)], 0)[0]
    assert track.bbox == pytest.approx((10.0, 20.0, 70.0, 100.0))
    assert track.center == pytest.approx((40.0, 60.0))


def test_center_au_barycentre_de_la_bbox():
    t = Tracker(Config(frames_confirmation=1))
    track = t.mettre_a_jour([d(100, 200, 40, 90)], 0)[0]
    x1, y1, x2, y2 = track.bbox
    assert track.center == pytest.approx(((x1 + x2) / 2, (y1 + y2) / 2))


# --------------------------------------------------------------------------
# État interne
# --------------------------------------------------------------------------


def test_reinitialiser_oublie_tout():
    t = Tracker(Config(frames_confirmation=1))
    premier = t.mettre_a_jour([d(10, 10)], 0)[0].track_id
    t.reinitialiser()
    assert t.nb_tracks_vus() == 0
    neuf = t.mettre_a_jour([d(10, 10)], 1)[0].track_id
    assert neuf == premier, "reinitialiser doit reconstruire un backend vierge"


def test_nb_tracks_vus_compte_les_identites_distinctes():
    """Le compteur d'identités, pas le nombre de frames ni de sorties."""
    t = Tracker(Config(frames_confirmation=1))
    assert t.nb_tracks_vus() == 0
    t.mettre_a_jour([d(10, 10), d(400, 300)], 0)
    assert t.nb_tracks_vus() == 2
    for i in (1, 2, 3):
        t.mettre_a_jour([d(10 + i, 10), d(400 + i, 300)], i)
    assert t.nb_tracks_vus() == 2, "les mêmes personnes ne doivent pas créer d'ID"
    # Une 3e personne apparaît : supervision n'attribue son identifiant qu'à la
    # frame suivante, on lui laisse le temps d'être activée.
    t.mettre_a_jour([d(10, 10), d(400, 300), d(700, 100)], 4)
    t.mettre_a_jour([d(11, 10), d(401, 300), d(701, 100)], 5)
    assert t.nb_tracks_vus() == 3


@pytest.mark.parametrize(
    "config",
    [
        Config(frames_confirmation=0),
        Config(survie_max=0),
        Config(seuil_matching=1.5),
    ],
    ids=["confirmation-nulle", "survie-nulle", "seuil-hors-bornes"],
)
def test_configuration_incoherence_refusee(config):
    with pytest.raises(ValueError):
        Tracker(config)


# --------------------------------------------------------------------------
# Backend injectable : c'est ce qui rend l'orchestrateur testable
# --------------------------------------------------------------------------


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aire_a = (a[2] - a[0]) * (a[3] - a[1])
    aire_b = (b[2] - b[0]) * (b[3] - b[1])
    union = aire_a + aire_b - inter
    return inter / union if union > 0 else 0.0


class FauxBackend:
    """Backend de substitution, de même forme de sortie que le vrai.

    Il reproduit ce dont l'enveloppe a réellement besoin : identifiants
    croissants jamais réutilisés, ``nb_hits`` croissant, disparition après
    ``survie_max`` frames sans détection, coordonnées en pixels source. Pas de
    filtre de Kalman, donc entièrement déterministe.

    Il reste plus simple que supervision sur un point, qu'il déclare
    explicitement : il n'a pas la latence d'activation d'une frame qu'impose
    supervision à un track né après la première frame.
    """

    def __init__(self, config):
        self.config = config
        self.reinitialisations = 0
        self._suivant = 1
        self._frame = 0
        self._vives: dict[int, dict] = {}

    def reinitialiser(self):
        self.reinitialisations += 1
        self._suivant = 1
        self._frame = 0
        self._vives = {}

    def mettre_a_jour(self, detections):
        self._frame += 1
        libres = list(range(len(detections)))
        apparies: dict[int, int] = {}
        for track_id in sorted(self._vives):
            vivante = self._vives[track_id]
            meilleur, meilleure_iou = None, self.config.seuil_matching
            for i in libres:
                score = _iou(vivante["bbox"], _boite(detections[i]))
                if score >= meilleure_iou:
                    meilleur, meilleure_iou = i, score
            if meilleur is not None:
                apparies[track_id] = meilleur
                libres.remove(meilleur)

        for i in libres:
            track_id = self._suivant
            self._suivant += 1
            self._vives[track_id] = {
                "bbox": _boite(detections[i]), "nb_hits": 1, "age": 0, "absent": 0,
            }

        for track_id, vivante in self._vives.items():
            if track_id in apparies:
                vivante["bbox"] = _boite(detections[apparies[track_id]])
                vivante["nb_hits"] += 1
                vivante["age"] += 1
                vivante["absent"] = 0
            else:
                vivante["age"] += 1
                vivante["absent"] += 1

        for track_id in [k for k, v in self._vives.items()
                         if v["absent"] > self.config.survie_max]:
            del self._vives[track_id]

        return [
            _vers_track_suivi(k, v)
            for k, v in sorted(self._vives.items())
            # supervision n'attribue d'identifiant qu'à un track activé, donc un
            # backend qui exposerait les tracks non confirmés parlerait une autre
            # forme que le vrai.
            if v["nb_hits"] >= self.config.frames_confirmation
        ]


def _boite(detection):
    return (detection.x1, detection.y1, detection.x2, detection.y2)


def _vers_track_suivi(track_id, vivante):
    x1, y1, x2, y2 = vivante["bbox"]
    return TrackSuivi(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        age=vivante["age"],
        nb_hits=vivante["nb_hits"],
    )


class BackendBavard(FauxBackend):
    """Expose aussi les tracks pas encore confirmés.

    supervision ne le fait pas (il n'attribue pas d'identifiant avant
    activation), mais le contrat de l'enveloppe doit tenir devant n'importe
    quel backend : c'est elle qui applique la règle de confirmation.
    """

    def mettre_a_jour(self, detections):
        self._frame += 1
        bruts = super().mettre_a_jour(detections)
        # On réinjecte les tracks non confirmés que la classe parente a filtrés.
        return bruts + [
            _vers_track_suivi(k, v)
            for k, v in sorted(self._vives.items())
            if v["nb_hits"] < self.config.frames_confirmation
        ]


def test_backend_bavard_ne_pas_fournir_de_track_non_confirme():
    """Non-vacuité du filtre de confirmation de l'enveloppe.

    Sans ce test, supprimer la condition `nb_hits >= frames_confirmation` ne
    casse rien : supervision filtre déjà en amont, et le test ne prouve donc
    pas que l'enveloppe applique la règle.
    """
    config = Config(frames_confirmation=3)
    t = Tracker(config, backend=BackendBavard(config))
    assert t.mettre_a_jour([d(10, 10)], 0) == []
    assert t.mettre_a_jour([d(11, 10)], 1) == []
    tracks = t.mettre_a_jour([d(12, 10)], 2)
    assert len(tracks) == 1
    assert tracks[0].confirmed is True


def test_backend_injecte_remplace_supervision():
    config = Config(frames_confirmation=2)
    faux = FauxBackend(config)
    t = Tracker(config, backend=faux)
    assert t.mettre_a_jour([d(10, 10)], 0) == []
    tracks = t.mettre_a_jour([d(11, 10)], 1)
    assert len(tracks) == 1
    assert tracks[0].center == pytest.approx((36.0, 35.0))


def test_reinitialiser_reinitialise_le_backend_injecte():
    """Le brief reconstruisait systématiquement un vrai supervision : un backend
    injecté aurait été silencieusement remplacé au premier reset."""
    config = Config(frames_confirmation=1)
    faux = FauxBackend(config)
    t = Tracker(config, backend=faux)
    t.mettre_a_jour([d(10, 10)], 0)
    t.reinitialiser()
    assert faux.reinitialisations == 1, "le backend injecté n'a pas été réinitialisé"
    assert t.nb_tracks_vus() == 0
    assert t.mettre_a_jour([d(10, 10)], 1)[0].track_id == 1


def test_faux_backend_et_backend_reel_ont_la_meme_forme():
    """Si le faux ne parle pas la même forme que supervision, ses tests ne
    prouvent rien. On compare donc les deux sur la même séquence de frames."""
    config = Config(frames_confirmation=2, survie_max=3)
    # Une personne apparaît 2 frames, disparaît plus longtemps que survie_max,
    # puis quelqu'un d'autre entre par l'autre côté de l'image.
    scenario = [
        [d(10, 10)],
        [d(11, 10)],
        [],
        [],
        [],
        [],
        [],
        [d(800, 500)],
        [d(801, 500)],
    ]
    vrai = _BackendSupervision(config)
    faux = FauxBackend(config)

    vrais, simules = [], []
    for dets in scenario:
        vrais.append(vrai.mettre_a_jour(dets))
        simules.append(faux.mettre_a_jour(dets))

    for brut in vrais + simules:
        assert isinstance(brut, list)
        for t in brut:
            assert isinstance(t, TrackSuivi)
            assert isinstance(t.track_id, int) and t.track_id > 0
            assert isinstance(t.bbox, tuple) and len(t.bbox) == 4
            assert all(isinstance(v, float) for v in t.bbox)
            assert isinstance(t.center, tuple) and len(t.center) == 2
            assert isinstance(t.age, int)
            assert isinstance(t.nb_hits, int)

    # Confirmation à la même frame des deux côtés.
    frames_confirmees = [i for i, lot in enumerate(vrais) if lot]
    frames_faux = [i for i, lot in enumerate(simules) if lot]
    assert frames_confirmees[0] == frames_faux[0]

    # Une fois confirmé, le track survit au tampon puis est abandonné. On ne
    # regarde que les frames postérieures à la confirmation : avant, une sortie
    # vide signifie « pas encore confirmé », pas « disparu ».
    for nom, lots in (("supervision", vrais), ("faux", simules)):
        debut = frames_confirmees[0] if nom == "supervision" else frames_faux[0]
        disparue = [i for i, lot in enumerate(lots) if i > debut and not lot]
        assert disparue, f"{nom} : le track n'a jamais été abandonné"
        assert disparue[0] >= debut + 1 + config.survie_max, (
            f"{nom} : track abandonné à la frame {disparue[0]}, avant la fin du "
            f"tampon de survie ({config.survie_max} frames)"
        )

    # Son identifiant n'est pas resservi à la nouvelle personne.
    ancien = vrais[1][0].track_id
    nouveau = vrais[-1][0].track_id
    assert simules[-1][0].track_id == nouveau, "le faux et le vrai divergent sur l'ID"
    assert nouveau != ancien, "l'identifiant d'une personne disparue a été réutilisé"
    assert nouveau > ancien, "les identifiants doivent croître, pas être recyclés"
