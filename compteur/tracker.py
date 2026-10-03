"""Tracking : détections frame par frame -> identités suivies (ByteTrack).

Une détection est un objet sans mémoire : la même personne vue en deux frames
produit deux boîtes qui n'ont aucun lien. C'est le tracker qui crée ce lien, en
assignant à chaque personne un identifiant stable tant qu'elle reste visible.
Sans lui, le compteur compte des apparitions, pas des personnes.

Le travail de suivi est délégué à `supervision` (ByteTrack). Ce module n'ajoute
que deux choses : le contrat en français et en pixels source, et la règle de
confirmation de la configuration.

Correspondance entre la configuration et supervision — elle n'est pas évidente,
et se tromper ici ne lève aucune erreur, ça ne produit simplement aucune track :

| Config                 | Supervision                     | Sens                              |
|------------------------|---------------------------------|-----------------------------------|
| `frames_confirmation`  | `minimum_consecutive_frames`    | frames avant activation           |
| `survie_max`           | `lost_track_buffer`             | frames gardées en mémoire tampon  |
| `seuil_matching`       | `minimum_matching_threshold`    | IoU minimal pour réassocier       |
| `seuil_confiance`      | `track_activation_threshold`    | seuil haut/bas de ByteTrack       |

`track_activation_threshold` est un seuil de **confiance**, pas un nombre de
frames : y mettre `frames_confirmation` reviendrait à exiger des scores >= 3,
et donc à ne jamais suivre personne. La confirmation en frames, c'est
`minimum_consecutive_frames`.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from .config import Config
from .types import Detection, Track

log = logging.getLogger(__name__)

# supervision convertit `lost_track_buffer` en frames via
# int(frame_rate / 30 * buffer). On fige 30 pour que `survie_max` reste bien un
# nombre de frames, quelle que soit la cadence annoncée ailleurs.
IPS_REFERENCE = 30.0

# supervision marque l'absence d'identifiant par un entier négatif sur les
# tracks pas encore activés (NO_ID vaut -1). On filtre sur le signe plutôt que
# sur la constante : un changement de sentinelle resterait sans effet ici.
_PAS_ENCORE_IDENTIFIE = 0


class TrackSuivi(NamedTuple):
    """Sortie brute du backend de tracking, avant filtrage."""

    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    age: int
    nb_hits: int


def _valider(config: Config) -> None:
    if config.frames_confirmation < 1:
        raise ValueError(
            f"frames_confirmation doit valoir au moins 1, reçu {config.frames_confirmation}"
        )
    if config.survie_max < 1:
        raise ValueError(f"survie_max doit valoir au moins 1, reçu {config.survie_max}")
    if not 0.0 < config.seuil_matching <= 1.0:
        raise ValueError(
            f"seuil_matching doit être dans ]0, 1], reçu {config.seuil_matching}"
        )


class _BackendSupervision:
    """ByteTrack via la bibliothèque supervision (testée, sans code maison)."""

    def __init__(self, config: Config) -> None:
        import supervision as sv

        self.config = config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.seuil_confiance,
            lost_track_buffer=config.survie_max,
            minimum_matching_threshold=config.seuil_matching,
            frame_rate=IPS_REFERENCE,
            minimum_consecutive_frames=config.frames_confirmation,
        )

    def _vers_track_suivi(self, track) -> TrackSuivi | None:
        identifiant = track.external_track_id
        if identifiant is None or identifiant < _PAS_ENCORE_IDENTIFIE:
            # Track pas encore activé : supervision ne lui a pas d'identifiant.
            return None
        x1, y1, x2, y2 = (float(v) for v in track.tlbr)
        return TrackSuivi(
            track_id=int(identifiant),
            bbox=(x1, y1, x2, y2),
            center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            # supervision ne stocke pas d'âge : il se déduit des frames.
            age=int(track.frame_id - track.start_frame),
            nb_hits=int(track.tracklet_len),
        )

    def mettre_a_jour(self, detections: list[Detection]) -> list[TrackSuivi]:
        import numpy as np
        import supervision as sv

        # Une frame sans détection doit quand même être transmise : c'est elle
        # qui fait vieillir les tracks dans leur tampon de survie. Renvoyer []
        # sans appeler le backend perdrait les personnes occultées.
        if detections:
            entree = sv.Detections(
                xyxy=np.array(
                    [[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=float
                ),
                confidence=np.array([d.score for d in detections], dtype=float),
                class_id=np.array([d.class_id for d in detections], dtype=int),
            )
        else:
            entree = sv.Detections.empty()
        self.tracker.update_with_detections(entree)

        sortie: list[TrackSuivi] = []
        # Les tracks « lost » sont occultées mais vivantes : les exclure ferait
        # perdre l'identité d'une personne dès qu'elle passe une seconde
        # derrière quelqu'un d'autre.
        for track in (*self.tracker.tracked_tracks, *self.tracker.lost_tracks):
            converti = self._vers_track_suivi(track)
            if converti is not None:
                sortie.append(converti)
        return sortie

    def reinitialiser(self) -> None:
        self.tracker.reset()


class Tracker:
    """Enveloppe du backend : applique les règles de confirmation de la config.

    ``backend`` est injectable pour les tests et doit exposer la même forme
    que ``_BackendSupervision`` : ``mettre_a_jour(list[Detection]) ->
    list[TrackSuivi]`` et ``reinitialiser()``.
    """

    def __init__(self, config: Config, backend=None) -> None:
        _valider(config)
        self.config = config
        self.backend = backend if backend is not None else _BackendSupervision(config)
        self._confirmes: set[int] = set()
        self._vus: set[int] = set()

    def mettre_a_jour(
        self, detections: list[Detection], frame_index: int
    ) -> list[Track]:
        bruts = self.backend.mettre_a_jour(detections)
        self._vus.update(t.track_id for t in bruts)
        log.debug("frame %d : %d track(s) suivis", frame_index, len(bruts))

        seuil = self.config.frames_confirmation
        tracks: list[Track] = []
        for t in bruts:
            # La confirmation est mémorisée et non recalculée à chaque frame :
            # ByteTrack remet son compteur de hits à zéro quand il réactive un
            # track perdu, et une personne confirmée depuis 50 frames ne doit
            # pas disparaître de la sortie parce qu'elle a été occultée 1 frame.
            if t.nb_hits >= seuil:
                self._confirmes.add(t.track_id)
            if t.track_id not in self._confirmes:
                continue
            tracks.append(
                Track(
                    track_id=t.track_id,
                    center=t.center,
                    bbox=t.bbox,
                    age=t.age,
                    confirmed=True,
                )
            )
        return tracks

    def reinitialiser(self) -> None:
        self.backend.reinitialiser()
        self._confirmes.clear()
        self._vus.clear()

    def nb_tracks_vus(self) -> int:
        """Identifiants distincts attribués depuis le début.

        Un track pas encore activé n'a pas d'identifiant : il n'est donc pas
        compté. Après ``reinitialiser()``, le backend repart de zéro.
        """
        return len(self._vus)
