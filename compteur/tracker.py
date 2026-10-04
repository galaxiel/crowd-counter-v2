"""Tracking : détections frame par frame -> identités suivies (association par IoU).

Une détection est un objet sans mémoire : la même personne vue en deux frames
produit deux boîtes qui n'ont aucun lien. C'est le tracker qui crée ce lien, en
assignant à chaque personne un identifiant stable tant qu'elle reste visible.
Sans lui, le compteur compte des apparitions, pas des personnes.

Pourquoi l'IoU et pas ByteTrack
------------------------------

ByteTrack associe deux boîtes par la **distance entre leurs centres**, après
prédiction par un filtre de Kalman. Sur la vidéo de référence (caméra fixe, plan
large), une tête mesure 29 px et se déplace de **0,69 px par frame** : à cette
échelle, deux personnes distinctes sont aussi proches l'une de l'autre que
chacune l'est de sa propre position à la frame d'avant. Les seuils de Kalman,
calibrés pour des cibles qui bougent de plusieurs dizaines de pixels par frame,
ne discriminent plus rien — mesuré : 3388 tracks créées pour 589 marches réelles,
et un compteur qui rend **zéro**.

L'IoU ne dépend pas de l'échelle du déplacement : deux têtes décalées de 0,69 px
se recouvrent encore à ~94 %, et deux personnes réellement distinctes à 30 px
l'une de l'autre se recouvrent à 25 %, sous le seuil. C'est le seul critère qui
tient à cette échelle ; il rend 101 personnes là où ByteTrack en rendait 0.

Correspondance entre la configuration et ce backend :

| Config                | Effet ici                              | Sens                               |
|-----------------------|----------------------------------------|------------------------------------|
| `frames_confirmation` | filtre `nb_hits >= frames_confirmation` | frames avant apparition en sortie  |
| `survie_max`          | purge `absent > survie_max`             | frames gardées en tampon           |
| `seuil_matching`      | IoU minimal pour associer               | recouvrement exigé                 |

`seuil_confiance` n'est plus lu ici : c'était le seuil d'ACTIVATION de ByteTrack,
c'est-à-dire un seuil de score de détection, pas un critère d'association. Il
appartient au détecteur, qui l'applique déjà.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .config import Config
from .types import Detection, Track

log = logging.getLogger(__name__)


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


@dataclass
class _TrackInterne:
    """Ce que le backend retient d'une personne entre deux frames.

    ``bbox`` est la dernière boîte **vue**, jamais une position prédite : c'est
    l'absence de prédiction qui rend l'appariement stable à cette échelle.
    ``absent`` compte les frames consécutives sans détection ; ``nb_hits`` compte
    les frames au cours desquelles la personne a été vue, et n'est jamais remis à
    zéro — une personne confirmée depuis 50 frames ne doit pas redevenir
    candidate à la confirmation après une seule frame d'occultation.
    """

    bbox: tuple[float, float, float, float]
    nb_hits: int = 1
    age: int = 0
    absent: int = 0


def _boite(detection: Detection) -> tuple[float, float, float, float]:
    return (detection.x1, detection.y1, detection.x2, detection.y2)


def _matrice_iou(boites_tracks: np.ndarray, boites_dets: np.ndarray) -> np.ndarray:
    """Recouvrement de surface entre chaque track et chaque détection.

    Matrice ``(nb_tracks, nb_détections)`` de valeurs dans ``[0, 1]``. Une union
    nulle — boîte de surface nulle — vaut 0 et non pas une division par zéro :
    une boîte dégénérée ne doit jamais « correspondre » à quelqu'un.
    """
    x1 = np.maximum(boites_tracks[:, None, 0], boites_dets[None, :, 0])
    y1 = np.maximum(boites_tracks[:, None, 1], boites_dets[None, :, 1])
    x2 = np.minimum(boites_tracks[:, None, 2], boites_dets[None, :, 2])
    y2 = np.minimum(boites_tracks[:, None, 3], boites_dets[None, :, 3])
    intersection = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)

    largeur_tracks = boites_tracks[:, 2] - boites_tracks[:, 0]
    hauteur_tracks = boites_tracks[:, 3] - boites_tracks[:, 1]
    largeur_dets = boites_dets[:, 2] - boites_dets[:, 0]
    hauteur_dets = boites_dets[:, 3] - boites_dets[:, 1]
    # Les deux termes sont broadcastés en lignes/colonnes : l'aire des tracks
    # sur l'axe des tracks, celle des détections sur l'axe des détections.
    union = (
        (largeur_tracks * hauteur_tracks)[:, None]
        + (largeur_dets * hauteur_dets)[None, :]
        - intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


class _BackendIoU:
    """Association par recouvrement de surface, sans prédiction de mouvement.

    L'appariement est glouton sur l'IoU décroissante : la paire la plus
    recouvrante d'abord, puis la détection et la track sont retirées de la
    course. Sur quelques dizaines de détections, l'écart avec un appariement
    optimal (Hunger) est négligeable ; le glouton a en plus l'avantage d'être
    déterministe d'une frame à l'autre, le tri stable départageant les ex æquo
    dans l'ordre des indices.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # Les identifiants croissent et ne sont jamais recyclés : un numéro
        # resservi à une autre personne fausserait le journal anti-recomptage du
        # compteur, qui tient un registre des identifiants déjà comptés.
        self._suivant = 1
        self._vives: dict[int, _TrackInterne] = {}

    def _apparier(
        self, detections: list[Detection]
    ) -> tuple[dict[int, int], set[int]]:
        """Associer tracks et détections par IoU décroissante.

        Retourne ``(track_id -> index de détection associée, indices de
        détection déjà pris)``.
        """
        track_ids = list(self._vives)
        if not track_ids or not detections:
            return {}, set()

        boites_tracks = np.array([self._vives[t].bbox for t in track_ids], dtype=float)
        boites_dets = np.array([_boite(d) for d in detections], dtype=float)
        iou = _matrice_iou(boites_tracks, boites_dets)

        apparies: dict[int, int] = {}
        pris: set[int] = set()
        # `argsort` sur la matrice aplatie : les paires sont parcourues par
        # recouvrement décroissant. `kind="stable"` rend le départage des ex æquo
        # déterministe d'une frame à l'autre.
        for position in np.argsort(-iou, axis=None, kind="stable"):
            if iou.flat[position] < self.config.seuil_matching:
                break  # trié : plus aucune paire ne peut passer le seuil
            ligne, colonne = divmod(int(position), iou.shape[1])
            track_id = track_ids[ligne]
            if track_id in apparies or colonne in pris:
                continue
            apparies[track_id] = colonne
            pris.add(colonne)
        return apparies, pris

    def mettre_a_jour(self, detections: list[Detection]) -> list[TrackSuivi]:
        apparies, pris = self._apparier(detections)

        for track_id, vivante in self._vives.items():
            vivante.age += 1
            index = apparies.get(track_id)
            if index is None:
                # Non associée : la personne est OCCULTÉE, pas disparue. La track
                # reste vivante et conserve sa dernière position connue.
                vivante.absent += 1
                continue
            vivante.bbox = _boite(detections[index])
            vivante.nb_hits += 1
            vivante.absent = 0

        for index, detection in enumerate(detections):
            if index in pris:
                continue
            self._vives[self._suivant] = _TrackInterne(bbox=_boite(detection))
            self._suivant += 1

        for track_id in [
            k for k, v in self._vives.items() if v.absent > self.config.survie_max
        ]:
            del self._vives[track_id]

        # Une track pas encore confirmée n'est pas renvoyée : c'est le filtre de
        # l'enveloppe qui applique `frames_confirmation`, mais il ne peut le
        # faire que si la track lui arrive. Exposer les tracks naissantes
        # changerait la forme de sortie que le compteur et ses tests attendent.
        sorties: list[TrackSuivi] = []
        for track_id, vivante in sorted(self._vives.items()):
            if vivante.nb_hits < self.config.frames_confirmation:
                continue
            x1, y1, x2, y2 = vivante.bbox
            sorties.append(
                TrackSuivi(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    age=vivante.age,
                    nb_hits=vivante.nb_hits,
                )
            )
        return sorties

    def reinitialiser(self) -> None:
        self._suivant = 1
        self._vives.clear()


# Nom historique conservé : les tests existants importent ce symbole, et le test
# d'égalité de forme doit continuer à tourner contre le VRAI backend. Rien dans
# cette classe n'a plus de rapport avec supervision — à supprimer quand le test
# sera renommé, pas avant.
_BackendSupervision = _BackendIoU


class Tracker:
    """Enveloppe du backend : applique les règles de confirmation de la config.

    ``backend`` est injectable pour les tests et doit exposer la même forme
    que ``_BackendIoU`` : ``mettre_a_jour(list[Detection]) -> list[TrackSuivi]``
    et ``reinitialiser()``.
    """

    def __init__(self, config: Config, backend=None) -> None:
        _valider(config)
        self.config = config
        self.backend = backend if backend is not None else _BackendIoU(config)
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
            # La confirmation est mémorisée et non recalculée à chaque frame : un
            # track réactivé après une occultation peut se voir remettre son
            # compteur de hits à zéro par son backend, et une personne confirmée
            # depuis 50 frames ne doit pas disparaître de la sortie pour une
            # simple frame d'occultation.
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

        Un track pas encore confirmé n'est pas encore dans la sortie du backend,
        donc pas compté. Après ``reinitialiser()``, le backend repart de zéro.
        """
        return len(self._vus)