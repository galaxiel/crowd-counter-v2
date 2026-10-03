"""Orchestrateur : video + config -> résultat.

Ce module ne connaît pas l'interface. Il est appelé par l'IHM, par les tests,
et demain par une éventuelle API web.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable

import numpy as np

from .config import Config
from .detecteur import Detecteur, charger_modele
from .ligne import Ligne
from .tracker import Tracker
from .types import Evenement, FrameResult, Resultat

log = logging.getLogger(__name__)


class Compteur:
    """Machine à états du comptage : une frame à la fois.

    Une traversée est un ÉVÉNEMENT, pas un ÉTAT. L'état à retenir est donc, par
    track, « de quel côté cette personne était-elle au dernier frame où elle
    était loin de la ligne ? », et non « depuis combien de frames
    ``a_traverse`` est-il vrai ? ». Voir le bloc de ``traiter_frame`` et la
    section « Algorithme de comptage » du plan.
    """

    def __init__(self, config: Config, detecteur, tracker) -> None:
        if config.fenetre_lissage < 1:
            raise ValueError(
                f"fenetre_lissage doit valoir au moins 1, "
                f"reçu {config.fenetre_lissage}"
            )
        self.config = config
        self.detecteur = detecteur
        self.tracker = tracker
        self.ligne: Ligne | None = None
        self.total = 0
        self.presents = 0
        self.presents_max = 0
        # Liste mutée en place, jamais réassignée : l'IHM (tâche 11) en garde
        # une référence pour son panneau d'audit.
        self.evenements: list[Evenement] = []
        # Position utilisée pour le test de franchissement : moyenne des K
        # dernières positions connues du track (K = `fenetre_lissage`).
        # K=1 y dépose la position instantanée — c'est le mode « brut ».
        self._derniers_centers: dict[int, tuple[float, float]] = {}
        # Historique brut par track, borné à K positions (deque à maxlen).
        # Nécessaire en plus de `_derniers_centers` : celui-ci ne retient que
        # la position lissée, pas les K échantillons qui l'ont produite.
        self._historiques: dict[int, deque[tuple[float, float]]] = {}
        # Dernier CÔTÉ stable observé par track (+1, -1 ; jamais 0).
        self._cotes: dict[int, int] = {}
        # Frames consécutives passées du côté de DÉPART, par track.
        self._stabilite: dict[int, int] = {}
        # Registre des identifiants déjà comptés. Volontairement NON purgé :
        # c'est un journal anti-recomptage, pas un état par track. ByteTrack
        # n'est jamais réinitialisé en cours de flux, donc purger ce registre
        # ne ferait que permettre un double comptage après une réapparition.
        self._deja_comptes: set[int] = set()
        self._somme_presents = 0
        self._nb_frames_vues = 0

    def ajuster_ligne(self, ligne: Ligne) -> None:
        self.ligne = ligne

    def reinitialiser(self) -> None:
        self.total = 0
        self.presents = 0
        self.presents_max = 0
        self.evenements.clear()
        self._derniers_centers.clear()
        self._historiques.clear()
        self._cotes.clear()
        self._stabilite.clear()
        self._deja_comptes.clear()
        self._somme_presents = 0
        self._nb_frames_vues = 0
        self.tracker.reinitialiser()

    def _purger_absents(self, vus: set[int]) -> None:
        """Oublie les tracks que le tracker ne renvoie plus.

        ``set(registre) - vus`` : soustraire un ``set`` d'une ``list`` lève un
        ``TypeError``, d'où la conversion explicite. Les registres sont des
        dict, on supprime donc par clé — ``dict`` n'a pas de ``discard``.
        """
        for registre in (
            self._derniers_centers,
            self._historiques,
            self._cotes,
            self._stabilite,
        ):
            for identifiant in set(registre) - vus:
                del registre[identifiant]

    def _position_lissee(
        self, identifiant: int, center: tuple[float, float]
    ) -> tuple[float, float]:
        """Moyenne des K dernières positions connues de ce track.

        Le bruit du détecteur est aléatoire d'une frame à l'autre : il
        s'annule par moyennage. Le mouvement réel, lui, est constant d'une
        frame à l'autre : il se cumule. C'est ce qui fait émerger le
        déplacement réel d'un signal où le déplacement médian mesuré entre
        deux frames est de 0,00 px.

        ``fenetre_lissage = 1`` rend la position instantanée EXACTEMENT :
        c'est le mode « brut », celui d'avant ce lissage, conservé comme
        mode de comparaison.
        """
        historique = self._historiques.get(identifiant)
        if historique is None:
            # `maxlen` borne la mémoire par track à K positions, sans purge
            # manuelle : au-delà de K frames, la plus ancienne tombe.
            historique = deque(maxlen=self.config.fenetre_lissage)
            self._historiques[identifiant] = historique
        historique.append(center)
        if len(historique) == 1:
            # Une seule position connue : la moyenne ne peut être qu'elle-même.
            return center
        n = len(historique)
        return (
            sum(p[0] for p in historique) / n,
            sum(p[1] for p in historique) / n,
        )

    def traiter_frame(
        self, img: np.ndarray, frame_index: int, timestamp_s: float
    ) -> FrameResult:
        detections = self.detecteur.detecter(img)
        tracks = self.tracker.mettre_a_jour(detections, frame_index)
        nouveaux: list[Evenement] = []
        ligne = self.ligne

        for t in tracks:
            identifiant = t.track_id
            # Position LISSEE : moyenne des K dernières positions de ce track.
            # C'est elle qui décide du franchissement — le bruit du détecteur
            # y est moyené, le mouvement réel y est cumulé. Le côté et la
            # stabilité, eux, restent lus sur la position instantanée : le
            # verrou anti-rebond est inchangé, seule l'entrée du test
            # `a_traverse` diffère.
            position = self._position_lissee(identifiant, t.center)
            precedent = self._derniers_centers.get(identifiant)
            self._derniers_centers[identifiant] = position

            if identifiant in self._deja_comptes or ligne is None:
                continue

            # État tel qu'il était AVANT cette frame : c'est lui qui décide du
            # franchissement, pas la position qu'on vient d'enregistrer.
            cote_avant = self._cotes.get(identifiant)
            stabilite = self._stabilite.get(identifiant, 0)

            cote = ligne.point_du_cote(t.center)
            if cote != 0:
                # Le côté n'est mémorisé que HORS de la bande : à ±1 px de la
                # ligne, `point_du_cote` renvoie 0, une personne qui oscille
                # n'écrit donc jamais d'état et ne peut rien déclencher.
                # Cette mise à jour a lieu dès la PREMIÈRE frame du track :
                # sinon le côté de départ d'une personne qui n'apparaît qu'une
                # fois de ce côté resterait inconnu, et le délai
                # `frames_hysteresis` serait amputé d'une frame.
                self._cotes[identifiant] = cote
                if cote > 0:
                    self._stabilite[identifiant] = self._stabilite.get(identifiant, 0) + 1
                else:
                    self._stabilite[identifiant] = 0

            if precedent is None or not ligne.a_traverse(precedent, position):
                continue

            # Verrou anti-rebond : la personne venait-elle du bon côté, et
            # y était-elle restée assez longtemps ? Cette formulation fonctionne
            # alors que `a_traverse` n'est vrai que sur UNE frame par
            # franchissement, là où un compteur de frames consécutives plafonne
            # à 1 et rend tout seuil > 1 inatteignable.
            if cote_avant is None or cote_avant <= 0:
                continue
            if stabilite < ligne.hysteresis:
                continue

            self._deja_comptes.add(identifiant)
            self.total += 1
            ev = Evenement(
                frame=frame_index,
                timestamp_s=timestamp_s,
                x=t.center[0],
                y=t.center[1],
                track_id=identifiant,
            )
            self.evenements.append(ev)
            nouveaux.append(ev)

        # Purge des tracks disparus pour éviter les fuites mémoire.
        self._purger_absents({t.track_id for t in tracks})

        self.presents = len(tracks)
        self.presents_max = max(self.presents_max, self.presents)
        self._somme_presents += self.presents
        self._nb_frames_vues += 1

        return FrameResult(
            image=img,
            detections=detections,
            tracks=tracks,
            total=self.total,
            presents=self.presents,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            evenements=nouveaux,
        )

    def resultat(self, modele: str, nb_frames: int, secondes: float) -> Resultat:
        presents_moyen = (
            self._somme_presents / self._nb_frames_vues if self._nb_frames_vues else 0.0
        )
        return Resultat(
            total=self.total,
            evenements=list(self.evenements),
            config=self.config,
            modele=modele,
            nb_frames=nb_frames,
            presents_max=self.presents_max,
            presents_moyen=presents_moyen,
            secondes=secondes,
        )


def analyser_video(
    chemin_video: str,
    config: Config,
    callback_frame: Callable[[FrameResult], None] | None = None,
    detecteur=None,
    tracker=None,
) -> Resultat:
    """Analyse un fichier vidéo de bout en bout.

    Si ``callback_frame`` est fourni, il est appelé à chaque frame — c'est ce
    mode qu'utilise l'interface pour l'affichage en direct.
    """
    import cv2

    if detecteur is None:
        detecteur = Detecteur(charger_modele(config.modele), config)
    if tracker is None:
        tracker = Tracker(config)

    compteur = Compteur(config, detecteur, tracker)
    if config.ligne is not None:
        compteur.ajuster_ligne(
            Ligne(
                (config.ligne[0], config.ligne[1]),
                (config.ligne[2], config.ligne[3]),
                epaisseur=config.epaisseur_bande,
                sens=config.sens,
                hysteresis=config.frames_hysteresis,
            )
        )

    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"vidéo illisible : {chemin_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    debut = time.time()
    index = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            resultat_frame = compteur.traiter_frame(img, index, index / fps)
            if callback_frame is not None:
                callback_frame(resultat_frame)
            index += 1
            if total_frames and index % 100 == 0:
                log.info("%d/%d frames", index, total_frames)
    finally:
        cap.release()

    return compteur.resultat(config.modele, index, time.time() - debut)
