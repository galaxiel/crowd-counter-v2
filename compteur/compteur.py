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

from .config import BANDE_DETECTION_PX, Config
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
        # Indicateurs de fiabilité, mesurés frame par frame :
        # - `_vies_track` : track_id -> nombre de frames où il a été vu.
        #   C'est la seule source honnête de la durée de vie d'un track —
        #   un événement de franchissement ne la donne pas.
        # - `_nb_tracks_vus`, `_nb_detections` : compteurs cumulés.
        self._vies_track: dict[int, int] = {}
        self._nb_tracks_vus = 0
        self._nb_detections = 0
        self._vus_precedents: set[int] = set()
        self._somme_presents = 0
        self._nb_frames_vues = 0
        # Identifiants dont le côté est devenu celui d'ARRIVÉE sur la frame en
        # cours. Ils sont purgés APRÈS la boucle de comptage — voir
        # `_purger_franchis`. C'est un ensemble de la frame courante, pas un
        # état cumulé : sans ce « par frame », un identifiant déjà là
        # continuerait de porter une entrée morte en mémoire pour rien.
        self._franchis: set[int] = set()

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
        self._vies_track.clear()
        self._nb_tracks_vus = 0
        self._nb_detections = 0
        self._vus_precedents.clear()
        self._somme_presents = 0
        self._nb_frames_vues = 0
        self._franchis.clear()
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

    def _purger_franchis(self) -> None:
        """Oublie les personnes qui ont franchi la ligne.

        **Une personne qui a franchi la ligne ne peut plus jamais générer de
        comptage.** Le test de franchissement exige un côté de départ
        *strictement* positif (`a_traverse` : ``c_avant > 0 >= c_apres``), donc
        une fois de l'autre côté, plus rien ne peut la faire compter à nouveau.
        Ses entrées de suivi sont donc mortes.

        Ce que ça achète, ce n'est PAS du temps de calcul sur la DÉTECTION —
        mesuré, la bande ne change rien au temps (71 à 76 s). C'est de la
        propreté d'état : les registres ci-dessous ne servent plus à rien pour
        ces tracks. Le tracker, lui, garde sa propre piste : le vider serait
        une optimisation DE PERFORMANCE que la mesure ne justifie pas, et dont
        le risque (une réapparition du même identifiant comptée comme une
        nouvelle personne) ne vaut pas le gain.

        **L'ordre est important, et c'est vérifié.** Le verrou anti-rebond lit
        `_cotes` et `_stabilite` — voir le bloc de `traiter_frame`. Cette purge
        est donc appelée APRÈS la boucle de comptage, jamais pendant : un track
        qui vient d'être compté a déjà été CEMÉ (`self._deja_comptes`) avant
        d'être purgé, donc même si la boucle le relisait, il serait ignoré au
        test anti-recomptage. Et `_deja_comptes` n'est volontairement PAS purgé
        : c'est le journal anti-recomptage, pas un état par track.

        Le résultat mesuré est la preuve que c'est gratuit : **256 personnes
        avec et sans cette purge**, sur les 3000 frames de référence.
        """
        for identifiant in self._franchis:
            self._derniers_centers.pop(identifiant, None)
            self._historiques.pop(identifiant, None)
            self._cotes.pop(identifiant, None)
            self._stabilite.pop(identifiant, None)

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

    def _bande_pour(self, img: np.ndarray) -> tuple[int, int, int, int] | None:
        """Rectangle de détection à appliquer à cette frame, ou `None`.

        **C'est ici que le `Compteur` assume la responsabilité de la bande.**
        `Detecteur` ne connaît pas la ligne — il ne fait qu'appliquer un
        rectangle qu'on lui donne. Cette frontière est ce qui permet de
        tester `detecteur.py` seul, et de remplacer le modèle sans toucher au
        comptage.

        Le rectangle est demandé à la LIGNE, pas recalculé ici : la bande est
        un attribut de la ligne, pas du compteur. C'est ce qui garantit qu'elle
        suit la ligne quand elle est déplacée, sans code de mise à jour — il n'y
        a rien à synchroniser parce qu'il n'y a pas deux copies.

        La taille de l'image est relue à CHAQUE frame plutôt que mise en cache :
        une source peut changer de résolution en cours de flux, et une bande
        calculée sur 1280 de large appliquée à une frame de 1920 rognerait au
        mauvais endroit — un décompte faux, sans rien à l'écran pour le signaler.
        """
        if self.ligne is None:
            return None
        hauteur, largeur = img.shape[:2]
        return self.ligne.rect_bande_detection(largeur, hauteur)

    def traiter_frame(
        self, img: np.ndarray, frame_index: int, timestamp_s: float
    ) -> FrameResult:
        detections = self.detecteur.detecter(img, self._bande_pour(img))
        tracks = self.tracker.mettre_a_jour(detections, frame_index)
        nouveaux: list[Evenement] = []
        ligne = self.ligne

        # Indicateurs de fiabilité, cumulés frame par frame.
        self._nb_detections += len(detections)
        vus: set[int] = set()
        for t in tracks:
            vus.add(t.track_id)
            self._vies_track[t.track_id] = self._vies_track.get(t.track_id, 0) + 1
        self._nb_tracks_vus += len(vus - self._vus_precedents)
        self._vus_precedents = vus

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

            # Le côté est calculé UNE fois par track et sert aux deux usages :
            # le marquage d'arrivée ci-dessous et, plus bas, l'écriture de
            # `_cotes`. Les dupliquer coûterait un calcul de projection par
            # track par frame pour rien.
            cote = ligne.point_du_cote(t.center) if ligne is not None else 0

            # Marquage des tracks arrivés, AVANT tout `continue`. Il doit l'être
            # pour tous, y compris ceux déjà comptés : ce sont eux les plus
            # nombreux, et c'est précisément eux dont l'état est mort. Placé
            # après le `continue`, une personne déjà comptée garderait ses
            # entrées pour rien — c'est le cas majoritaire.
            if cote < 0:
                self._franchis.add(identifiant)

            if identifiant in self._deja_comptes or ligne is None:
                continue

            # État tel qu'il était AVANT cette frame : c'est lui qui décide du
            # franchissement, pas la position qu'on vient d'enregistrer.
            cote_avant = self._cotes.get(identifiant)
            stabilite = self._stabilite.get(identifiant, 0)

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
        # Purge des tracks ayant franchi, APRÈS la boucle de comptage : le
        # verrou anti-rebond lit `_cotes` et `_stabilite`, les effacer pendant
        # la boucle lui retirerait la moitié de son information de décision.
        # L'ordre est donc déterminant, pas cosmétique — voir `_purger_franchis`.
        self._purger_franchis()
        # L'ensemble est consommé : le vider ici empêche un identifiant de
        # rester marqué sur les frames suivantes, où il serait de nouveau
        # légitime de le purger (il peut revenir du côté du départ).
        self._franchis.clear()

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

    def resultat(
        self,
        modele: str,
        nb_frames: int,
        secondes: float,
        duree_video_s: float = 0.0,
    ) -> Resultat:
        """Bilan de l'analyse.

        `secondes` est le temps de CALCUL écoulé ; `duree_video_s` la durée de
        la vidéo. Ce ne sont pas la même chose (le GPU va plus vite que le
        temps réel) et les confondre fausse le débit par minute.

        Les indicateurs de fiabilité sont mesurés ICI, pas reconstruits : on
        connaît la durée de vie réelle de chaque track parce qu'on l'a vue
        frame par frame. Les laisser à `None` dans l'export, ce que le
        premier jet faisait, revient à masquer la seule information qui permet
        à un utilisateur de juger son propre décompte.
        """
        presents_moyen = (
            self._somme_presents / self._nb_frames_vues if self._nb_frames_vues else 0.0
        )
        vies = list(self._vies_track.values())
        duree_vie_moy = sum(vies) / len(vies) if vies else 0.0
        return Resultat(
            total=self.total,
            evenements=list(self.evenements),
            config=self.config,
            modele=modele,
            nb_frames=nb_frames,
            presents_max=self.presents_max,
            presents_moyen=presents_moyen,
            secondes=secondes,
            duree_video_s=duree_video_s,
            duree_vie_track_moy=duree_vie_moy,
            nb_tracks_vus=self._nb_tracks_vus,
            detections_par_frame=(
                self._nb_detections / self._nb_frames_vues if self._nb_frames_vues else 0.0
            ),
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
        # `peripherique` ne fait qu'indiquer OÙ charger le modèle : la
        # logique de comptage plus bas est strictement inchangée.
        detecteur = Detecteur(
            charger_modele(config.modele, config.peripherique), config
        )
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
                # La bande est un attribut de la ligne, pas un réglage à part :
                # elle est donc passée ici au même titre que l'épaisseur. C'est
                # ce qui fait qu'une analyse hors interface rogne exactement
                # comme celle de la fenêtre.
                bande_detection_px=BANDE_DETECTION_PX,
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

    return compteur.resultat(
        config.modele,
        index,
        time.time() - debut,
        duree_video_s=index / fps if fps > 0 else 0.0,
    )
