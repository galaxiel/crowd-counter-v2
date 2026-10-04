"""Structures de données échangées entre les briques du moteur."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # évite une dépendance circulaire à l'exécution
    from .config import Config


@dataclass
class Detection:
    """Boîte produite par le détecteur, en coordonnées pixels de l'image source."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aire(self) -> float:
        return self.width * self.height


@dataclass
class Track:
    """Identité suivie dans le temps, telle que-maintenue par le tracker."""

    track_id: int
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    age: int
    confirmed: bool


@dataclass
class Evenement:
    """Un comptage effectif, daté et localisé."""

    frame: int
    timestamp_s: float
    x: float
    y: float
    track_id: int
    # Boîte de la personne AU MOMENT du franchissement, en pixels image.
    #
    # Elle ne sert qu'à l'affichage post-comptage (la boîte verte). Elle est
    # donc optionnelle : un `Evenement` construit à la main — un test, un
    # script — n'a pas de boîte à peindre, et doit rester constructible sans.
    #
    # Elle est stockée et non suivis : le track est lâché au franchissement
    # (optimisation validée de la tâche 19), donc la boîte est le SEUL vestige
    # de cette personne après le compte. C'est de la mémoire d'affichage, pas
    # du tracking — voir `interface.app._boites_comptees`.
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class FrameResult:
    """Tout ce que l'interface peut afficher pour une frame donnée."""

    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    total: int = 0
    presents: int = 0
    frame_index: int = 0
    timestamp_s: float = 0.0
    evenements: list[Evenement] = field(default_factory=list)


@dataclass
class Resultat:
    """Bilan complet d'une analyse."""

    total: int
    evenements: list[Evenement] = field(default_factory=list)
    config: "Config | None" = None
    modele: str = ""
    nb_frames: int = 0
    presents_max: int = 0
    presents_moyen: float = 0.0
    # Deux durées distinctes, à ne pas confondre :
    # - `secondes` = temps de CALCUL écoulé (le GPU travaille plus vite que
    #   le temps réel de la vidéo) ;
    # - `duree_video_s` = durée de la VIDÉO analysée.
    # Le débit par minute se calcule sur `duree_video_s`. Les confondre
    # sous-estime le débit d'un facteur video/temps de calcul, soit ~2x ici.
    secondes: float = 0.0
    duree_video_s: float = 0.0
    # Indicateurs de FIABILITÉ du comptage. L'export les publie pour qu'un
    # utilisateur puisse juger lui-même si le décompte tient.
    duree_vie_track_moy: float = 0.0
    nb_tracks_vus: int = 0
    detections_par_frame: float = 0.0
