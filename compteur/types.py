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

    def vers_ligne_csv(self) -> str:
        return f"{self.frame},{self.timestamp_s:.3f},{self.x:.1f},{self.y:.1f},{self.track_id}"


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
    secondes: float = 0.0
