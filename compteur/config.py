"""Réglages du compteur : dataclasses, sérialisation JSON, valeurs par défaut."""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import asdict, dataclass, fields

log = logging.getLogger(__name__)

RACINE = pathlib.Path(__file__).resolve().parent.parent


def chemin_defaut_config() -> pathlib.Path:
    """Chemin du fichier de valeurs par défaut, versionné avec le projet."""
    return RACINE / "config" / "default.json"


@dataclass
class Config:
    # Détection
    # medium.pt : modèle de tête entraîné sur SCUT-HEAD (foule dense). Sur la
    # vidéo de référence il donne des têtes de 29 px contre 21 px pour
    # yolov8n-head.pt, soit 38 % de marches réelles en plus.
    modele: str = "medium.pt"
    seuil_confiance: float = 0.25
    # Une tête détectée sur cette vidéo fait 20-29 px : un seuil de 20 px
    # filtrerait la moitié des détections. 3 px ne filtre que le bruit.
    taille_min_px: int = 3
    classes_retenues: list[int] | None = None
    taille_entree: int = 640
    # Tracker
    frames_confirmation: int = 3
    survie_max: int = 30
    # Une tete de 29 px qui bouge de 15 px/frame a une IoU de 0.32 avec
    # elle-meme : a 0.5 le tracker perdrait la personne. Sur la video de
    # reference le deplacement est de 0.69 px (IoU 0.95), tres au-dessus.
    seuil_matching: float = 0.3
    # Ligne
    ligne: tuple[float, float, float, float] | None = None
    epaisseur_bande: int = 30
    sens: int = 1
    frames_hysteresis: int = 2
    # Lissage
    # Fenêtre de la moyenne mobile des positions d'un track, en frames. Le
    # mouvement réel d'une personne (~0,7 px/frame sur la vidéo de référence)
    # est noyé dans le bruit du détecteur : sans lissage, le test de
    # franchissement porte sur une position qui ne « bouge » pas d'une frame à
    # l'autre (déplacement médian mesuré : 0,00 px). La moyenne des K
    # dernières positions fait ressortir la tendance — le bruit aléatoire
    # s'annule par moyennage, le mouvement constant se cumule.
    # K=1 = mode « brut » : position instantanée, comportement inchangé.
    # K>1 a été essayé puis retiré : le lissage retarde la position d'une
    # demi-fenetre, donc au moment où le croisement est détecté la position
    # lissée est encore dans la bande, `point_du_cote` renvoie 0, le côté de
    # départ n'est jamais mémorisé et le comptage est perdu. Mesuré : K=5
    # invalide déjà le franchissement, quelle que soit la vitesse.
    fenetre_lissage: int = 1

    def vers_dict(self) -> dict:
        d = asdict(self)
        # JSON n'a pas de tuple : on repasse en liste.
        d["ligne"] = list(self.ligne) if self.ligne is not None else None
        return d

    @classmethod
    def depuis_dict(cls, d: dict) -> "Config":
        connus = {f.name for f in fields(cls)}
        inconnus = set(d) - connus
        if inconnus:
            raise ValueError(f"clés de configuration inconnues : {sorted(inconnus)}")
        d = dict(d)
        if d.get("ligne") is not None:
            d["ligne"] = tuple(float(v) for v in d["ligne"])
        if d.get("classes_retenues") is not None:
            d["classes_retenues"] = [int(v) for v in d["classes_retenues"]]
        return cls(**d)

    @classmethod
    def depuis_fichier(cls, chemin: str | pathlib.Path) -> "Config":
        return cls.depuis_dict(json.loads(pathlib.Path(chemin).read_text(encoding="utf-8")))

    def vers_fichier(self, chemin: str | pathlib.Path) -> None:
        p = pathlib.Path(chemin)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.vers_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def defauts(cls) -> "Config":
        """Valeurs par défaut issues du fichier versionné."""
        p = chemin_defaut_config()
        if p.exists():
            return cls.depuis_fichier(p)
        # Repli sur les valeurs du dataclass (cas PyInstaller sans config/).
        # La contrainte « toute valeur par défaut doit lire config/default.json »
        # n'est plus respectée : on le signale plutôt que de démarrer en silence.
        log.warning(
            "config/default.json introuvable (%s) : repli sur les valeurs "
            "par défaut codées en dur, qui peuvent diverger du fichier.",
            p,
        )
        return cls()
