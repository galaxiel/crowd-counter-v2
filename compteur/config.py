"""Réglages du compteur : dataclasses, sérialisation JSON, valeurs par défaut."""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from dataclasses import asdict, dataclass, fields

log = logging.getLogger(__name__)


def _racine() -> pathlib.Path:
    """Dossier racine des données de l'application.

    Deux régimes, et ils n'ont rien en commun :

    - **Depuis les sources** (`python main.py`) : le dépôt, déduit de
      l'emplacement de ce fichier — deux niveaux au-dessus de `compteur/`.
    - **Sous PyInstaller** (`sys.frozen`) : le fichier n'est pas dans le
      dossier de l'application. En mode *onefile*, `__file__` pointe vers le
      dossier temporaire d'extraction (`%TEMP%\\_MEIxxxxxx`), et
      `parent.parent` ne remonte nulle part : `config/default.json` serait
      introuvable et `Config.defauts()` partirait sur un repli silencieux.
      Le dossier d'extraction se lit dans `sys._MEIPASS`.

    `sys._MEIPASS` a la priorité : c'est le seul emplacement garanti par le
    fichier `.spec` (`datas=[("config/default.json", "config")]`). Si un
    `one-folder` le définit aussi, les deux points convergent.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return pathlib.Path(meipass)
        # one-folder sans _MEIPASS : le dossier de l'exécutable, où le .spec
        # dépose `config/` en une collection.
        return pathlib.Path(sys.executable).resolve().parent
    return pathlib.Path(__file__).resolve().parent.parent


RACINE = _racine()


def chemin_defaut_config() -> pathlib.Path:
    """Chemin du fichier de valeurs par défaut, versionné avec le projet.

    Résolu à chaque appel, et non figé dans `RACINE` à l'import : sous
    PyInstaller, `RACINE` est figé avant que quiconque ne sache dans quel
    dossier on a été décompressé, et les tests figent `sys.frozen` bien après
    l'import du module. Recalculer rend le chemin vérifiable.
    """
    return _racine() / "config" / "default.json"


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
    # K>1 est VOLONTAIREMENT inactif par défaut. Raison mesurée : le lissage
    # retarde la détection du croisement d'une demi-fenetre, mais le verrou
    # anti-rebond lit toujours le côté de la position BRUTE. Au moment où le
    # croisement lissé est détecté, la position brute est déjà passée de
    # l'autre côté : `_cotes` vaut -1 et `_stabilite` 0, donc le verrou
    # REJETTE. Mesuré : la marche de référence passe de 1 comptage (K=1) à 0
    # (K=5, 10, 20). Pour activer K>1, il faut d'abord faire lire
    # `_cotes`/`_stabilite` à la position lissée elle aussi — voir le rapport
    # de tâche 6, section 4.4.
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
