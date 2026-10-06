"""Mesure de contrôle : décompte sur 3000 frames de la vidéo de référence.

Comparaison dissymétrique (200 avant / 100 après) contre l'ancienne bande
symétrique de 200 px, sur la MÊME ligne. La seule variable est la géométrie.

Usage :
    /c/Python311/python tools/mesurer_bande.py "D:/Bureau/manif_test.mp4"
"""

from __future__ import annotations

import pathlib
import sys
import time

# Le script est lancé depuis `tools/` : le dépôt doit être dans le chemin pour
# que `compteur` soit importable, sans quoi il faut l'installer.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from compteur.compteur import Compteur
from compteur.config import (
    BANDE_APRES_PX,
    BANDE_AVANT_PX,
    Config,
)
from compteur.detecteur import Detecteur, charger_modele, peripherique_effectif
from compteur.ligne import Ligne
from compteur.tracker import Tracker

CADENCE = 0.69  # px/frame, mesuré sur la vidéo de référence
DUREE_AFFICHAGE_FRAMES = round(BANDE_APRES_PX / CADENCE)


def compter(chemin: str, sense: int, avant: int | None, apres: int | None):
    config = Config(ligne=(640.0, 0.0, 640.0, 720.0), sens=sense)
    detecteur = Detecteur(
        charger_modele(config.modele, config.peripherique),
        config,
        half=peripherique_effectif(config.peripherique).cuda,
    )
    compteur = Compteur(config, detecteur, Tracker(config))
    compteur.ajuster_ligne(
        Ligne(
            (640.0, 0.0),
            (640.0, 720.0),
            epaisseur=config.epaisseur_bande,
            sens=sense,
            hysteresis=config.frames_hysteresis,
            bande_avant_px=avant,
            bande_apres_px=apres,
        )
    )

    cap = cv2.VideoCapture(chemin)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    debut = time.time()
    index = 0
    while index < 3000:
        ok, img = cap.read()
        if not ok:
            break
        compteur.traiter_frame(img, index, index / fps)
        index += 1
    cap.release()
    return compteur.total, index, time.time() - debut


def main() -> None:
    chemin = sys.argv[1] if len(sys.argv) > 1 else r"D:\Bureau\manif_test.mp4"
    sense = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    print(f"vidéo {chemin}, ligne x=640, sens={sense}, 3000 frames")
    print(f"cadence mesurée {CADENCE} px/frame")
    print(
        f"durée d'affichage verte = {BANDE_APRES_PX} px / {CADENCE} "
        f"= {DUREE_AFFICHAGE_FRAMES} frames\n"
    )

    resultats = {}
    for nom, avant, apres in (
        ("bande DISSYMÉTRIQUE 200/100", BANDE_AVANT_PX, BANDE_APRES_PX),
        ("bande SYMÉTRIQUE 200 (ancien)", 100, 100),
        ("bande SYMÉTRIQUE 300", 150, 150),
    ):
        total, n, duree = compter(chemin, sense, avant, apres)
        resultats[nom] = total
        print(f"{nom:32} : {total:4d} personnes  ({n} frames, {duree:.0f} s)")

    nouveau = resultats["bande DISSYMÉTRIQUE 200/100"]
    ancien = resultats["bande SYMÉTRIQUE 200 (ancien)"]
    print(f"\n dissymétrique vs ancienne symétrique : {nouveau - ancien:+d} "
          f"({(nouveau - ancien) / max(1, ancien) * 100:+.1f} %)")


if __name__ == "__main__":
    main()