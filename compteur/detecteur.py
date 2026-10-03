"""Détection : image numpy -> liste de boîtes, via YOLO (ultralytics).

Ce module est l'entrée du pipeline : il transforme une frame en
`Detection` exploitables par le suivi et le comptage.

Deux règles structurantes :

- `ultralytics` et `torch` ne sont importés qu'à l'intérieur de
  :func:`charger_modele`. Les importer au niveau du module ferait échouer
  toute la suite de tests sur une machine sans ces paquets, et
  `ultralytics` tire derrière lui des modules graphiques.
- l'API ultralytics réellement attendue est vérifiée par les tests :
  ``modele(img, **kw)`` renvoie une **liste** de résultats, ``resultat.boxes``
  est un **attribut**, et ``boxes.xyxy`` / ``.conf`` / ``.cls`` sont des
  **propriétés** renvoyant des tenseurs.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np

from .config import Config
from .types import Detection

log = logging.getLogger(__name__)

FORMATS_MODELES = (".pt", ".onnx")


class Detecteur:
    """Enveloppe mince autour d'un modèle ultralytics.

    Le modèle est injecté pour que les tests n'aient besoin ni d'un GPU ni de
    poids sur disque. En production, le construire avec :func:`charger_modele`.

    Les filtrages (seuil de confiance, taille minimale, classes retenues) sont
    appliqués ici même si le modèle est déjà paramétré pour le seuil : c'est
    la seule garantie que la valeur de `Config` fait autorité, quel que soit
    le modèle fourni.
    """

    def __init__(self, modele, config: Config) -> None:
        self.modele = modele
        self.config = config

    def detecter(self, img: np.ndarray) -> list[Detection]:
        results = self.modele(
            img,
            conf=self.config.seuil_confiance,
            imgsz=self.config.taille_entree,
            verbose=False,
        )
        # Une image en entrée donne un résultat ; ultralytics renvoie une liste.
        resultat = results[0]
        boites = resultat.boxes
        if boites is None or len(boites) == 0:
            return []

        detections: list[Detection] = []
        for coords, score, class_id in zip(
            boites.xyxy.tolist(), boites.conf.tolist(), boites.cls.tolist()
        ):
            x1, y1, x2, y2 = coords
            class_id = int(class_id)
            score = float(score)

            if score < self.config.seuil_confiance:
                continue
            if (
                self.config.classes_retenues is not None
                and class_id not in self.config.classes_retenues
            ):
                continue
            largeur = x2 - x1
            hauteur = y2 - y1
            mini = self.config.taille_min_px
            if largeur < mini or hauteur < mini:
                continue

            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=score,
                    class_id=class_id,
                )
            )
        return detections

    def changer_taille_entree(self, taille: int) -> None:
        """Change la taille d'inférence utilisée au prochain `detecter`."""
        self.config.taille_entree = int(taille)

    def noms_classes(self) -> dict[int, str]:
        """Correspondance id -> nom, telle que le modèle la publie.

        `modele.names` est un dict dans les versions courantes d'ultralytics,
        mais une liste a déjà circulé : les deux formes sont acceptées plutôt
        que de renvoyer silencieusement un dict vide.
        """
        noms = getattr(self.modele, "names", None)
        if isinstance(noms, dict):
            return dict(noms)
        if noms is None:
            return {}
        return {i: str(n) for i, n in enumerate(noms)}


def charger_modele(chemin: str):
    """Charge un modèle YOLO depuis le disque, en privilégiant le GPU.

    Le chemin est validé *avant* tout import lourd : sur une machine sans
    `torch`, l'erreur promise par l'API (`FileNotFoundError`) doit rester
    celle que l'appelant reçoit, pas un `ModuleNotFoundError`.
    """
    p = pathlib.Path(chemin)
    if not p.exists():
        raise FileNotFoundError(
            f"modèle introuvable : {p}\n"
            "Place le fichier .pt dans le dossier du projet, ou change le "
            "réglage « modele » dans l'interface."
        )
    if p.suffix.lower() not in FORMATS_MODELES:
        raise ValueError(
            f"format de modèle non supporté : {p.suffix} "
            f"(attendu : {', '.join(FORMATS_MODELES)})"
        )

    # Imports paresseux : le reste du paquet doit rester utilisable et
    # testable sans ultralytics ni torch.
    import torch
    from ultralytics import YOLO

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info("périphérique de détection : %s", device)
    if device == "cpu":
        log.warning("GPU non disponible : l'analyse sera très lente.")
    return YOLO(str(p))
