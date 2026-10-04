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
from typing import NamedTuple

import numpy as np

from .config import (
    PERIPHERIQUE_AUTO,
    PERIPHERIQUE_CPU,
    PERIPHERIQUE_CUDA,
    Config,
    mode_peripherique,
)
from .types import Detection

log = logging.getLogger(__name__)

FORMATS_MODELES = (".pt", ".onnx")

#: Valeur passée à ultralytics pour utiliser le GPU.
DEVICE_CUDA = "cuda:0"

#: Libellé affiché quand aucun GPU n'est utilisé, ou quand aucun GPU n'existe.
LIBELLE_CPU = "CPU uniquement"


class ChoixPeripherique(NamedTuple):
    """Ce qui sera réellement utilisé, et pourquoi.

    `avertissement` est `None` quand tout va bien. Il n'est pas vide quand
    l'opérateur a demandé quelque chose que la machine ne peut pas faire : le
    texte est fait pour être affiché tel quel dans la barre de statut.
    """

    device: str
    libelle: str
    cuda: bool
    avertissement: str | None = None


def cuda_disponible() -> bool:
    """Vrai si torch voit un GPU utilisable.

    Import de torch PARESSEUX, pour une raison structurelle : le reste du
    paquet doit rester importable et testable sur une machine sans torch, et
    `compteur/` ne doit rien importer de lourd au niveau du module.

    On interroge `device_count()` et NON `is_available()`. Mesuré sur cette
    machine (Windows, torch 2.14.1+cu126, `CUDA_VISIBLE_DEVICES=""`) :
    `is_available()` renvoie `True` alors que `device_count()` renvoie 0 — le
    premier lit un état initialisé et mis en cache, le second recompte à
    chaque appel. Sur un PC sans carte, c'est exactement le genre d'écart qui
    fait croire à un GPU alors qu'il n'y en a pas, et qui envoie l'opérateur
    chercher une carte graphique pendant une heure.

    Torch absent n'est pas une panne : il n'y a rien à accélérer, donc le
    repli CPU est la seule chose possible. En revanche un torch *présent mais
    cassé* (DLL CUDA manquante) lève : là, dire « CPU uniquement » serait faux,
    puisque le CPU n'est pas forcément utilisable non plus. On laisse
    remonter l'erreur.
    """
    try:
        import torch
    except ImportError:
        log.warning("torch absent : seul le CPU est envisageable.")
        return False

    return int(torch.cuda.device_count()) > 0


def nom_cuda() -> str:
    """Nom commercial de la carte, ou chaîne vide si elle est inconnue."""
    import torch

    try:
        return str(torch.cuda.get_device_name(0)).strip()
    except Exception as exc:  # noqa: BLE001 — nom purely informatif
        log.debug("nom de la carte CUDA illisible : %s", exc)
        return ""


def peripherique_effectif(choisi: str = PERIPHERIQUE_AUTO) -> ChoixPeripherique:
    """Traduit le réglage `Config.peripherique` en périphérique réellement utilisé.

    Les trois cas, et ce qu'ils produisent :

    - `auto` : CUDA si la machine en a un, CPU sinon. Rien à signaler.
    - `cuda` : CUDA exigé. Si la machine n'en a pas, on bascule sur le CPU
      **avec un message** plutôt que de lever. Le choix est assumé : sur le
      terrain, une analyse quatre fois plus lente mais qui rend un chiffre
      vaut mieux qu'un plantage, parce que l'opérateur voit la cause affichée
      et peut décider. Une erreur franche serait plus « propre » en théorie et
      plus pénible sur le terrain : l'opérateur est devant un laptop qui n'a
      pas de GPU NVIDIA, pas devant une bibliothèque.
    - `cpu` : CPU exigé, même avec un GPU présent. Rien à signaler non plus :
      c'est un choix, pas une privation. Si aucun GPU n'est là, on le dit
      quand même — « CPU uniquement » n'a pas la même signification selon la
      machine.
    """
    mode = mode_peripherique(choisi)
    cuda = cuda_disponible()

    if mode == PERIPHERIQUE_CPU:
        avertissement = None
        if cuda:
            avertissement = (
                "Calcul forcé sur le CPU alors qu'un GPU est disponible : "
                "l'analyse sera plus lente."
            )
        else:
            avertissement = "Aucun GPU détecté sur cette machine : calcul sur le CPU."
        return ChoixPeripherique("cpu", LIBELLE_CPU, False, avertissement)

    if mode == PERIPHERIQUE_CUDA and not cuda:
        return ChoixPeripherique(
            "cpu",
            LIBELLE_CPU,
            False,
            "Calcul sur GPU demandé, mais aucun GPU CUDA n'est disponible : "
            "l'analyse bascule sur le CPU, beaucoup plus lentement.",
        )

    if cuda:
        nom = nom_cuda()
        libelle = f"CUDA — {nom}" if nom else "CUDA"
        return ChoixPeripherique(DEVICE_CUDA, libelle, True, None)

    # Mode auto, aucune carte : c'est le cas normal d'un portable sans GPU.
    return ChoixPeripherique(
        "cpu",
        LIBELLE_CPU,
        False,
        "Aucun GPU CUDA détecté : calcul sur le CPU, l'analyse sera très lente.",
    )


def libelle_peripherique(choisi: str = PERIPHERIQUE_AUTO) -> str:
    """Texte court pour la barre de statut : « Calcul : … ».

    Exposé séparément de `peripherique_effectif` parce que l'interface n'a
    besoin que du texte, jamais du.device : afficher « cuda:0 » à
    l'opérateur serait du jargon de développeur.
    """
    return f"Calcul : {peripherique_effectif(choisi).libelle}"


class Detecteur:
    """Enveloppe mince autour d'un modèle ultralytics.

    Le modèle est injecté pour que les tests n'aient besoin ni d'un GPU ni de
    poids sur disque. En production, le construire avec :func:`charger_modele`.

    Les filtrages (seuil de confiance, taille minimale, classes retenues) sont
    appliqués ici même si le modèle est déjà paramétré pour le seuil : c'est
    la seule garantie que la valeur de `Config` fait autorité, quel que soit
    le modèle fourni.

    **Ce module ne connaît pas la ligne de comptage.** Il ne fait qu'appliquer
    un rectangle qu'on lui fournit — voir `detecter`. C'est le `Compteur`, qui
    possède la ligne, qui décide de la bande ; il n'a qu'à la geometrie, jamais
    d'affichage.
    """

    def __init__(self, modele, config: Config) -> None:
        self.modele = modele
        self.config = config

    def detecter(
        self, img: np.ndarray, bande: tuple[int, int, int, int] | None = None
    ) -> list[Detection]:
        """Boîtes de `img`, éventuellement restreintes à `bande`.

        `bande` est un rectangle ``(x1, y1, x2, y2)`` en pixels de l'image
        NATIVE, ou `None` pour analyser l'image entière. `None` est aussi ce
        qui est renvoyé si la bande déborde entièrement de l'image : on
        analyse alors tout, ce qui est plus sûr qu'une coupe vide.

        **Le rognage se fait ici, sur l'image native, AVANT la redimension
        pour le modèle.** C'est le seul endroit où il coûte quelque chose : la
        copie numpy est faite une fois, sur la frame telle qu'elle sort de la
        vidéo. Rogner après la mise à l'échelle `imgsz` reviendrait à
        redimensionner toute l'image puis à jeter 80 % du résultat — le même
        travail, aucun gain.

        **Les coordonnées sont ramenées dans le repère de l'image pleine.** Le
        modèle ne connaît que la bande : ses boîtes sont décalées vers 0. Sans
        la correction ci-dessous, le tracker verrait toutes les personnes
        ramassées contre x=0 et le compteur les ferait passer à côté de la
        ligne — un décompte faux, sans le moindre signe à l'écran puisque
        l'affichage se fait lui aussi en coordonnées d'image pleine.
        """
        decalage_x = decalage_y = 0
        if bande is not None:
            x1, y1, x2, y2 = (int(v) for v in bande)
            # `copy()` : la coupe doit être CONTIGUË. Un slicing numpy est une
            # vue avec un stride, et OpenCV refuse — ou pire, lit de travers —
            # un buffer non contigu. La copie est faite une fois par frame,
            # quelques dizaines de kilo-octets : négligeable devant le calcul.
            img = np.ascontiguousarray(img[y1:y2, x1:x2])
            decalage_x, decalage_y = x1, y1

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
            bx1, by1, bx2, by2 = coords
            class_id = int(class_id)
            score = float(score)

            if score < self.config.seuil_confiance:
                continue
            if (
                self.config.classes_retenues is not None
                and class_id not in self.config.classes_retenues
            ):
                continue
            largeur = bx2 - bx1
            hauteur = by2 - by1
            mini = self.config.taille_min_px
            if largeur < mini or hauteur < mini:
                continue

            detections.append(
                Detection(
                    x1=float(bx1) + decalage_x,
                    y1=float(by1) + decalage_y,
                    x2=float(bx2) + decalage_x,
                    y2=float(by2) + decalage_y,
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


def charger_modele(chemin: str, peripherique: str = PERIPHERIQUE_AUTO):
    """Charge un modèle YOLO depuis le disque, sur le périphérique demandé.

    Le chemin est validé *avant* tout import lourd : sur une machine sans
    `torch`, l'erreur promise par l'API (`FileNotFoundError`) doit rester
    celle que l'appelant reçoit, pas un `ModuleNotFoundError`.

    `peripherique` est le réglage de `Config.peripherique` (`auto`, `cuda`,
    `cpu`). La valeur par défaut est `auto`, ce qui reproduit exactement le
    comportement historique : CUDA si la machine en a un, CPU sinon.
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

    choix = peripherique_effectif(peripherique)
    if choix.avertissement:
        log.warning("%s", choix.avertissement)
    log.info("périphérique de détection : %s", choix.device)
    modele = YOLO(str(p))
    # Le device est passé À LA PRÉDICTION, pas au constructeur : `YOLO(...)`
    # accepte `device=` mais qui l'ignore sur certaines versions, alors que
    # `.to()` passe par le chemin standard de torch et fonctionne partout.
    return modele.to(choix.device)
