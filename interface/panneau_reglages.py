"""Panneau de réglages : produit une Config à partir de widgets Qt.

Le panneau ne décide rien. Il lit des widgets, il rend une `Config`. Toute la
logique de détection reste dans `compteur/`, qui n'importe jamais PySide6.

Trois points méritent un mot, parce qu'ils ne sont pas évidents à la lecture :

- **Les valeurs par défaut vivent dans `config/default.json`, pas ici.** Ce
  module se contente de refléter la `Config` qu'on lui donne. Les bornes des
  widgets sont des contraintes d'interface (on ne saisit pas un seuil de 0,95
  dans une liste déroulante), pas des valeurs métier.
- **Les modèles `.pt` ne sont pas versionnés** (voir `.gitignore`). Le panneau
  liste ce qui est réellement sur le disque et, si le dossier est vide, propose
  un repli au lieu de planter ou d'inventer une liste.
- **`fenetre_lissage` n'existe pas encore dans `Config`** (tâche parallèle).
  On le lit avec `getattr` pour que le panneau fonctionne avant comme après,
  et on ne l'écrit que si le champ existe.
"""

from __future__ import annotations

import logging
import pathlib

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from compteur.config import Config

log = logging.getLogger(__name__)

#: Emplacement par défaut des profils nommés, relatif au dépôt.
DOSSIER_PROFILS = pathlib.Path("config") / "profils"

#: Dossiers balayés pour trouver les poids de modèle, dans cet ordre.
DOSSIERS_MODELES: tuple[pathlib.Path, ...] = (
    pathlib.Path("."),
    pathlib.Path("modeles"),
)

#: Modèle proposé quand aucun `.pt` n'est présent sur le disque.
#:
#: `medium.pt` est un détecteur de têtes entraîné sur SCUT-HEAD. Sur la vidéo de
#: référence il trouve des têtes de 29 px là où `yolov8n-head.pt` en trouve de
#: 21, soit 38 % de marches réelles en plus. Les `.pt` étant absents du dépôt,
#: ce nom est une *suggestion* : le panneau l'affiche, sans l'imposer.
MODELE_REPLI = "medium.pt"

#: Tailles d'entrée proposées. Une valeur absente de la liste est ajoutée à
#: l'affichage plutôt que rejetée en silence.
TAILLES_ENTREE: tuple[int, ...] = (320, 640, 1280)

#: Repli tant que `Config.fenetre_lissage` n'existe pas (tâche parallèle).
FENETRE_LISSAGE_DEFAUT = 10


def copier_config(config: Config, **champs) -> Config:
    """Copie de `config` avec `champs` modifiés.

    `dataclasses.replace` conviendrait, mais passer par `vers_dict`/`depuis_dict`
    impose le même passage JSON qu'un profil : un réglage illisible dans un
    fichier de profil ne peut pas non plus l'être en mémoire.
    """
    donnees = config.vers_dict()
    donnees.update(champs)
    return Config.depuis_dict(donnees)


def lister_modeles(
    dossiers: tuple[pathlib.Path, ...] | list[pathlib.Path] | None = None,
) -> list[str]:
    """Noms des fichiers `.pt` présents dans `dossiers`, sans doublon.

    Un dossier absent, vide ou illisible n'est pas une erreur : le dépôt ne
    contient pas les poids, et une installation neuve doit pouvoir ouvrir le
    panneau.
    """
    if dossiers is None:
        dossiers = DOSSIERS_MODELES
    vus: list[str] = []
    for dossier in dossiers:
        try:
            trouves = sorted(pathlib.Path(dossier).glob("*.pt"))
        except OSError as exc:  # permissions, chemin cassé...
            log.warning("dossier de modèles illisible (%s) : %s", dossier, exc)
            continue
        vus.extend(p.name for p in trouves)
    uniques = list(dict.fromkeys(vus))
    return uniques or [MODELE_REPLI]


def lister_profils(dossier: pathlib.Path | None = None) -> list[str]:
    """Noms des profils déjà enregistrés sur le disque, triés."""
    d = pathlib.Path(dossier) if dossier is not None else DOSSIER_PROFILS
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


class PanneauReglages(QWidget):
    """Quatre groupes de réglages + gestion des profils nommés."""

    config_modifiee = Signal(object)

    def __init__(
        self,
        config: Config,
        parent=None,
        dossiers_modeles: tuple[pathlib.Path, ...] | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._dossiers_modeles = (
            tuple(dossiers_modeles)
            if dossiers_modeles is not None
            else DOSSIERS_MODELES
        )
        # Tant que le panneau se construit, aucun signal ne doit sortir : on
        # pose les widgets dans le silence, l'application n'a rien demandé.
        self._bloquer = True
        self._widgets: dict[str, QWidget] = {}

        racine = QVBoxLayout(self)
        racine.setContentsMargins(8, 8, 8, 8)
        racine.addWidget(self._groupe_detection(config))
        racine.addWidget(self._groupe_tracker(config))
        racine.addWidget(self._groupe_ligne(config))
        racine.addWidget(self._groupe_lissage(config))
        racine.addWidget(self._groupe_profils())
        racine.addStretch(1)

        self._bloquer = False

    # -- Construction ----------------------------------------------------

    def _groupe_detection(self, c: Config) -> QGroupBox:
        g = QGroupBox("Détection")
        v = QVBoxLayout(g)

        v.addWidget(QLabel("Modèle (fichier .pt)"))
        self._modele = QComboBox()
        # Editable : le modèle peut venir d'ailleurs (téléchargé à la main,
        # chemin absolu) et n'est pas forcément dans le dossier balayé.
        self._modele.setEditable(True)
        self._modele.addItems(lister_modeles(self._dossiers_modeles))
        self._modele.setCurrentText(c.modele)
        self._modele.currentTextChanged.connect(self._emettre)
        v.addWidget(self._modele)
        self._widgets["modele"] = self._modele

        self._etiquette_seuil = QLabel()
        v.addWidget(self._etiquette_seuil)
        self._seuil = QDoubleSpinBox()
        self._seuil.setRange(0.05, 0.95)
        self._seuil.setSingleStep(0.05)
        self._seuil.setDecimals(3)
        self._seuil.setValue(c.seuil_confiance)
        self._seuil.valueChanged.connect(self._emettre)
        v.addWidget(self._seuil)
        self._widgets["seuil_confiance"] = self._seuil

        v.addWidget(QLabel("Taille minimale d'une détection (px)"))
        self._taille_min = QSpinBox()
        self._taille_min.setRange(0, 300)
        # À 1280x720 une tête fait 20-29 px : un plancher à 20 éliminait la
        # moitié des détections sur la vidéo de référence.
        self._taille_min.setValue(c.taille_min_px)
        self._taille_min.valueChanged.connect(self._emettre)
        v.addWidget(self._taille_min)
        self._widgets["taille_min_px"] = self._taille_min

        v.addWidget(QLabel("Taille d'entrée"))
        self._taille_entree = QComboBox()
        self._taille_entree.addItems(self._tailles_entree_items(c.taille_entree))
        self._taille_entree.setCurrentText(str(c.taille_entree))
        self._taille_entree.currentTextChanged.connect(self._emettre)
        v.addWidget(self._taille_entree)
        self._widgets["taille_entree"] = self._taille_entree
        return g

    def _groupe_tracker(self, c: Config) -> QGroupBox:
        g = QGroupBox("Tracker")
        v = QVBoxLayout(g)

        v.addWidget(QLabel("Frames de confirmation"))
        self._confirmation = QSpinBox()
        self._confirmation.setRange(1, 20)
        # 3 : les têtes bougent de ~0,7 px par frame, une personne doit être vue
        # trois fois de suite pour être confirmée.
        self._confirmation.setValue(c.frames_confirmation)
        self._confirmation.valueChanged.connect(self._emettre)
        v.addWidget(self._confirmation)
        self._widgets["frames_confirmation"] = self._confirmation

        v.addWidget(QLabel("Survie max sans détection (frames)"))
        self._survie = QSpinBox()
        self._survie.setRange(1, 120)
        self._survie.setValue(c.survie_max)
        self._survie.valueChanged.connect(self._emettre)
        v.addWidget(self._survie)
        self._widgets["survie_max"] = self._survie

        v.addWidget(QLabel("Seuil de matching (IoU)"))
        self._matching = QDoubleSpinBox()
        self._matching.setRange(0.1, 0.9)
        self._matching.setSingleStep(0.05)
        self._matching.setDecimals(3)
        self._matching.setValue(c.seuil_matching)
        self._matching.valueChanged.connect(self._emettre)
        v.addWidget(self._matching)
        self._widgets["seuil_matching"] = self._matching
        return g

    def _groupe_ligne(self, c: Config) -> QGroupBox:
        g = QGroupBox("Ligne de franchissement")
        v = QVBoxLayout(g)

        v.addWidget(QLabel("Épaisseur de la bande (px)"))
        self._epaisseur = QSpinBox()
        self._epaisseur.setRange(5, 100)
        self._epaisseur.setValue(c.epaisseur_bande)
        self._epaisseur.valueChanged.connect(self._emettre)
        v.addWidget(self._epaisseur)
        self._widgets["epaisseur_bande"] = self._epaisseur

        v.addWidget(QLabel("Sens de traversée"))
        self._sens = QComboBox()
        self._sens.addItems(["Avant → après", "Après → avant"])
        self._sens.setCurrentIndex(0 if c.sens == 1 else 1)
        self._sens.currentIndexChanged.connect(self._emettre)
        v.addWidget(self._sens)
        self._widgets["sens"] = self._sens

        v.addWidget(QLabel("Frames d'hystérésis"))
        self._hysteresis = QSpinBox()
        self._hysteresis.setRange(0, 10)
        self._hysteresis.setValue(c.frames_hysteresis)
        self._hysteresis.valueChanged.connect(self._emettre)
        v.addWidget(self._hysteresis)
        self._widgets["frames_hysteresis"] = self._hysteresis
        return g

    def _groupe_lissage(self, c: Config) -> QGroupBox:
        """Fenêtre de lissage des positions.

        Le champ arrive dans `Config` par une tâche parallèle : le panneau le
        lit avec `getattr` pour fonctionner avant comme après, et `lire()` ne
        l'écrit que s'il existe.
        """
        g = QGroupBox("Lissage")
        v = QVBoxLayout(g)
        v.addWidget(QLabel("Fenêtre de lissage (frames)"))
        self._lissage = QSpinBox()
        self._lissage.setRange(1, 60)
        self._lissage.setValue(
            int(getattr(c, "fenetre_lissage", FENETRE_LISSAGE_DEFAUT))
        )
        self._lissage.valueChanged.connect(self._emettre)
        v.addWidget(self._lissage)
        self._widgets["fenetre_lissage"] = self._lissage
        return g

    def _groupe_profils(self) -> QGroupBox:
        g = QGroupBox("Profils")
        h = QVBoxLayout(g)
        self._nom_profil = QComboBox()
        self._nom_profil.setEditable(True)
        self._nom_profil.addItems(lister_profils())
        h.addWidget(self._nom_profil)
        self._btn_enregistrer = QPushButton("Enregistrer ce profil")
        self._btn_enregistrer.clicked.connect(self._on_enregistrer)
        h.addWidget(self._btn_enregistrer)
        self._btn_charger = QPushButton("Charger ce profil")
        self._btn_charger.clicked.connect(self._on_charger)
        h.addWidget(self._btn_charger)
        self._btn_defauts = QPushButton("Valeurs par défaut")
        self._btn_defauts.clicked.connect(self._on_defauts)
        h.addWidget(self._btn_defauts)
        return g

    # -- API publique ----------------------------------------------------

    def modeles_proposes(self) -> list[str]:
        """Noms de modèles actuellement proposés dans la liste déroulante."""
        return [self._modele.itemText(i) for i in range(self._modele.count())]

    def noms_profils(self) -> list[str]:
        """Noms de profils actuellement proposés à l'utilisateur."""
        return [
            self._nom_profil.itemText(i) for i in range(self._nom_profil.count())
        ]

    def lire(self) -> Config:
        """Traduit l'état des widgets en `Config`.

        On part de `self._config` et non d'une `Config()` neuve : la ligne
        tracée et les classes retenues sont fixées ailleurs (dans la vue vidéo),
        et repartir de zéro les effacerait au premier mouvement de curseur.
        """
        c = copier_config(self._config)
        c.modele = self._modele.currentText().strip()
        c.seuil_confiance = round(self._seuil.value(), 3)
        c.taille_min_px = self._taille_min.value()
        c.taille_entree = int(self._taille_entree.currentText())
        c.frames_confirmation = self._confirmation.value()
        c.survie_max = self._survie.value()
        c.seuil_matching = round(self._matching.value(), 3)
        c.epaisseur_bande = self._epaisseur.value()
        c.sens = 1 if self._sens.currentIndex() == 0 else -1
        c.frames_hysteresis = self._hysteresis.value()
        if hasattr(c, "fenetre_lissage"):
            setattr(c, "fenetre_lissage", self._lissage.value())
        return c

    # -- Réglages ponctuels (tests, raccourcis clavier) -------------------

    def definir_seuil(self, valeur: float) -> None:
        self._seuil.setValue(valeur)

    def definir_taille_min(self, valeur: int) -> None:
        self._taille_min.setValue(valeur)

    def definir_epaisseur_bande(self, valeur: int) -> None:
        self._epaisseur.setValue(valeur)

    def definir_lissage(self, valeur: int) -> None:
        self._lissage.setValue(valeur)

    def definir_ligne(self, ligne: tuple[float, float, float, float] | None) -> None:
        """Enregistre la ligne tracée dans la vue vidéo.

        La ligne n'a pas de widget ici : elle se dessine sur l'image. Ce setter
        évite à l'appelant de devoir corriger la `Config` porteuse à la main.
        """
        self._config = copier_config(self._config, ligne=ligne)

    def appliquer(self, config: Config) -> None:
        """Pousse une `Config` dans l'IHM, puis émet une seule fois.

        Les widgets sont inhibés pendant la mise à jour : sans cela, un profil à
        dix réglages déclencherait dix signaux et dix rechargements de modèle.
        """
        self._config = config
        items = self._tailles_entree_items(config.taille_entree)
        self._bloquer = True
        try:
            self._modele.setCurrentText(config.modele)
            self._seuil.setValue(config.seuil_confiance)
            self._taille_min.setValue(config.taille_min_px)
            self._taille_entree.clear()
            self._taille_entree.addItems(items)
            self._taille_entree.setCurrentText(str(config.taille_entree))
            self._confirmation.setValue(config.frames_confirmation)
            self._survie.setValue(config.survie_max)
            self._matching.setValue(config.seuil_matching)
            self._epaisseur.setValue(config.epaisseur_bande)
            self._sens.setCurrentIndex(0 if config.sens == 1 else 1)
            self._hysteresis.setValue(config.frames_hysteresis)
            self._lissage.setValue(
                int(getattr(config, "fenetre_lissage", FENETRE_LISSAGE_DEFAUT))
            )
            self._maj_etiquette_seuil()
        finally:
            self._bloquer = False
        self.config_modifiee.emit(self.lire())

    def enregistrer_profil(
        self, nom: str, dossier: pathlib.Path | None = None
    ) -> pathlib.Path:
        """Écrit `config/profils/<nom>.json` et retourne son chemin."""
        nom = nom.strip()
        if not nom:
            raise ValueError("un profil doit avoir un nom")
        d = pathlib.Path(dossier) if dossier is not None else DOSSIER_PROFILS
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{nom}.json"
        self.lire().vers_fichier(p)
        if nom not in self.noms_profils():
            self._nom_profil.addItem(nom)
        log.info("profil enregistré : %s", p)
        return p

    def charger_profil(
        self, nom: str, dossier: pathlib.Path | None = None
    ) -> bool:
        """Relit un profil et le pousse dans l'IHM. Faux s'il est illisible."""
        nom = nom.strip()
        if not nom:
            return False
        d = pathlib.Path(dossier) if dossier is not None else DOSSIER_PROFILS
        p = d / f"{nom}.json"
        if not p.exists():
            log.info("profil inconnu : %s", p)
            return False
        try:
            config = Config.depuis_fichier(p)
        except Exception as exc:  # noqa: BLE001 — un JSON écrit à la main peut tout être
            log.warning("profil %s illisible : %s", nom, exc)
            return False
        self.appliquer(config)
        return True

    # -- Slots -----------------------------------------------------------

    def _tailles_entree_items(self, valeur: int) -> list[str]:
        tailles = list(TAILLES_ENTREE)
        if valeur not in tailles:
            tailles.append(valeur)
        return [str(t) for t in tailles]

    def _maj_etiquette_seuil(self) -> None:
        self._etiquette_seuil.setText(f"Seuil de confiance — {self._seuil.value():.2f}")

    def _emettre(self, *_args) -> None:
        if self._bloquer:
            return
        self._maj_etiquette_seuil()
        self.config_modifiee.emit(self.lire())

    def _on_enregistrer(self) -> None:
        nom = self._nom_profil.currentText().strip()
        if nom:
            self.enregistrer_profil(nom)

    def _on_charger(self) -> None:
        self.charger_profil(self._nom_profil.currentText().strip())

    def _on_defauts(self) -> None:
        """Reprend `config/default.json`, en conservant la ligne tracée.

        La ligne vit dans la vue vidéo, pas dans un réglage : « valeurs par
        défaut » ne doit pas désarmer le comptage que l'utilisateur met au point.
        """
        defauts = copier_config(Config.defauts(), ligne=self._config.ligne)
        self.appliquer(defauts)