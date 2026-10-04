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
- **Le mode de résolution précède la taille d'entrée, parce qu'il la gouverne.**
  En mode `Automatique`, la liste « Taille d'entrée » est un simple miroir de
  ce que la vidéo impose : elle est GRISÉE, pas effacée, pour que l'opérateur
  voie qu'elle existe et qu'il ne peut pas la piloter. Griser plutôt que
  cacher est un choix : un réglage absent se cherche, un réglage grisé se
  comprend.
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

from compteur.config import (
    MODE_AUTO,
    MODE_MANUEL,
    PERIPHERIQUE_AUTO,
    PERIPHERIQUE_CPU,
    PERIPHERIQUE_CUDA,
    Config,
    mode_peripherique,
    mode_resolution,
    taille_entree_automatique,
)
from compteur.ligne import (
    LIBELLES_SENS,
    ORIENTATION_HORIZONTALE,
    ORIENTATION_VERTICALE,
    Ligne,
)

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

#: Choix de « Résolution d'analyse », `(libellé affiché, mode stocké)`.
#:
#: Le mode stocké est SANS ACCENT (`auto` / `manuel`) : c'est un identifiant
#: qui voyage dans les profils JSON. Le libellé, lui, porte la recommandation,
#: parce que c'est ce que l'opérateur lit.
CHOIX_RESOLUTION_ANALYSE: tuple[tuple[str, str], ...] = (
    ("Automatique (recommandé)", MODE_AUTO),
    ("Manuelle", MODE_MANUEL),
)

#: Choix de « Calcul sur », `(libellé affiché, mode stocké)`.
#:
#: Même contrat que `CHOIX_RESOLUTION_ANALYSE` : le mode stocké est un
#: identifiant sans accent qui voyage dans les profils, le libellé porte la
#: consequence — c'est ce que l'opérateur doit lire avant de choisir.
CHOIX_PERIPHERIQUE: tuple[tuple[str, str], ...] = (
    ("Automatique (recommandé)", PERIPHERIQUE_AUTO),
    ("GPU NVIDIA (CUDA)", PERIPHERIQUE_CUDA),
    ("CPU uniquement", PERIPHERIQUE_CPU),
)

#: Repli tant que `Config.fenetre_lissage` n'existe pas (tâche parallèle).
FENETRE_LISSAGE_DEFAUT = 10

#: Orientation presumed tant qu'aucune ligne n'est tracée.
#:
#: Le panneau doit s'ouvrir avant que l'opérateur ait cliqué quoi que ce soit :
#: sans ligne, il n'y a pas d'axe dominant à regarder, et un menu vide serait
#: moins utile qu'une hypothèse explicite. Verticale est l'hypothèse retenue
#: parce que c'est le seul cas que ce programme sait compter aujourd'hui (la
#: bande traverse le cortège de bout en bout) et parce que les libellés
#: verticaux sont ceux que l'opérateur retrouvera dans neuf cas sur dix.
#: `orientation_par_defaut_sens()` expose cette hypothèse pour qu'un test
#: puisse la vérifier au lieu de la supposer.
ORIENTATION_SANS_LIGNE = ORIENTATION_VERTICALE

#: Explications des réglages, en français, indexées par nom de champ de
#: `Config`. Trois questions par réglage, dans cet ordre : ce que ça fait, ce
#: que ça change quand on monte ou on descend, et la valeur conseillée pour
#: une manifestation filmée en surplomb.
#:
#: Volontairement COURTES — trois à cinq lignes, une ligne par question. Un
#: pavé de douze lignes est un pavé que personne ne lit : au survol, l'opérateur
#: voit un mur et il retient le chiffre, ou rien du tout. Tout ce qui a été
#: retiré ici (redondances, justifications, développement) vit dans le README et
#: dans `docs/design/`, où il peut être lu d'un bout à l'autre.
#:
#: Les chiffres, eux, restent tous : ils ne sont pas des avis, ils viennent des
#: mesures faites sur la vidéo de référence (`D:/Bureau/manif_test.mp4`, 7399
#: frames, medium.pt). Une valeur conseillée sans mesure derrière n'est
#: qu'une opinion déguisée en réglage par défaut.
#:
#: Une seule entrée ne suit pas cette règle, et c'est délibéré :
#: `vitesse_presentation` n'est pas un champ de `Config`, c'est le curseur que
#: l'opérateur actionne le plus souvent APRÈS le lancement, et la confusion
#: qu'il crée (« ralentir l'affichage ralentit le comptage ») coûte une analyse
#: jetée. Son avertissement doit rester impossible à manquer en survol court.
AIDE: dict[str, str] = {
    "modele": (
        "Modèle de détection (.pt).\n\n"
        "Ce que ça fait : cherche des têtes dans l'image — seul réglage qui "
        "change ce qui est vu, les autres mieux exploitent ce qu'il a vu.\n"
        "Ce que ça change : plus lent = meilleures petites têtes, donc plus "
        "de personnes, analyse d'autant plus longue.\n"
        "Valeur conseillée : medium.pt — il trouve des têtes de 29 px là où "
        "yolov8n-head.pt en trouve de 21, soit 38 % de marches en plus."
    ),
    "seuil_confiance": (
        "Seuil de confiance de la détection.\n\n"
        "Ce que ça fait : une tête n'est retenue que si le modèle est sûr "
        "d'elle à plus de ce seuil.\n"
        "Ce que ça change : trop bas, tu perds les lointaines et floues ; "
        "trop haut, il ne reste que les premiers rangs.\n"
        "Valeur conseillée : 0,25. Au-dessus de 0,5, une foule dense vue en "
        "plongeur ne donne presque plus rien."
    ),
    "taille_min_px": (
        "Taille minimale d'une détection (côté, en pixels).\n\n"
        "Ce que ça fait : ignore toute boîte plus petite que ce côté.\n"
        "Ce que ça change : plus haut filtre le bruit et les vraies personnes "
        "lointaines ; plus bas garde plus de monde et plus de faux positifs.\n"
        "Valeur conseillée : 3. Mesuré : une tête fait 20 à 29 px sur cette "
        "vidéo, donc un plancher à 20 filtrerait la moitié des détections."
    ),
    "taille_entree": (
        "Taille à laquelle l'image est réduite avant la détection.\n\n"
        "Ce que ça fait : le modèle voit l'image à cette taille ; les boîtes "
        "sont ramenées à l'échelle de l'originale.\n"
        "Ce que ça change : plus haut trouve plus de petites têtes mais coûte "
        "plus de temps ; plus bas va plus vite au prix de personnes perdues.\n"
        "Valeur conseillée : ne touche pas à ce réglage, laisse « Résolution "
        "d'analyse » sur Automatique. Mesuré : 640 → 1280 sur une vidéo 720p "
        "fait passer le décompte de 315 à 368 (+17 %) sans surcoût de calcul "
        "(272 s contre 260 s)."
    ),
    "resolution_analyse": (
        "Résolution d'analyse — à quelle taille la vidéo est analysée.\n\n"
        "Ce que ça fait : en mode Automatique, l'image est analysée à la "
        "résolution RÉELLE de la vidéo, jamais plus haut que 1280, et « "
        "Taille d'entrée » devient inactif.\n"
        "Ce que ça change : rien au nombre de personnes comptées en soi, c'est "
        "la même scène vue plus finement ; mesuré, 640 → 1280 donne +17 % de "
        "personnes pour +5 % de temps.\n"
        "Valeur conseillée : Automatique. Mesuré sur 3000 frames : 640 compte "
        "231 personnes en 179 s, 1280 en compte 257 en 230 s, 1920 en compte "
        "271 en 324 s — le dernier cran coûte deux fois plus cher pour deux "
        "fois moins de gain."
    ),
    "frames_confirmation": (
        "Frames de confirmation.\n\n"
        "Ce que ça fait : une personne n'entre dans le décompte qu'après "
        "avoir été vue ce nombre de frames d'affilée. Filtre anti-faux "
        "positif.\n"
        "Ce que ça change : plus haut élimine le bruit mais perd les passages "
        "très rapides ; plus bas compte plus tôt, avec plus de bruit.\n"
        "Valeur conseillée : 3 — 0,1 s à 30 i/s, les têtes bougeant de 0,69 px "
        "par frame. Au-delà, on perd des passages sans rien gagner."
    ),
    "survie_max": (
        "Survie max sans détection (frames).\n\n"
        "Ce que ça fait : une personne perdue de vue reste suivie ce nombre de "
        "frames, figée à sa dernière position.\n"
        "Ce que ça change : trop élevé, les tracks mortes gonflent le "
        "compteur de présents (219 → 916 mesuré) sans gagner de comptage ; trop "
        "bas, les tracks cassent et la même personne est comptée plusieurs "
        "fois.\n"
        "Valeur conseillée : 30 — le total ne bouge que de +3 % (315 → 325 à "
        "400 frames). Monter à 150 n'ajoute que 2 personnes."
    ),
    "seuil_matching": (
        "Seuil de matching (IoU).\n\n"
        "Ce que ça fait : deux détections ne sont la même personne que si "
        "leurs boîtes se recouvrent d'au moins ce taux.\n"
        "Ce que ça change : plus haut sépare deux voisins mais perd le suivi "
        "quand quelqu'un bouge ; plus bas fusionne les personnes qui se "
        "croisent.\n"
        "Valeur conseillée : 0,3. Mesuré : les têtes se recouvrent à 0,95 "
        "avec elles-mêmes, mais seulement à 0,32 après 15 px de déplacement — "
        "au-delà de 0,35 le tracker perd la personne."
    ),
    "epaisseur_bande": (
        "Épaisseur de la bande (px).\n\n"
        "Ce que ça fait : la zone que quelqu'un doit traverser pour être "
        "compté. La flèche verte la dessine au milieu.\n"
        "Ce que ça change : plus large absorbe le bruit de position mais "
        "compte les gens arrêtés sur la ligne ; plus étroit devient exigeant "
        "sur une foule compacte.\n"
        "Valeur conseillée : 30, assez large pour le bruit du détecteur et "
        "assez étroit pour qu'un groupe arrêté ne compte pas comme un passage."
    ),
    "sens": (
        "Sens de traversée — le mouvement que tu veux compter.\n\n"
        "Ce que ça fait : ne compte que les personnes allant dans ce sens. Le "
        "libellé suit la ligne tracée, la flèche verte montre le même sens.\n"
        "Ce que ça change : rien si le cortège va dans l'autre sens — tu "
        "obtiens 0 ou 1, inverse et relance dans ce cas.\n"
        "Valeur conseillée : le sens réel du cortège. Mesuré sur la vidéo de "
        "référence : 315 personnes de droite à gauche, 1 dans l'autre sens."
    ),
    "frames_hysteresis": (
        "Frames d'hystérésis.\n\n"
        "Ce que ça fait : il faut avoir été du côté de départ ce nombre de "
        "frames avant qu'un franchissement compte.\n"
        "Ce que ça change : plus haut écarte ceux qui frôlent la ligne mais "
        "perd les passages rapides ; plus bas compte plus vite, avec plus de "
        "risque de double comptage.\n"
        "Valeur conseillée : 2, assez pour écarter un tremblement de "
        "position, assez court pour ne pas retarder le décompte."
    ),
    "fenetre_lissage": (
        "Fenêtre de lissage (frames).\n\n"
        "Ce que ça fait : la position testée est la moyenne des K dernières "
        "positions connues, ce qui filtre le bruit du détecteur.\n"
        "Ce que ça change : augmenter lisse mais retarde le franchissement "
        "d'une demi-fenetre, et le verrou anti-rebond lit encore la position "
        "brute : le comptage est REJETÉ.\n"
        "Valeur conseillée : 1, aucun lissage. Mesuré : la marche de référence "
        "passe de 1 comptage (K=1) à 0 pour K=5, 10 et 20."
    ),
    # Le seul réglage d'AIDE qui ne soit pas un champ de `Config` : la vitesse
    # de présentation ne voyage pas dans un profil, elle est remise à zéro à
    # chaque ouverture. Elle est ici pour une seule raison : ce curseur est
    # celui que l'opérateur actionne le plus souvent APRÈS le lancement, et
    # un réglage aussi fréquent sans explication est un réglage qu'on change
    # au hasard — en croyant ralentir le comptage.
    "vitesse_presentation": (
            "Vitesse de présentation — à quel rythme l'analyse est affichée.\n\n"
            "Ce que ça fait : ralentit ou accélère l'affichage pendant que le "
            "comptage, lui, continue à la vitesse maximale de la machine. Les "
            "images intermédiaires sont sautées, rien n'est recalculé.\n"
            "Ce que ça change : UNIQUEMENT la fluidité de ce que tu vois. NE "
            "CHANGE PAS le décompte : il est exactement le même à 0,25× qu'à 4×. "
            "Ralentir sert à regarder un passage de près.\n"
            "Valeur conseillée : « max », qui affiche chaque image dès qu'elle "
            "est prête. 0,25× pour regarder de près, 1× pour le confort de "
            "lecture, 4× ou « max » pour aller au bout d'une longue vidéo."
    ),
}


def orientation_par_defaut_sens() -> str:
    """Orientation des libellés de sens quand aucune ligne n'est tracée.

    Exposée pour les tests : l'hypothèse « verticale » est un choix, il faut
    donc pouvoir la lire sans passer par un widget.
    """
    return ORIENTATION_SANS_LIGNE


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
        self._sens_actuel: int | None = None

        racine = QVBoxLayout(self)
        racine.setContentsMargins(8, 8, 8, 8)
        racine.addWidget(self._groupe_detection(config))
        racine.addWidget(self._groupe_tracker(config))
        racine.addWidget(self._groupe_ligne(config))
        racine.addWidget(self._groupe_lissage(config))
        racine.addWidget(self._groupe_peripherique(config))
        racine.addWidget(self._groupe_profils())
        racine.addStretch(1)

        self._bloquer = False

    # -- Construction ----------------------------------------------------

    def _aide(self, champ: str) -> str:
        """Texte d'aide d'un réglage, jamais vide.

        Une entrée sans explication est un réglage que l'opérateur ne peut pas
        utiliser en connaissance de cause : il le change au hasard et conclut
        que le programme est faux. `AIDE` est donc IndexError plutôt qu'un
        `.get` silencieux si un champ du panneau n'a pas encore son texte.
        """
        return AIDE[champ]

    def _poser(
        self,
        layout,
        champ: str,
        titre: str,
        widget: QWidget,
        *,
        aide_visible: bool = True,
    ) -> QWidget:
        """Pose un réglage : titre, widget, aide, et enregistre le widget.

        L'aide va à trois endroits, volontairement : l'info-bulle du widget
        (au survol, sur la cible naturelle), l'info-bulle du titre (le survol
        tombe souvent sur le texte, pas sur le champ), et un petit encart
        lisible sous le champ. L'opérateur qui ne découvre pas les info-bulles
        voit quand même, sans rien cliquer, ce que fait le réglage et ce qu'on
        lui conseille.

        L'encart est replié sur sa valeur conseillée quand le texte complet
        serait trop long : c'est le nombre qui décide, pas l'explication.
        """
        aide = self._aide(champ)
        etiquette = QLabel(titre)
        etiquette.setToolTip(aide)
        layout.addWidget(etiquette)
        widget.setToolTip(aide)
        layout.addWidget(widget)
        if aide_visible:
            layout.addWidget(self._encart(aide))
        self._widgets[champ] = widget
        return widget

    def _encart(self, aide: str) -> QLabel:
        """Petit texte d'aide lisible sous un réglage : l'essentiel en deux lignes."""
        conseil = ""
        for ligne in aide.split("\n"):
            if ligne.startswith("Valeur conseillée"):
                conseil = ligne
                break
        resume = "\n".join(ligne for ligne in aide.split("\n") if ligne.startswith("Ce que"))
        texte = f"{resume}\n{conseil}".strip()
        etiquette = QLabel(texte)
        etiquette.setObjectName("aide_reglage")
        etiquette.setWordWrap(True)
        etiquette.setToolTip(aide)
        return etiquette


    def _groupe_detection(self, c: Config) -> QGroupBox:
        g = QGroupBox("Détection")
        v = QVBoxLayout(g)

        self._modele = QComboBox()
        # Editable : le modèle peut venir d'ailleurs (téléchargé à la main,
        # chemin absolu) et n'est pas forcément dans le dossier balayé.
        self._modele.setEditable(True)
        self._modele.addItems(lister_modeles(self._dossiers_modeles))
        self._modele.setCurrentText(c.modele)
        self._modele.currentTextChanged.connect(self._emettre)
        self._poser(v, "modele", "Modèle (fichier .pt)", self._modele)

        # Le titre du seuil porte la valeur : c'est le seul réglage dont
        # l'effet n'est pas lisible dans la boîte qui le contient.
        self._etiquette_seuil = QLabel()
        self._seuil = QDoubleSpinBox()
        self._seuil.setRange(0.05, 0.95)
        self._seuil.setSingleStep(0.05)
        self._seuil.setDecimals(3)
        self._seuil.setValue(c.seuil_confiance)
        self._seuil.valueChanged.connect(self._emettre)
        self._poser(
            v, "seuil_confiance", "", self._seuil, aide_visible=False
        )
        # Le titre reste dans le layout, l'aide non : cet encart serait le
        # troisième sous un réglage qui en a déjà deux.
        v.insertWidget(v.indexOf(self._seuil), self._etiquette_seuil)
        self._etiquette_seuil.setToolTip(self._aide("seuil_confiance"))
        self._seuil.setToolTip(self._aide("seuil_confiance"))
        v.insertWidget(v.indexOf(self._seuil) + 1, self._encart(self._aide("seuil_confiance")))

        self._taille_min = QSpinBox()
        self._taille_min.setRange(0, 300)
        # À 1280x720 une tête fait 20-29 px : un plancher à 20 éliminait la
        # moitié des détections sur la vidéo de référence.
        self._taille_min.setValue(c.taille_min_px)
        self._taille_min.valueChanged.connect(self._emettre)
        self._poser(
            v,
            "taille_min_px",
            "Taille minimale d'une détection (px)",
            self._taille_min,
        )

        self._taille_entree = QComboBox()
        self._taille_entree.addItems(self._tailles_entree_items(c.taille_entree))
        self._taille_entree.setCurrentText(str(c.taille_entree))
        self._taille_entree.currentTextChanged.connect(self._emettre)

        # « Résolution d'analyse » PRÉCÈDE « Taille d'entrée », parce que c'est
        # lui qui décide si la taille d'entrée compte encore. Dans l'ordre
        # inverse, l'opérateur règle une valeur sans aucun effet, et ne le voit
        # pas : le réglage manuel est en dessous, actif ou non.
        self._resolution = QComboBox()
        for libelle, _mode in CHOIX_RESOLUTION_ANALYSE:
            self._resolution.addItem(libelle)
        self._resolution.setCurrentIndex(
            max(
                0,
                [m for _, m in CHOIX_RESOLUTION_ANALYSE].index(
                    mode_resolution(c.resolution_analyse)
                ),
            )
        )
        self._resolution.currentIndexChanged.connect(self._emettre)
        self._poser(
            v, "resolution_analyse", "Résolution d'analyse", self._resolution
        )

        self._poser(v, "taille_entree", "Taille d'entrée", self._taille_entree)
        self._maj_activite_taille_entree()
        return g

    def _groupe_tracker(self, c: Config) -> QGroupBox:
        g = QGroupBox("Tracker")
        v = QVBoxLayout(g)

        self._confirmation = QSpinBox()
        self._confirmation.setRange(1, 20)
        # 3 : les têtes bougent de ~0,7 px par frame, une personne doit être vue
        # trois fois de suite pour être confirmée.
        self._confirmation.setValue(c.frames_confirmation)
        self._confirmation.valueChanged.connect(self._emettre)
        self._poser(
            v,
            "frames_confirmation",
            "Frames de confirmation",
            self._confirmation,
        )

        self._survie = QSpinBox()
        # La plage monte à 400 : on a MESURÉ que 400 frames change le total
        # (325 contre 315) en faisant passer les présents simultanés de 219 à
        # 916. Le réglage doit donc être atteignable pour que ce constat soit
        # vérifiable par l'opérateur, pas seulement dans un rapport.
        self._survie.setRange(1, 400)
        self._survie.setValue(c.survie_max)
        self._survie.valueChanged.connect(self._emettre)
        self._poser(
            v, "survie_max", "Survie max sans détection (frames)", self._survie
        )

        self._matching = QDoubleSpinBox()
        self._matching.setRange(0.1, 0.9)
        self._matching.setSingleStep(0.05)
        self._matching.setDecimals(3)
        self._matching.setValue(c.seuil_matching)
        self._matching.valueChanged.connect(self._emettre)
        self._poser(
            v, "seuil_matching", "Seuil de matching (IoU)", self._matching
        )
        return g

    def _groupe_ligne(self, c: Config) -> QGroupBox:
        g = QGroupBox("Ligne de franchissement")
        v = QVBoxLayout(g)

        self._epaisseur = QSpinBox()
        self._epaisseur.setRange(5, 100)
        self._epaisseur.setValue(c.epaisseur_bande)
        self._epaisseur.valueChanged.connect(self._emettre)
        self._poser(
            v, "epaisseur_bande", "Épaisseur de la bande (px)", self._epaisseur
        )

        self._sens = QComboBox()
        # Les libellés dépendent de la ligne TRACÉE, pas d'un sens abstrait :
        # « Avant → après » ne disait pas avant quoi, et l'opérateur a dû
        # essayer les deux sens pour trouver le bon. `remplir_libelles_sens`
        # les refait dès que la ligne change.
        self._sens.currentIndexChanged.connect(self._emettre)
        self._poser(v, "sens", "Sens de traversée", self._sens, aide_visible=False)
        self._remplir_libelles_sens(c.sens, self._config.ligne)
        self._sens.setToolTip(self._aide("sens"))
        v.insertWidget(
            v.indexOf(self._sens) + 1, self._encart(self._aide("sens"))
        )

        self._hysteresis = QSpinBox()
        self._hysteresis.setRange(0, 10)
        self._hysteresis.setValue(c.frames_hysteresis)
        self._hysteresis.valueChanged.connect(self._emettre)
        self._poser(
            v,
            "frames_hysteresis",
            "Frames d'hystérésis",
            self._hysteresis,
        )
        return g

    def _groupe_lissage(self, c: Config) -> QGroupBox:
        """Fenêtre de lissage des positions.

        Le champ arrive dans `Config` par une tâche parallèle : le panneau le
        lit avec `getattr` pour fonctionner avant comme après, et `lire()` ne
        l'écrit que s'il existe.
        """
        g = QGroupBox("Lissage")
        v = QVBoxLayout(g)
        self._lissage = QSpinBox()
        self._lissage.setRange(1, 60)
        self._lissage.setValue(
            int(getattr(c, "fenetre_lissage", FENETRE_LISSAGE_DEFAUT))
        )
        self._lissage.valueChanged.connect(self._emettre)
        self._poser(
            v, "fenetre_lissage", "Fenêtre de lissage (frames)", self._lissage
        )
        return g

    def _groupe_peripherique(self, c: Config) -> QGroupBox:
        """Choix du processeur de détection.

        Groupe à part entière, et non une ligne de plus dans « Détection » :
        ce n'est pas un réglage de la détection mais du MATÉRIEL qui la porte,
        et sa consequence (un facteur quatre sur la vitesse) est d'une autre
        nature que celle des seuils. L'opérateur doit pouvoir le retrouver —
        et le comprendre même après un plantage du GPU — sans le confondre avec
        un seuil qu'il aurait mal réglé.

        Pas d'encart d'aide ici, et c'est voulu : l'opérateur n'a pas à apprendre
        ce que veut dire « CUDA » au survol. Les trois choix se lisent sur leurs
        libellés, et le périphérique RÉELLEMENT utilisé est écrit en permanence
        dans la barre de statut, sous le compteur — c'est là que l'information
        se trouve, pas dans une bulle qu'on ne voit pas. Le réglage est donc
        construit à la main, sans `_poser` : `_poser` exige une entrée dans
        `AIDE`, et il n'y en a plus.
        """
        g = QGroupBox("Périphérique de calcul")
        v = QVBoxLayout(g)
        self._peripherique = QComboBox()
        for libelle, _mode in CHOIX_PERIPHERIQUE:
            self._peripherique.addItem(libelle)
        self._peripherique.setCurrentIndex(
            max(
                0,
                [m for _, m in CHOIX_PERIPHERIQUE].index(
                    mode_peripherique(getattr(c, "peripherique", PERIPHERIQUE_AUTO))
                ),
            )
        )
        self._peripherique.currentIndexChanged.connect(self._emettre)
        etiquette = QLabel("Calcul sur")
        v.addWidget(etiquette)
        v.addWidget(self._peripherique)
        self._widgets["peripherique"] = self._peripherique
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
        c.resolution_analyse = mode_resolution(self._mode_resolution())
        # En mode automatique, la liste déroulante n'est qu'un MIROIR de ce que
        # la vidéo courante impose ; elle ne décide de rien. La renvoyer quand
        # même évite que la Config affichée contredise l'écran.
        c.taille_entree = int(self._taille_entree.currentText())
        c.frames_confirmation = self._confirmation.value()
        c.survie_max = self._survie.value()
        c.seuil_matching = round(self._matching.value(), 3)
        c.epaisseur_bande = self._epaisseur.value()
        # Le sens n'est plus déduit de l'INDICE mais de la valeur associée au
        # libellé affiché : réordonner le menu pour cause de ligne horizontale
        # ne peut plus inverser le sens compté en silence.
        choix = [v for _, v in self.choix_sens()]
        index = self._sens.currentIndex()
        c.sens = choix[index] if 0 <= index < len(choix) else int(self._sens_actuel or 1)
        c.frames_hysteresis = self._hysteresis.value()
        if hasattr(c, "fenetre_lissage"):
            setattr(c, "fenetre_lissage", self._lissage.value())
        if hasattr(self, "_peripherique"):
            c.peripherique = mode_peripherique(self._mode_peripherique())
        return c

    def _mode_peripherique(self) -> str:
        """Mode de périphérique actuellement sélectionné.

        Même règle que `_mode_resolution` : le menu porte le MODE, jamais
        l'indice, pour que réordonner `CHOIX_PERIPHERIQUE` ne puisse pas
        retourner silencieusement un autre mode que celui affiché.
        """
        index = self._peripherique.currentIndex()
        modes = [m for _, m in CHOIX_PERIPHERIQUE]
        if 0 <= index < len(modes):
            return modes[index]
        return PERIPHERIQUE_AUTO

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

        Les libellés du menu de sens sont recalculés ICI et pas seulement à la
        construction : c'est le moment où l'on passe d'une orientation supposée
        à une orientation réelle. Le `sens` sélectionné est reporté tel quel —
        une ligne verticale qui devient horizontale renomme les choix, elle ne
        doit pas changer ce qui est compté.
        """
        self._config = copier_config(self._config, ligne=ligne)
        if hasattr(self, "_sens"):
            self._remplir_libelles_sens(self._sens_actuel, ligne)

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
            self._resolution.setCurrentIndex(
                max(
                    0,
                    [m for _, m in CHOIX_RESOLUTION_ANALYSE].index(
                        mode_resolution(config.resolution_analyse)
                    ),
                )
            )
            self._taille_entree.clear()
            self._taille_entree.addItems(items)
            self._taille_entree.setCurrentText(str(config.taille_entree))
            self._confirmation.setValue(config.frames_confirmation)
            self._survie.setValue(config.survie_max)
            self._matching.setValue(config.seuil_matching)
            self._epaisseur.setValue(config.epaisseur_bande)
            self._remplir_libelles_sens(config.sens, config.ligne)
            self._hysteresis.setValue(config.frames_hysteresis)
            self._lissage.setValue(
                int(getattr(config, "fenetre_lissage", FENETRE_LISSAGE_DEFAUT))
            )
            if hasattr(self, "_peripherique"):
                self._peripherique.setCurrentIndex(
                    max(
                        0,
                        [m for _, m in CHOIX_PERIPHERIQUE].index(
                            mode_peripherique(
                                getattr(config, "peripherique", PERIPHERIQUE_AUTO)
                            )
                        ),
                    )
                )
            self._maj_etiquette_seuil()
            self._maj_activite_taille_entree()
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

    def _ligne_de_la_config(self, ligne=None) -> Ligne | None:
        """`Ligne` construite depuis la ligne mémorisée, ou ``None``.

        On ne construit que ce qu'il faut pour NOMMER les libellés : une `Ligne`
        valide lève sur deux points confondus, or une valeur mémorisée peut
        venir d'un fichier de profil écrit à la main. Une ligne inexploitable
        ne doit pas empêcher le panneau de s'ouvrir — l'opérateur en tracera
        une autre.
        """
        points = self._config.ligne if ligne is None else ligne
        if points is None:
            return None
        try:
            return Ligne((points[0], points[1]), (points[2], points[3]))
        except (ValueError, TypeError, IndexError) as exc:
            log.warning("ligne mémorisée inexploitable (%s) : %s", points, exc)
            return None

    def choix_sens(self) -> list[tuple[str, int]]:
        """Les deux choix affichés, ``(libellé, sens)``, d'après la ligne tracée.

        Sans ligne tracée, l'orientation est supposée verticale — voir
        `ORIENTATION_SANS_LIGNE`. C'est le SEUL endroit qui décide de l'ordre
        des libellés : `lire()` et `appliquer()` passent tous deux par là, donc
        un menu et un décompte ne peuvent pas se contredire.
        """
        ligne = self._ligne_de_la_config()
        if ligne is None:
            return [
                (LIBELLES_SENS[orientation_par_defaut_sens()][0], 1),
                (LIBELLES_SENS[orientation_par_defaut_sens()][1], -1),
            ]
        return ligne.choix_sens()

    def _remplir_libelles_sens(
        self, sens: int | None = None, ligne=None
    ) -> None:
        """Recalcule les libellés du menu de sens et resélectionne ``sens``.

        Le `sens` PROTÉGÉ est prioritaire : changer la ligne ne doit pas changer
        le sens compté. C'est tout l'intérêt d'un libellé explicite — l'opérateur
        qui a réglé son sens le retrouve, écrit à l'identique, sur la vidéo
        suivante.
        """
        if sens is None:
            sens = self._sens_actuel
        if sens is None:
            sens = self._config.sens
        # Passe par `_ligne_de_la_config` même quand la ligne est fournie : une
        # ligne inexploitable (profil écrit à la main) doit retomber sur
        # l'orientation par défaut, pas lever une ValueError depuis un setter.
        memorisee = self._config.ligne
        self._config = copier_config(self._config, ligne=ligne)
        try:
            choix = self.choix_sens()
        finally:
            self._config = copier_config(self._config, ligne=memorisee)
        self._sens_actuel = int(sens) if sens in (1, -1) else 1
        etait_bloque = self._bloquer
        self._bloquer = True
        try:
            self._sens.clear()
            for libelle, _valeur in choix:
                self._sens.addItem(libelle)
            self._sens.setCurrentIndex(
                max(0, [v for _, v in choix].index(self._sens_actuel))
            )
        finally:
            self._bloquer = etait_bloque

    def _mode_resolution(self) -> str:
        """Mode de résolution d'analyse actuellement sélectionné.

        Le menu porte le MODE, jamais l'indice : réordonner `CHOIX_RESOLUTION_
        ANALYSE` ne peut pas retourner silencieusement le mode stocké, à
        l'inverseexactement du bug que `sens` a déjà donné une fois.
        """
        index = self._resolution.currentIndex()
        modes = [m for _, m in CHOIX_RESOLUTION_ANALYSE]
        if 0 <= index < len(modes):
            return modes[index]
        return MODE_AUTO

    def _maj_activite_taille_entree(self) -> None:
        """Active la taille d'entrée SEULEMENT en mode manuel.

        Griser le réglage plutôt que le cacher : l'opérateur voit qu'il existe,
        qu'il est inactif, et pourquoi. Le laisser actif en mode automatique
        donnerait l'impression qu'une taille choisie là compte.
        """
        manuel = self._mode_resolution() == MODE_MANUEL
        self._taille_entree.setEnabled(manuel)

    def definir_peripherique(self, mode: str) -> None:
        """Sélectionne un mode de périphérique (raccourci et tests).

        Le menu reste la voie normale : cette méthode existe pour qu'un test
        n'ait pas à énumérer les choix, ce qui casse silencieusement dès que
        l'ordre du menu change.
        """
        cible = mode_peripherique(mode)
        index = [m for _, m in CHOIX_PERIPHERIQUE].index(cible)
        if self._peripherique.currentIndex() != index:
            self._peripherique.setCurrentIndex(index)

    def definir_resolution_manuelle(self, manuel: bool = True) -> None:
        """Bascule le mode de résolution d'analyse (raccourci et tests).

        Le menu reste la voie normale : cette méthode existe pour qu'un test
        n'ait pas à énumérer les choix, ce qui casse silencieusement dès que
        l'ordre du menu change.
        """
        cible = MODE_MANUEL if manuel else MODE_AUTO
        index = [m for _, m in CHOIX_RESOLUTION_ANALYSE].index(cible)
        if self._resolution.currentIndex() != index:
            self._resolution.setCurrentIndex(index)
        self._maj_activite_taille_entree()

    def definir_resolution_video(self, largeur: int, hauteur: int) -> int:
        """Recalcule la résolution d'analyse d'après la vidéo chargée.

        Renvoie la taille EFFECTIVE retenue, que la fenêtre affiche dans la
        barre de statut : l'opérateur doit pouvoir voir à quoi sa vidéo est
        analysée, sinon « automatique » est une promesse qu'il ne peut pas
        vérifier.

        En mode manuel, la vidéo est ignorée et rien ne bouge : c'est
        précisément ce que l'opérateur a demandé.
        """
        if self._mode_resolution() != MODE_AUTO:
            return int(self._taille_entree.currentText())
        taille = taille_entree_automatique(largeur, hauteur)
        etait_bloque = self._bloquer
        self._bloquer = True
        try:
            items = self._tailles_entree_items(taille)
            if self._taille_entree.count() != len(items) or [
                self._taille_entree.itemText(i)
                for i in range(self._taille_entree.count())
            ] != items:
                self._taille_entree.clear()
                self._taille_entree.addItems(items)
            self._taille_entree.setCurrentText(str(taille))
            self._config = copier_config(self._config, taille_entree=taille)
        finally:
            self._bloquer = etait_bloque
        return taille

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
        choix = [v for _, v in self.choix_sens()]
        index = self._sens.currentIndex()
        if 0 <= index < len(choix):
            self._sens_actuel = choix[index]
        self._maj_etiquette_seuil()
        self._maj_activite_taille_entree()
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