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
#: Les chiffres ne sont pas des avis : ils viennent des mesures faites sur la
#: vidéo de référence (`D:/Bureau/manif_test.mp4`, 7399 frames, medium.pt). Une
#: valeur conseillée sans mesure derrière n'est qu'une opinion déguisée en
#: réglage par défaut, et c'est précisément ce que l'opérateur ne peut pas
#: vérifier lui-même.
AIDE: dict[str, str] = {
    "modele": (
        "Modèle de détection (.pt).\n\n"
        "Ce que ça fait : cherche des têtes dans l'image. C'est le seul "
        "réglage qui change ce qui est vu, tous les autres ne font que "
        "mieux exploiter ce qu'il a vu.\n"
        "Ce que ça change : un modèle plus lent détecte mieux les petites "
        "têtes, donc plus de personnes, mais l'analyse ralentit d'autant.\n"
        "Valeur conseillée : medium.pt. Mesuré sur la vidéo de référence, il "
        "trouve des têtes de 29 px là où yolov8n-head.pt en trouve de 21, "
        "soit 38 % de marches réelles en plus."
    ),
    "seuil_confiance": (
        "Seuil de confiance de la détection.\n\n"
        "Ce que ça fait : une tête n'est retenue que si le modèle est sûr "
        "d'elle à plus de ce seuil.\n"
        "Ce que ça change : en dessous, tu perds les personnes lointaines et "
        "floues — donc des pans entiers de foule si la caméra est haute. Au "
        "dessus, tu ne gardes que les visages nets, donc les premiers rangs.\n"
        "Valeur conseillée : 0,25. Au-dessus de 0,5, une foule dense vue en "
        "plongeur ne donne presque plus rien : les têtes y sont petites et le "
        "modèle est peu sûr."
    ),
    "taille_min_px": (
        "Taille minimale d'une détection (côté, en pixels).\n\n"
        "Ce que ça fait : ignore toute boîte plus petite que ce côté.\n"
        "Ce que ça change : augmenter filtre le bruit, mais aussi les vraies "
        "persones lointaines. Diminuer garde plus de monde, et plus de faux "
        "positifs — une tache sombre dans la foule.\n"
        "Valeur conseillée : 3. Mesuré : une tête détectée sur cette vidéo "
        "fait 20 à 29 px. Un plancher à 20 filtrerait donc la moitié des "
        "détections."
    ),
    "taille_entree": (
        "Taille à laquelle l'image est réduite avant la détection.\n\n"
        "Ce que ça fait : le modèle voit l'image à cette largeur, en "
        "redimensionnant la vidéo. Les boîtes sont ramenées à l'échelle de "
        "l'originale.\n"
        "Ce que ça change : augmenter trouve plus de petites têtes, donc plus "
        "de personnes, mais coûte plus de temps de calcul. Diminuer fait "
        "l'inverse et va plus vite, au prix de personnes perdues.\n"
        "Valeur conseillée : 1280. Mesuré sur la vidéo de référence, le "
        "décompte passe de 315 à 368 (+17 %) sans surcoût de calcul notable "
        "(272 s contre 260 s sur 7399 frames), pour 64 détections par frame "
        "au lieu de 48. C'est le SEUL réglage de la liste qui vaille le coup "
        "d'être modifié. Une caméra de surveillance plafonne en général à "
        "1280 : monte aussi haut que la source le permet. Au-delà, 1920 ne "
        "rapporte que 149 détections par frame pour un temps de calcul bien "
        "supérieur — c'est le meilleur rapport, pas le plus gros chiffre."
    ),

    "frames_confirmation": (
        "Frames de confirmation.\n\n"
        "Ce que ça fait : une personne n'apparaît dans le décompte "
        "qu'après avoir été vue ce nombre de frames d'affilée. C'est le "
        "filtre anti-faux positif.\n"
        "Ce que ça change : augmenter élimine les faux positifs, mais fait "
        "perdre les personnes qui traversent très vite, et ralentit le "
        "décompte. Diminuer compte plus tôt et plus fort, avec plus de bruit.\n"
        "Valeur conseillée : 3. Mesuré : les têtes se déplacent de 0,69 px "
        "par frame, une personne reste donc des dizaines de frames dans le "
        "champ ; 3 = 0,1 s à 30 i/s. Au-dessus de 3, on commence à perdre des "
        "passages sans rien gagner en fiabilité — le bruit est déjà filtré."
    ),
    "survie_max": (
        "Survie max sans détection (frames).\n\n"
        "Ce que ça fait : une personne que le détecteur perd de vue reste "
        "quand même suivie ce nombre de frames, à sa dernière position.\n"
        "Ce que ça change : augmenter préserve les personnes derrière un "
        "groupe ou un drapeau, mais elles gardent leur dernière position — donc "
        "un groupe arrêté sur la ligne gonfle le nombre de présents et peut "
        "faire compter un faux passage. Diminuer casse les tracks et compte la "
        "même personne plusieurs fois.\n"
        "Valeur conseillée : 30. Mesuré sur la vidéo de référence, le total "
        "passe de 315 à 314 (60 frames), 317 (150) et 325 (400) : +3 % au "
        "maximum, alors que les présents simultanés passent de 219 à 916 et "
        "les tracks vus chutent de 9873 à 4315. La mémoire prédictive ne "
        "rachète presque rien ici et rend le compteur de présents faux : "
        "reste à 30. Si tu subis de longues occultations — drapeaux, portiques "
        "— monte à 150, le surcoût y est de 2 personnes."
    ),

    "seuil_matching": (
        "Seuil de matching (IoU).\n\n"
        "Ce que ça fait : deux détections ne sont la même personne que si "
        "leurs boîtes se recouvrent d'au moins ce taux.\n"
        "Ce que ça change : augmenter sépare mieux deux personnes voisines, "
        "mais perd le suivi dès qu'une personne bouge entre deux frames. "
        "Diminuer garde le suivi mais fusionne deux personnes qui se "
        "croisent.\n"
        "Valeur conseillée : 0,3. Mesuré : ici les têtes se déplacent de "
        "0,69 px par frame, soit un recouvrement de 0,95 avec elles-mêmes. Une "
        "tête de 29 px qui en bouge de 15 px par frame ne recouvre plus qu'à "
        "0,32 : au-dessus de 0,35, le tracker la perdrait."
    ),
    "epaisseur_bande": (
        "Épaisseur de la bande (px).\n\n"
        "Ce que ça fait : la zone que quelqu'un doit traverser pour être "
        "compté. La flèche verte la dessine au milieu.\n"
        "Ce que ça change : élargir aide quand les positions oscillent, mais "
        "compte aussi les gens qui s'arrêtent sur la ligne. Rétrécir devient "
        "exigeant sur une foule compacte.\n"
        "Valeur conseillée : 30. Assez large pour absorber le bruit de "
        "position du détecteur, assez étroit pour qu'un groupe arrêté sur la "
        "ligne ne soit pas compté comme un passage."
    ),
    "sens": (
        "Sens de traversée — le mouvement que tu veux compter.\n\n"
        "Ce que ça fait : ne compte que les personnes qui vont dans ce sens. "
        "Le libellé suit l'orientation de la ligne que tu as tracée : "
        "« Gauche → droite » / « Droite → gauche » sur une ligne verticale, "
        "« Haut → bas » / « Bas → haut » sur une ligne horizontale. La flèche "
        "verte dessinée sur l'image montre le même sens.\n"
        "Ce que ça change : rien au décompte si le cortège va dans l'autre "
        "sens — tu obtiens zéro ou presque. Si tu obtiens le chiffre à 0 ou à "
        "1, c'est presque toujours celui-ci : inverse et relance.\n"
        "Valeur conseillée : choisis le sens dans lequel marche ton cortège. "
        "Sur la vidéo de référence, le cortège va de droite à gauche : 315 "
        "personnes dans ce sens, 1 dans l'autre."
    ),
    "frames_hysteresis": (
        "Frames d'hystérésis.\n\n"
        "Ce que ça fait : il faut avoir été du côté de départ pendant ce "
        "nombre de frames avant qu'un franchissement compte.\n"
        "Ce que ça change : augmenter filtre les gens qui frôlent la ligne, "
        "mais fait perdre ceux qui passent vite. Diminuer compte plus vite, "
        "avec plus de risque de double comptage.\n"
        "Valeur conseillée : 2. Assez pour écarter un tremblement de "
        "position, assez court pour ne pas retarder le décompte."
    ),
    "fenetre_lissage": (
        "Fenêtre de lissage (frames).\n\n"
        "Ce que ça fait : la position servant au test de franchissement est "
        "la moyenne des K dernières positions connues, ce qui filtre le bruit "
        "du détecteur.\n"
        "Ce que ça change : augmenter lisse davantage, mais retarde le "
        "franchissement d'une demi-fenetre — et le verrou anti-rebond lit "
        "encore la position brute, donc le comptage est REJETÉ.\n"
        "Valeur conseillée : 1, c'est-à-dire aucun lissage. Mesuré : la "
        "marche de référence passe de 1 comptage (K=1) à 0 comptage pour "
        "K=5, 10 et 20. K>1 est aujourd'hui nuisible, il n'est exposé que "
        "parce que le champ existe dans la configuration."
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
        self._poser(v, "taille_entree", "Taille d'entrée", self._taille_entree)
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
        # Le sens n'est plus déduit de l'INDICE mais de la valeur associée au
        # libellé affiché : réordonner le menu pour cause de ligne horizontale
        # ne peut plus inverser le sens compté en silence.
        choix = [v for _, v in self.choix_sens()]
        index = self._sens.currentIndex()
        c.sens = choix[index] if 0 <= index < len(choix) else int(self._sens_actuel or 1)
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