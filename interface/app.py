"""Fenêtre principale : assemble le moteur, la lecture et l'affichage.

Trois décisions structurent ce module, et chacune a un test qui la verrouille.

**1. Deux minuteries, jamais une.** Le traitement d'une frame (détection YOLO
sur GPU, tracking, franchissement) et son affichage ont deux cadences
indépendantes : le traitement est aussi rapide que la machine le permet,
l'affichage est ce que l'opérateur a choisi. Le curseur de vitesse ne règle
donc QUE l'intervalle du timer d'affichage — celui du traitement reste à 0 ms.
Avec un seul minuteur (ce que le plan proposait), ralentir l'affichage
ralentirait le COMPTAGE : le décompte dépendrait de la vitesse de présentation.

**2. La ligne se trace à la souris, en coordonnées image.** `WidgetVideo.clic`
émet déjà des pixels de l'image SOURCE ; ce module ne fait que les assembler en
`Ligne`. Toute la conversion écran -> pixel est dans le widget, testée
ailleurs : la dupliquer ici rouvrirait le risque d'une ligne décalée, invisible
à l'écran et qui ne se révèle qu'en comptant faux.

**3. Rien de lourd au démarrage.** CUDA est présent et `medium.pt` met plusieurs
secondes à monter. Le modèle est chargé sur « Lancer », jamais dans
`__init__` : la fenêtre s'ouvre immédiatement, l'opérateur trace sa ligne
pendant que le GPU se réveille.

Les boîtes de dialogue sont modales **seulement** si la fenêtre est visible.
Sans cette garde, un test qui provoque une erreur resterait bloqué sur un
« OK » que personne ne peut cliquer : le message tombe alors dans le journal et
dans la barre de statut, qui est visible dans les deux cas.
"""

from __future__ import annotations

import logging
import pathlib

import cv2
from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from compteur import VERSION
from compteur.compteur import TAILLE_LOT_DETECTION, Compteur
from compteur.config import (
    BANDE_APRES_PX,
    BANDE_AVANT_PX,
    DUREE_BOITE_COMPTEE_FRAMES,
    MODE_MANUEL,
    Config,
    mode_peripherique,
    mode_resolution,
)
from compteur.detecteur import (
    Detecteur,
    charger_modele,
    peripherique_effectif,
)
from compteur.ligne import Ligne
from compteur.tracker import Tracker
from interface.overlay import dessiner
from interface.panneau_reglages import (
    LIBELLES_SENS,
    PanneauReglages,
    orientation_par_defaut_sens,
)
from interface.recap import PanneauRecap
from interface.style import appliquer_style  # noqa: F401  (ré-export pour main.py)
from interface.widgets_video import WidgetVideo

log = logging.getLogger(__name__)

#: Cadence de référence quand la vidéo n'annonce pas la sienne (codec exotique,
#: conteneur mal écrit). 25 i/s est le standard des caméras de surveillance.
FPS_REPLI = 25.0

#: Cadence d'affichage pendant une analyse, en images par seconde. Le
#: traitement, lui, n'est PAS limité (timer à intervalle 0) : c'est le rendu
#: CPU par frame affichée qui coûte cher, et l'opérateur n'a pas besoin d'un
#: aperçu fluide pour lire un compteur. Voir le bloc des minuteries.
CADENCE_AFFICHAGE_IPS = 15

#: Lien du bouton de soutien. Le `data-slug` du widget Buy Me a Coffee fourni
#: par l'opérateur est `galaxiel` — le bouton natif ouvre cette page dans le
#: navigateur, ce qui donne le même résultat sans embarqué de JavaScript.
LIEN_SOUTIEN = "https://buymeacoffee.com/galaxiel"

#: Durée du flash rouge de comptage, en frames AFFICHÉES. Le flash sert à
#: repérer d'un coup d'œil l'instant d'un comptage ; le compter en frames
#: traitées le rendrait subliminal dès qu'on ralentit l'affichage.
FLASH_FRAMES = 3

#: Filtres du sélecteur de fichier, par ordre de préférence.
FILTRES_VIDEOS = (
    "Vidéos (*.mp4 *.avi *.mkv *.mov *.m4v *.webm);;Tous les fichiers (*)"
)

#: TAILLE DE LA LISTE des boîtes vertes affichées après un comptage.
#:
#: La liste est bornée DEUX fois, par deux règles indépendantes, parce que
#: chacune rattrape une défaillance de l'autre :
#:
#: 1. **Une durée** (`DUREE_BOITE_COMPTEE_FRAMES`, 145 frames ≈ 5 s) : c'est
#:    la règle qui décide de la vie réelle d'une boîte verte. 100 px après la
#:    ligne à 0,69 px/frame mesuré : c'est le temps que la personne met à sortir
#:    de la zone utile. Au-delà, elle n'est plus là, la garder n'apprend rien.
#: 2. **Un nombre** (cette constante, 200) : c'est le filet de sécurité. Si la
#:    règle de durée venait à ne pas purger — un `frame_index` qui ne
#:    progresse pas, une pause, un test qui fige le compteur — la liste ne peut
#:    pas croître sans fin. 200 boîtes vertes à l'écran seraient absurdes et
#:    illisibles ; en mémoire cela ne fait que ~200 tuples, quelques dizaines de
#:    kilo-octets, donc ce plafond ne coûte rien et garantit qu'aucune fuite
#:    n'est possible.
#:
#: Pourquoi 200 : à 256 personnes sur 3000 frames, il y a en moyenne 0,085
#: comptage par frame. Une durée de 145 frames donne donc ~12 boîtes vivantes
#: en régime normal. 200 est un ordre de grandeur au-dessus : le plafond ne
#: se déclenche jamais sur une analyse réelle, ce qui est exactement ce qu'on
#: veut d'un filet de sécurité — s'il se déclenchait, il masquerait un bug au
#: lieu de l'empêcher.
TAILLE_MAX_BOITES_COMPTES = 200

#: Durée de vidéo ANALYSÉE au-delà de laquelle un compteur resté à zéro doit
#: être signalé. Exprimé en SECONDES DE VIDÉO et non en frames affichées :
#: l'avertissement parle à l'opérateur qui regarde l'écran, or lui voit du
#: temps, pas un compteur d'images. Trois secondes suffisent à voir un groupe
#: arriver et à ne pas le compter, sans attendre la fin de la vidéo.
SEUIL_AVERTISSEMENT_SENS_S = 3.0

#: Le rappel, mot pour mot. Court, et une seule question : le défaut `sens`
#: vaut +1 alors que le cortège de la vidéo de référence marche vers la
#: gauche, et un zéro « rien ne traverse » ne se distingue pas, à l'écran,
#: d'un zéro « tout le monde est compté dans l'autre sens ».
AVERTISSEMENT_SENS = (
    "0 compté — le sens de la flèche correspond-il au sens de marche "
    "du cortège ?"
)


class FenetrePrincipale(QMainWindow):
    """La fenêtre : vidéo à gauche, compteur et réglages à droite."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Compteur de manifestation — {VERSION}")
        self.resize(1400, 850)

        self.config: Config = Config.defauts()
        self.cap = None
        #: Nombre total de frames annoncées par la vidéo chargée (0 si le
        #: conteneur ne le dit pas) : c'est le maximum de la barre de
        #: progression. Un conteneur muet sur ce chiffre bascule la barre en
        #: mode « occupé » plutôt qu'en pourcentage faux.
        self._nb_frames_video = 0
        self.video_finiment: pathlib.Path | None = None
        self.detecteur = None
        self.tracker = None
        self.compteur = None
        self.ligne: Ligne | None = None

        #: Points posés par l'opérateur, en cours de tracé (0, 1 ou 2).
        self._points_ligne: list[tuple[float, float]] = []
        #: Tracé « armé » : le clic suivant pose le premier point. Sans cette
        #: étape, un clic parasite en plein réglage décalerait la ligne du
        #: premier point venu — invisible à l'écran, faux au compteur.
        self._trace_arme = False
        #: Le traitement tourne (minuterie de traitement active).
        self._en_analyse = False
        #: Une analyse a été démarrée et n'est pas terminée (pause comprise).
        self._session_ouverte = False
        #: L'avertissement « 0 compté » a déjà été montré pour CETTE analyse.
        #: Vrai une fois, remis à faux à chaque nouveau comptage.
        self._avertissement_sens_affiche = False
        #: Nom du modèle effectivement chargé dans `self.detecteur`.
        self._modele_charge = ""
        self._flash = 0
        #: Dernier `FrameResult` traité, en attente d'affichage.
        self._resultat_courant = None
        #: Index du dernier résultat déjà affiché, pour ne pas redessiner.
        self._index_affiche = -1
        #: Position du marqueur de premier point, en pixels image.
        self._apercu_point: tuple[int, int] | None = None
        #: Boîtes VERTES des personnes comptées, encore à l'écran.
        #:
        #: Ce n'est PAS du tracking figé : le track continue d'être suivi après
        #: le franchissement (`_purger_franchis` ne touche pas au tracker), donc
        #: on stocke le `track_id` et on cherche sa position COURANTE dans le
        #: résultat — la boîte verte suit la tête au lieu de rester collée à la
        #: ligne. Quand la personne sort de la bande, elle n'est plus détectée,
        #: le track meurt et la boîte disparaît avec lui.
        #:
        #: Liste de tuples `(frame_comptage, track_id)`, bornée en nombre —
        #: voir `TAILLE_MAX_BOITES_COMPTES`.
        self._boites_comptees: list[tuple[int, int]] = []
        #: Nombre d'événements du compteur déjà transformés en boîtes vertes.
        #: L'affichage peut sauter des frames (vitesse lente) : sans ce
        #: marqueur, on relirait sans cesse les mêmes événements et on
        #: re-créerait des boîtes déjà vues. Il progresse d'un cran par
        #: événement consommé, et repart de zéro quand l'historique est vidé
        #: (nouvelle analyse).
        self._dernier_evenement_affiche = 0
        #: Position de la ligne AU DÉBUT du glissement en cours, ou None.
        #: Le décalage émis par le widget est cumulé depuis l'ancrage : il faut
        #: donc repartir de la position de départ à chaque mouvement, sinon la
        #: ligne dériverait au lieu de suivre le curseur.
        self._ligne_avant_glissement: tuple | None = None

        self._construire()
        self._connecter()
        self.compteur_total = 0
        self.presents = 0
        self.frames = 0
        self.maj_compteurs(0, 0, 0)
        self._maj_boutons()

    # -- Construction ----------------------------------------------------

    def _construire(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        racine = QHBoxLayout(central)
        racine.setContentsMargins(8, 8, 8, 8)
        racine.setSpacing(10)

        # La vidéo ET sa barre de progression forment un bloc : la barre est
        # lisible SOUS l'image, là où l'œil reste pendant l'analyse. La
        # noyer dans la colonne de droite, entre le compteur et les réglages,
        # la rendrait invisible — et l'opérateur repartirait à ne pas savoir
        # où en est le traitement.
        self.video = WidgetVideo()
        self.video.setMinimumWidth(760)
        colonne_video = QVBoxLayout()
        colonne_video.addWidget(self.video, stretch=1)

        self.barre_progression = QProgressBar()
        self.barre_progression.setFormat("%v / %m frames (%p %)")
        self.barre_progression.setVisible(False)
        colonne_video.addWidget(self.barre_progression)

        racine.addLayout(colonne_video, stretch=3)

        colonne = QVBoxLayout()
        racine.addLayout(colonne, stretch=2)

        self.titre_compteur = QLabel("PERSONNES COMPTÉES")
        self.titre_compteur.setObjectName("compteur_titre")
        self.titre_compteur.setAlignment(Qt.AlignmentFlag.AlignCenter)
        colonne.addWidget(self.titre_compteur)

        self.label_compteur = QLabel("0")
        self.label_compteur.setObjectName("compteur")
        self.label_compteur.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Largeur minimale : sans elle, la colonne se resserre quand le chiffre
        # passe de 1 à 2 chiffres et le nombre se décale sous l'œil de
        # l'opérateur. C'est le seul chiffre lu de loin — il doit rester au
        # même endroit.
        self.label_compteur.setMinimumWidth(280)
        colonne.addWidget(self.label_compteur)

        # Le compteur de « présents » et celui de « frames » ont été retirés
        # de l'affichage : deux chiffres que l'opérateur ne regardait jamais,
        # et qui occupaient la place sous le seul chiffre qui compte. Les
        # VALEURS restent calculées — `maj_compteurs` continue de les
        # mémoriser, et le récapitulatif de fin d'analyse en dépendent. On ne
        # retire que l'étiquette.

        # Le périphérique de calcul est affiché EN PERMANENCE, dans son PROPRE
        # label et non dans `label_statut`.
        #
        # `label_statut` est écrasé à chaque changement d'état (vidéo chargée,
        # analyse en cours, pause, erreur) : un indicateur de calcul mis là
        # disparaîtrait précisément quand l'opérateur en a le plus besoin. Un
        # label dédié n'est jamais réécrit, donc l'information est toujours là.
        # Il est placé SOUS le compteur et au-dessus de la barre de statut, et
        # jamais à côté du chiffre : le compteur reste libre pour le résultat.
        self.label_peripherique = QLabel()
        self.label_peripherique.setObjectName("sous_titre")
        self.label_peripherique.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._maj_peripherique()
        colonne.addWidget(self.label_peripherique)

        # L'etat de TORCH lui-meme (version, GPU reellement utilise, ou
        # « absent ») est un label distinct de `label_peripherique`, et pour
        # une raison differente : `label_peripherique` dit ce que le REGLAGE
        # produit, y compris quand torch manque (« CPU uniquement » parce
        # qu'aucun GPU n'est visible). `label_torch` dit si le moteur de
        # calcul est INSTALLE. Les deux se contredisent dans le cas
        # interessant — une machine avec GPU et sans torch — et c'est
        # justement ce cas qu'il faut voir sans ambiguite.
        self.label_torch = QLabel()
        self.label_torch.setObjectName("sous_titre")
        self.label_torch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_torch.setWordWrap(True)
        self._etat_torch = None
        colonne.addWidget(self.label_torch)

        # L'avertissement « 0 compté » vit dans son PROPRE label, comme le
        # périphérique de calcul et pour la même raison : `label_statut` est
        # réécrit à chaque changement d'état (vidéo chargée, pause, erreur,
        # fin de lecture) et l'effacerait au moment précis où l'opérateur en a
        # besoin. Un label dédié, jamais réécrit sauf par cet avertissement
        # lui-même, tient jusqu'à ce que le compteur reparte.
        #
        # Il est SOUS le compteur, dans la colonne de droite, juste au-dessus
        # de la barre de statut : c'est là que l'œil va chercher pourquoi le
        # chiffre ne bouge pas. Il est masqué (et non « vide ») quand il n'a
        # rien à dire — un label vide garde sa place dans la colonne et ferait
        # sauter la mise en page.
        self.label_avertissement = QLabel()
        self.label_avertissement.setObjectName("avertissement")
        self.label_avertissement.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_avertissement.setWordWrap(True)
        self.label_avertissement.setVisible(False)
        colonne.addWidget(self.label_avertissement)

        self.label_statut = QLabel("Charge une vidéo pour commencer.")
        self.label_statut.setObjectName("sous_titre")
        self.label_statut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_statut.setWordWrap(True)
        self.label_statut.setMinimumHeight(56)
        colonne.addWidget(self.label_statut)

        # Le récapitulatif de fin d'analyse est une SECTION de la colonne de
        # droite, pas un onglet. La fenêtre n'a pas d'organisation par onglets
        # — elle a une colonne d'affichage et une colonne de réglages — et un
        # QTabWidget aurait imposé de choisir à chaque instant entre la vidéo
        # et son bilan : l'opérateur ne verrait plus la scène en lisant le
        # décompte. Une section masquée jusqu'à la fin de l'analyse garde les
        # deux visibles en même temps, et ne coûte aucune place avant qu'il y
        # ait quelque chose à montrer.
        self.recap = PanneauRecap()
        self.recap.setVisible(False)
        colonne.addWidget(self.recap, stretch=1)

        self.panneau = PanneauReglages(self.config)
        # Le panneau est dans une zone défilante, et la zone reste nécessaire
        # même aide repliée. Mesuré : panneau à 984 px aide masquée, 2138 px
        # aide dépliée, pour 618 px disponibles dans une fenêtre de 850 — la
        # vidéo, le compteur et la barre de boutons mangent déjà 232 px. Sans
        # zone, Qt comprimerait le panneau et le bouton « Lancer » deviendrait
        # inatteignable : l'analyse ne serait plus lançable du tout.
        #
        # Ce que la tâche 18 a changé n'est donc pas la zone mais son contenu :
        # les encarts d'aide sont repliés derrière le bouton « Aide » du panneau.
        # On a gagné 1154 px de défilement sans retirer un mot d'explication —
        # elles sont toutes là, à deux endroits : au survol du champ, et d'un clic
        # dans le panneau quand on veut les lire.
        zone = QScrollArea()
        zone.setWidget(self.panneau)
        zone.setWidgetResizable(True)
        zone.setFrameShape(QScrollArea.Shape.NoFrame)
        colonne.addWidget(zone, stretch=1)

        barre = QHBoxLayout()
        self.btn_video = QPushButton("Charger la vidéo")
        self.btn_ligne = QPushButton("Tracer la ligne")
        self.btn_ligne.setToolTip(
            "Clique ensuite deux points sur l'image, aux deux extrémités de "
            "la limite que tu veux compter.\n"
            "Le sens se lit ensuite dans les réglages, avec des libellés "
            "explicites : « Gauche → droite » ou « Droite → gauche » sur une "
            "ligne verticale, « Haut → bas » ou « Bas → haut » sur une ligne "
            "horizontale."
        )
        self.btn_lancer = QPushButton("Lancer")
        self.btn_lancer.setObjectName("primaire")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        for b in (
            self.btn_video,
            self.btn_ligne,
            self.btn_lancer,
            self.btn_pause,
            self.btn_stop,
        ):
            barre.addWidget(b)
        colonne.addLayout(barre)

        # Bouton de soutien, discret, tout en bas sous la barre de boutons.
        # Ce n'est pas un réglage : il ouvre simplement la page de don dans le
        # navigateur. Pas de JS, pas de widget web — un lien natif, qui marche
        # hors ligne et ne ralentit rien.
        self.btn_soutenir = QPushButton("☕ Buy me a coffee")
        self.btn_soutenir.setObjectName("soutenir")
        self.btn_soutenir.setToolTip(
            "Soutenir le développement du compteur — ouvre la page de don "
            "dans ton navigateur."
        )
        colonne.addWidget(self.btn_soutenir, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Traitement : intervalle 0, donc « aussi vite que possible ». C'est
        # le GPU qui fixe la cadence de cette minuterie, jamais un réglage
        # utilisateur : l'analyse va aussi vite que la machine le permet.
        self._timer_traitement = QTimer(self)
        self._timer_traitement.setInterval(0)
        # Affichage : ~15 i/s, PAS la cadence de traitement. Rendre la frame
        # (voile, boîtes, conversion Qt, mise à l'échelle) coûte en CPU autant
        # que l'inférence en GPU, et les deux tournent sur le même thread :
        # afficher chaque frame traitée plafonnerait le débit de comptage.
        # L'opérateur lit un compteur, pas un film — 15 i/s suffisent largement.
        # Le rattrapage des événements sautés est déjà prévu (voir
        # `_dernier_evenement_affiche`, écrit pour des frames non affichées),
        # et le comptage, lui, reste à la cadence du timer de traitement.
        self._timer_affichage = QTimer(self)
        self._timer_affichage.setInterval(1000 // CADENCE_AFFICHAGE_IPS)

    def _connecter(self) -> None:
        self.btn_video.clicked.connect(self._on_charger_video)
        self.btn_ligne.clicked.connect(self._on_tracer_ligne)
        self.btn_lancer.clicked.connect(self.lancer)
        self.btn_pause.clicked.connect(self._basculer_pause)
        self.btn_stop.clicked.connect(self.arreter)
        self.btn_soutenir.clicked.connect(self._on_soutenir)
        self.video.clic.connect(self._on_clic_video)
        self.video.deplacement.connect(self._on_deplacement_video)
        self.video.deplacement_fini.connect(self._on_deplacement_fini)
        self.panneau.config_modifiee.connect(self._sur_config)
        self._timer_traitement.timeout.connect(self._tick)
        self._timer_affichage.timeout.connect(self._afficher)

        # Etat de torch initial, AVANT tout affichage. `preparer()` (v2)
        # rappelle `definir_etat_torch` juste apres la construction ; depuis
        # les sources, ou dans un test, c'est cette ligne qui donne un
        # indicateur honnete au lieu d'un label vide.
        self.definir_etat_torch(None)

    def _maj_peripherique(self, config: Config | None = None) -> None:
        """Recalcule et affiche « Calcul : CUDA — … » ou « Calcul : CPU … ».

        L'opérateur a cherché à confirmer le GPU avec le gestionnaire de
        performances sans y arriver ; la réponse doit donc être à l'écran, en
        permanence, et nommer la carte quand elle existe.

        L'avertissement éventuel (GPU demandé mais absent, CPU forcé avec un
        GPU présent) part dans le JOURNAL et non dans le label : un encart de
        deux lignes ici remplacerait le compteur, et l'opérateur n'a rien à
        faire de l'information une fois qu'il l'a lue — sauf changer le
        réglage, ce que le menu « Périphérique de calcul » permet déjà.
        """
        config = config or getattr(self, "config", None) or Config.defauts()
        mode = mode_peripherique(getattr(config, "peripherique", "auto"))
        choix = peripherique_effectif(mode)
        if choix.avertissement:
            log.warning("%s", choix.avertissement)
        self.label_peripherique.setText(f"Calcul : {choix.libelle}")
        self._peripherique_effectif = choix

    def definir_etat_torch(self, etat) -> None:
        """Enregistre l'etat de torch et met l'indicateur a jour.

        Appele par `main.py` avec le retour de `interface.demarrage.preparer`.
        Un argument absent ou faux reste acceptable : la fenetre doit pouvoir
        etre construite seule, comme le font les 481 tests existants, et dans
        ce cas l'indicateur retombe sur une detection directe de torch.

        `label_torch` est un label DEDIE, pas une ligne de plus dans
        `label_statut`. La meme raison que pour `label_peripherique` :
        `label_statut` est reecrit a chaque changement d'etat (video chargee,
        analyse, pause, erreur) et effacerait precisement l'information que
        l'operateur cherche au moment ou il la cherche. Un label jamais
        reecrit tient jusqu'a la fin de session.
        """
        from interface.demarrage import EtatTorch, texte_indicateur

        if etat is None:
            # Fenetre construite seule (tests, ou appel sans `preparer`) :
            # on deduit l'etat de ce qui est reellement importable.
            version = ""
            try:
                import torch

                version = str(getattr(torch, "__version__", ""))
            except ImportError:
                pass
            from interface.demarrage import _cuda_reelle

            etat = EtatTorch(version=version, cuda=_cuda_reelle(), gpu="")
        self._etat_torch = etat
        self.label_torch.setText(texte_indicateur(etat))
        if etat.probleme:
            log.warning("%s", etat.probleme)

    def etat_torch(self):
        """Etat de torch affiche, pour les tests et le recapitulatif."""
        return getattr(self, "_etat_torch", None)

    def device_effectif(self) -> str:
        """Périphérique réellement utilisé, au format attendu par torch.

        Exposé pour que les tests vérifient leTexte affiché ET la valeur qui
        part dans `charger_modele` : les deux doivent parler du même matériel.
        """
        return self._peripherique_effectif.device

    # -- Boîtes de dialogue ----------------------------------------------

    def _signaler(self, titre: str, message: str, grave: bool = False) -> None:
        """Signale un problème à l'opérateur, sans jamais bloquer un test.

        Le message va toujours dans la barre de statut et le journal. La boîte
        modale n'apparaît QUE si la fenêtre est visible : sous `offscreen`, ou
        dans une fenêtre jamais affichée, un « OK » modal serait un blocage
        définitif.
        """
        (log.error if grave else log.warning)("%s : %s", titre, message)
        self.label_statut.setText(f"{titre} — {message}")
        if not self.isVisible():
            return
        boite = QMessageBox.critical if grave else QMessageBox.warning
        boite(self, titre, message)

    def _on_soutenir(self) -> None:
        """Ouvre la page de don dans le navigateur par défaut.

        L'échec d'ouverture (navigateur absent, aucune application liée au
        protocole http) n'est pas une erreur grave : il est journalisé et
        signalé dans la barre de statut, sans boîte modale qui interromprait
        l'analyse.
        """
        if QDesktopServices.openUrl(QUrl(LIEN_SOUTIEN)):
            self.label_statut.setText(
                "Ouverture de la page de soutien dans le navigateur…"
            )
        else:
            log.warning("impossible d'ouvrir le navigateur vers %s", LIEN_SOUTIEN)
            self.label_statut.setText(
                "Impossible d'ouvrir le navigateur — copie le lien : "
                f"{LIEN_SOUTIEN}"
            )

    # -- Actions : la vidéo ----------------------------------------------

    def _on_charger_video(self) -> None:
        chemin, _ = QFileDialog.getOpenFileName(
            self, "Choisir une vidéo", "", FILTRES_VIDEOS
        )
        if chemin:
            self.charger_video(chemin)

    def charger_video(self, chemin: str) -> bool:
        """Ouvre ``chemin`` et affiche sa première frame. Faux si illisible."""
        p = pathlib.Path(chemin)
        if not p.exists():
            self._signaler("Vidéo introuvable", f"Fichier absent :\n{chemin}")
            return False

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            cap.release()
            self._signaler("Vidéo illisible", f"Impossible d'ouvrir :\n{chemin}")
            return False

        self.arreter()
        self._liberer_capture()
        self.cap = cap
        self.video_finiment = p
        self._reset_analyse()
        self._afficher_premiere_frame()

        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = self._fps()
        # La barre de progression est calée sur le total ANNONCÉ par le
        # conteneur. Total inconnu (0) : plage vide (min == max), le mode
        # « occupé » de QProgressBar — mieux qu'un pourcentage faux.
        self._nb_frames_video = nb_frames
        if nb_frames > 0:
            self.barre_progression.setRange(0, nb_frames)
        else:
            self.barre_progression.setRange(0, 0)
        self.barre_progression.setValue(0)
        analyse = self._resolution_analyse_pour(cap)
        largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.label_statut.setText(
            f"{p.name} — {largeur}×{hauteur}, {nb_frames} frames à {fps:.0f} i/s.\n"
            f"Analysée à {analyse} px. Trace la ligne, puis lance."
        )
        self._maj_boutons()
        return True

    def _resolution_analyse_pour(self, cap) -> int:
        """Résolution d'analyse retenue pour la vidéo ouverte, et appliquée.

        La résolution est LUE sur la `VideoCapture`
        (`CAP_PROP_FRAME_WIDTH`/`HEIGHT`) — c'est ce que le conteneur déclare,
        et cela ne coûte rien. Décoder une frame pour la mesurer serait plus
        lent d'un facteur cent et donnerait le même chiffre.

        Le réglage Manuel court-circuite tout : l'opérateur a dit qu'il savait
        ce qu'il faisait, et son choix n'a pas à être corrigé par la source.
        """
        largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if mode_resolution(self.config.resolution_analyse) == MODE_MANUEL:
            return self.config.taille_entree
        taille = self.panneau.definir_resolution_video(largeur, hauteur)
        # Le panneau a recalculé pour son affichage ; `self.config` doit porter
        # la MÊME valeur, sinon la fenêtre annoncerait une résolution et le
        # détecteur en utiliserait une autre.
        self.config = self.panneau.lire()
        log.info(
            "vidéo %dx%d analysée à %d px (mode %s)",
            largeur,
            hauteur,
            taille,
            mode_resolution(self.config.resolution_analyse),
        )
        return taille

    def _reset_analyse(self) -> None:
        """Repart d'un état neuf pour une nouvelle vidéo.

        La ligne tracée est EFFACÉE : ses deux extrémités sont des pixels de
        l'image qui vient d'être remplacée. La conserver donnerait une ligne
        numériquement identique et géométriquement fausse sur la nouvelle
        scène — un défaut qu'aucune fenêtre ne permet de voir.
        """
        self.config = self.panneau.lire()
        self.detecteur = None
        self.tracker = None
        self.compteur = None
        self._modele_charge = ""
        self.ligne = None
        self._points_ligne = []
        self._trace_arme = False
        self._apercu_point = None
        self._resultat_courant = None
        self._index_affiche = -1
        self._flash = 0
        # Les boîtes vertes de l'analyse précédente appartiennent à la vidéo
        # précédente : les laisser afficher sur la nouvelle donnerait à
        # l'opérateur des personnes qui n'existent pas dans cette scène.
        self._boites_comptees.clear()
        # Le marqueur d'événements consommés repart avec : l'historique du
        # compteur est celui de la vidéo précédente, et il sera vidé avec elle.
        self._dernier_evenement_affiche = 0
        # Nouvelle vidéo : l'ancienne ligne était verrouillée pour l'analyse
        # qui vient de s'arrêter. Sans ce déverrouillage, l'opérateur ne
        # pourrait plus la déplacer, et le bouton « Tracer la ligne » échouerait
        # silencieusement.
        if self.ligne is not None:
            self.ligne.deverrouiller()
        self.video.definir_zone_deplacement(None)
        # Le récapitulatif concernait l'analyse précédente : il n'a plus de
        # rapport avec cette vidéo-ci et disparaît avec elle.
        self.recap.effacer()
        self.recap.setVisible(False)
        # La barre de progression appartient à la vidéo précédente : elle est
        # masquée et repartira de zéro au prochain « Lancer ».
        self.barre_progression.setValue(0)
        self.barre_progression.setVisible(False)
        self._nb_frames_video = 0
        self.maj_compteurs(0, 0, 0)

    def _afficher_premiere_frame(self) -> None:
        """Affiche la première frame sans la traiter.

        L'opérateur doit voir la scène avant de poser sa ligne ; le compteur,
        lui, reste à zéro.
        """
        if self.cap is None:
            return
        ok, img = self.cap.read()
        if not ok:
            self._signaler("Vidéo vide", "Aucune image lisible dans ce fichier.")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video.definir_image(img)

    def _fps(self) -> float:
        """Cadence annoncée par la vidéo, avec repli si elle est aberrante.

        Un `CAP_PROP_FPS` à 0 ou négatif — conteneur mal écrit, certains codecs
        d'observation — donnerait une division par zéro dans le calcul du
        timestamp, et donc un bilan faux ou un plantage.
        """
        if self.cap is None:
            return FPS_REPLI
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (1.0 <= fps <= 240.0):
            return FPS_REPLI
        return fps

    # -- Actions : la ligne ----------------------------------------------

    def _on_tracer_ligne(self) -> None:
        """Arme le tracé : le prochain clic pose le premier point."""
        self._points_ligne = []
        self.ligne = None
        self._trace_arme = True
        self._apercu_point = None
        self._rafraichir_affichage()
        self._maj_boutons()
        self.label_statut.setText(
            "Clique deux points sur l'image, en haut et en bas de la ligne "
            "que tu veux compter."
        )

    def _on_clic_video(self, x: int, y: int) -> None:
        """Un clic de l'opérateur, EN COORDONNÉES IMAGE (déjà converties).

        Rien n'est fait si le tracé n'est pas armé : c'est ce qui distingue un
        clic volontaire d'un clicRaté en plein réglage.
        """
        if not self._trace_arme or self._en_analyse:
            return

        if not self._points_ligne:
            self._points_ligne.append((float(x), float(y)))
            self._apercu_point = (x, y)
            self._rafraichir_affichage()
            self.label_statut.setText(
                f"Premier point en ({x}, {y}). Clique le second."
            )
            return

        self._points_ligne.append((float(x), float(y)))
        self._apercu_point = None
        self._trace_arme = False
        self.definir_ligne(*self._points_ligne)

    def _zone_deplacement(self) -> "tuple[int, int, int, int] | None":
        """Rectangle de la bande, en pixels image, si le glissement a lieu.

        `None` — donc pas de zone déplaçable — pendant une analyse : la ligne
        est verrouillée, `Ligne.deplacer` refuserait le déplacement, et le
        glisser doit être INOPÉRANT, pas planté. `None` aussi sans ligne.
        """
        if self.ligne is None or self._en_analyse or self.ligne.verrouillee:
            return None
        # Les dimensions sont celles de l'image AFFICHÉE, pas d'une lecture
        # vidéo : c'est elle que l'opérateur clique, donc elle seule donne un
        # rectangle dans lequel le clic tombera vraiment.
        dimensions = self.video.dimensions_image()
        if dimensions is None:
            return None
        largeur, hauteur = dimensions
        return self.ligne.rect_bande_detection(largeur, hauteur)

    def _rafraichir_zone_deplacement(self) -> None:
        """Redonne au widget la zone déplaçable courante.

        Appelée après chaque déplacement et après chaque changement de ligne :
        la zone est DÉRIVÉE de la ligne, elle doit donc suivre le déplacement
        au lieu de rester figée sur la bande d'origine.
        """
        self.video.definir_zone_deplacement(self._zone_deplacement())

    def _on_deplacement_video(self, dx: int, dy: int) -> None:
        """Un glissement : la ligne ET sa bande bougent de (dx, dy) pixels.

        Le décalage est appliqué à `p1` et `p2` PAR LA MÊME valeur : la bande
        n'est pas une coordonnée stockée mais une grandeur dérivée
        (`Ligne.rect_bande_detection`), donc elle suit la ligne sans qu'aucun
        second état puisse diverger. C'est aussi pourquoi on n'arme pas de
        copie de bande ici : il n'y en a pas.
        """
        if self.ligne is None or self._en_analyse or self.ligne.verrouillee:
            return
        if self._ligne_avant_glissement is None:
            self._ligne_avant_glissement = (self.ligne.p1, self.ligne.p2)
        (ax, ay), (bx, by) = self._ligne_avant_glissement
        self.ligne.deplacer((ax + dx, ay + dy), (bx + dx, by + dy))
        self.config.ligne = (
            self.ligne.p1[0],
            self.ligne.p1[1],
            self.ligne.p2[0],
            self.ligne.p2[1],
        )
        self.panneau.definir_ligne(self.config.ligne)
        self._rafraichir_affichage()
        self._rafraichir_zone_deplacement()

    def _on_deplacement_fini(self) -> None:
        """Fin du glissement : l'ancrage est oublié, la zone est rafraîchie."""
        self._ligne_avant_glissement = None
        self._maj_boutons()
        self._rafraichir_zone_deplacement()

    def definir_ligne(self, p1, p2) -> bool:
        """Fixe la ligne de comptage à partir de deux points image.

        `Ligne` valide : points confondus, `sens` hors {+1, -1}, épaisseur
        nulle. Chaque cas est un réglage que l'opérateur peut se tromper, donc
        un message — pas une exception qui ferme la fenêtre au milieu d'une
        manifestation.
        """
        try:
            self.ligne = Ligne(
                p1,
                p2,
                epaisseur=self.config.epaisseur_bande,
                sens=self.config.sens,
                hysteresis=self.config.frames_hysteresis,
                bande_avant_px=BANDE_AVANT_PX,
                bande_apres_px=BANDE_APRES_PX,
            )
        except ValueError as exc:
            self._points_ligne = []
            self._apercu_point = None
            self._maj_boutons()
            self._signaler("Ligne invalide", str(exc))
            return False

        # La ligne tracée est pushed dans la Config : c'est elle que
        # `analyser_video` relira pour une analyse hors interface.
        self.config.ligne = (
            self.ligne.p1[0],
            self.ligne.p1[1],
            self.ligne.p2[0],
            self.ligne.p2[1],
        )
        self.panneau.definir_ligne(self.config.ligne)
        self._rafraichir_affichage()
        self._maj_boutons()
        # La bande devient immédiatement déplaçable : l'opérateur vient de la
        # poser et veut souvent l'ajuster d'un cheveu avant de lancer.
        self._rafraichir_zone_deplacement()
        # Le sens compté est écrit EN CLAIR dans la barre de statut, et pas
        # seulement dessiné par la flèche verte. L'opérateur voit ainsi, avant
        # de lancer, dans quel sens la vidéo va être comptée — c'était
        # exactement l'information manquante qui lui coûtait de tester les
        # deux sens à chaque fois.
        self.label_statut.setText(
            f"Ligne {self.ligne.orientation} posée de "
            f"({self.ligne.p1[0]:.0f}, {self.ligne.p1[1]:.0f}) à "
            f"({self.ligne.p2[0]:.0f}, {self.ligne.p2[1]:.0f}).\n"
            f"Sens compté : {self._sens_en_clair()}. "
            "Si ton cortège va dans l'autre sens, change-le dans les réglages "
            "avant de lancer."
        )
        return True

    def _sens_en_clair(self) -> str:
        """Le sens compté, nommé à l'écran, d'après la ligne posée.

        Délègue à `Ligne.libelle_sens` : la barre de statut et la flèche verte
        ne peuvent donc pas nommer deux choses différentes.
        """
        if self.ligne is None:
            return LIBELLES_SENS[orientation_par_defaut_sens()][
                0 if self.config.sens == 1 else 1
            ]
        return self.ligne.libelle_sens(self.ligne.sens)

    def _rafraichir_affichage(self) -> None:
        """Redessine la frame courante avec l'aperçu de la ligne en cours.

        Sans cela, l'opérateur clique deux points à l'aveugle et ne voit sa
        ligne qu'en lançant le comptage — c'est-à-dire trop tard pour la
        corriger.
        """
        img = self._image_courante()
        if img is None:
            return
        sortie = img
        if self.ligne is not None:
            # Le voile passe ici aussi, et AVANT la ligne : l'opérateur doit voir
            # la colonne qu'il est en train de placer, pas seulement une fois
            # l'analyse lancée. C'est le seul moment où il peut encore corriger
            # son geste.
            sortie = self.ligne.voiler(sortie)
            sortie = self.ligne.dessiner(sortie)
        if self._apercu_point is not None:
            sortie = _marquer_point(sortie, self._apercu_point)
        self.video.definir_image(sortie)

    def _image_courante(self) -> "np.ndarray | None":  # noqa: F821
        """Dernière frame traitée, ou la première frame lue au chargement."""
        if self._resultat_courant is not None:
            return self._resultat_courant.image
        if self.cap is None:
            return None
        position = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        ok, img = self.cap.read()
        if not ok:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        return img

    # -- Actions : les réglages ------------------------------------------

    def _sur_config(self, config: Config) -> None:
        """Le panneau a changé : on propage sans jamais casser la fenêtre.

        Deux cas distincts :

        - un réglage ordinaire (seuil, bande, hystérésis) se répercute tel quel
          sur le détecteur, le tracker et la ligne déjà tracée, qui garde ses
          DEUX POINTS — c'est la géométrie que l'opérateur a posée qu'il ne
          faut pas perdre ;
        - un MODÈLE différent, lui, rend les poids chargés caducs : le
          détecteur est lâché. Le conserver compterait avec l'ancien modèle
          pendant que l'écran annonce le nouveau.
        """
        # Le périphérique se met à jour AVANT l'affectation de `self.config` :
        # la comparaison ci-dessous porte sur l'ANCIEN réglage, donc elle doit
        # être faite avant que la référence ne soit écrasée. Écraser d'abord
        # rendrait tout changement invisible — et l'indicateur afficherait le
        # périphérique précédent, ce qui est le mensonge exact que cet
        # indicateur sert à éviter.
        peripherique_change = (
            mode_peripherique(getattr(config, "peripherique", "auto"))
            != mode_peripherique(getattr(self.config, "peripherique", "auto"))
        )
        if peripherique_change:
            log.info("périphérique de calcul : %s", getattr(config, "peripherique", "auto"))
            self._maj_peripherique(config)

        self.config = config
        if self.ligne is not None:
            try:
                self.ligne = Ligne(
                    self.ligne.p1,
                    self.ligne.p2,
                    epaisseur=config.epaisseur_bande,
                    sens=config.sens,
                    hysteresis=config.frames_hysteresis,
                    bande_avant_px=BANDE_AVANT_PX,
                bande_apres_px=BANDE_APRES_PX,
                )
            except ValueError as exc:
                # Réglage devenu impossible (épaisseur 0, sens nul) : on garde
                # l'ancienne ligne plutôt que de laisser le moteur sans ligne,
                # qui ne compterait plus personne.
                log.warning("réglage de ligne refusé : %s", exc)
        if self.compteur is not None:
            self.compteur.config = config
        # Un changement de périphérique lâche le détecteur comme un changement
        # de modèle : le device entre dans `charger_modele`.
        if self.detecteur is not None:
            if config.modele and config.modele != self._modele_charge:
                log.info("modèle changé (%s -> %s)", self._modele_charge, config.modele)
                self.detecteur = None
                self.tracker = None
                self._modele_charge = ""
                if self._en_analyse:
                    self.label_statut.setText(
                        "Modèle changé : l'analyse en cours va s'arrêter. "
                        "Relance pour l'appliquer."
                    )
                    self.arreter()
            elif peripherique_change:
                # Un modèle déjà monté reste sur son périphérique d'origine :
                # `Detecteur` ne déplace pas le modèle à chaud. Le lâcher est
                # donc obligatoire, sinon l'écran annoncerait « CPU » pendant
                # que l'analyse tourne encore sur le GPU — l'inverse exact du
                # problème que cet indicateur sert à résoudre.
                log.info("périphérique changé : rechargement du modèle nécessaire.")
                self.detecteur = None
                self.tracker = None
                self._modele_charge = ""
                if self._en_analyse:
                    self.label_statut.setText(
                        "Périphérique de calcul changé : l'analyse en cours va "
                        "s'arrêter. Relance pour l'appliquer."
                    )
                    self.arreter()
            else:
                self.detecteur.config = config
        if self.tracker is not None:
            self.tracker.config = config

    # -- Actions : la lecture --------------------------------------------

    def lancer(self) -> None:
        """Démarre l'analyse. Ne charge le modèle qu'ici, jamais au démarrage."""
        if self.cap is None or self.ligne is None or self._en_analyse:
            return

        # Un réglage peut avoir bougé depuis le dernier « Lancer » : on
        # repart de l'état du panneau, jamais d'une copie périmée.
        self.config = self.panneau.lire()
        self.config.ligne = self._ligne_en_tuple()
        self._reconstruire_ligne_depuis_config()

        try:
            if self.detecteur is None:
                log.info("chargement du modèle %s", self.config.modele)
                self.detecteur = Detecteur(
                    charger_modele(self.config.modele, self.config.peripherique),
                    self.config,
                    # Demi-précision uniquement sur un GPU réel : sur CPU, la
                    # valeur ne servirait à rien et sur les faux modèles des
                    # tests elle changerait la forme de l'appel.
                    half=peripherique_effectif(self.config.peripherique).cuda,
                )
                self._modele_charge = self.config.modele
            if self.tracker is None:
                self.tracker = Tracker(self.config)
            self.compteur = Compteur(self.config, self.detecteur, self.tracker)
        except Exception as exc:  # noqa: BLE001 — un échec ici doit rester récupérable
            # Ni la vidéo ni la ligne ne sont touchées : l'opérateur corrige le
            # modèle et relance, sans retracer.
            self._signaler(
                "Démarrage impossible",
                f"{exc}\n\nLa vidéo et la ligne sont conservées.",
                grave=True,
            )
            self._maj_boutons()
            return

        # La ligne et sa bande de détection sont VERROUILLÉES dès le lancement.
        # Pendant l'analyse, ni l'une ni l'autre ne bougent : sinon le
        # décompte afficherait deux zones différentes dans la même vidéo, et
        # l'opérateur verrait sa colonne glisser sous ses yeux sans pouvoir croire
        # le chiffre. Le verrou est levé par `arreter()` et `_reset_analyse`.
        self.ligne.verrouiller()
        # La zone déplaçable est RETIRÉE, pas seulement ignorée : le clic dans
        # la bande redevient inerte au lieu de partir sur un déplacement que
        # `Ligne.deplacer` refuse.
        self._ligne_avant_glissement = None
        self._rafraichir_zone_deplacement()
        self.compteur.ajuster_ligne(self.ligne)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.compteur_total = 0
        self.presents = 0
        self.frames = 0
        self._resultat_courant = None
        self._index_affiche = -1
        self._flash = 0
        self._boites_comptees.clear()
        self._dernier_evenement_affiche = 0
        # « Une fois par analyse » : le compteur repart de zéro ici, donc le
        # rappel doit pouvoir se redéployer. La réouverture de session après
        # une pause NE le réarme pas : sans cela, chaque reprise en pause le
        # ferait clignoter, alors que l'opérateur n'a pas relancé d'analyse.
        self._avertissement_sens_affiche = False
        # Le récapitulatif de l'analyse PRÉCÉDENTE disparaît ici. Le laisser en
        # place pendant qu'un nouveau décompte démarre ferait lire l'ancien
        # total et l'ancienne courbe comme s'ils étaient en cours.
        self.recap.effacer()
        self.recap.setVisible(False)
        self._ouvrir_session()
        # La barre de progression vit de « Lancer » au prochain chargement de
        # vidéo : en fin d'analyse elle reste visible à sa dernière valeur —
        # l'opérateur voit d'un coup d'œil que la scène est allée au bout.
        self.barre_progression.setValue(0)
        self.barre_progression.setVisible(True)
        self._maj_boutons()
        self.label_statut.setText("Analyse en cours…")
        self._timer_traitement.start()
        self._timer_affichage.start()

    def _ligne_en_tuple(self) -> tuple[float, float, float, float] | None:
        if self.ligne is None:
            return None
        return (self.ligne.p1[0], self.ligne.p1[1], self.ligne.p2[0], self.ligne.p2[1])

    def _reconstruire_ligne_depuis_config(self) -> None:
        """Réapplique les réglages du panneau à la ligne existante."""
        try:
            self.ligne = Ligne(
                self.ligne.p1,
                self.ligne.p2,
                epaisseur=self.config.epaisseur_bande,
                sens=self.config.sens,
                hysteresis=self.config.frames_hysteresis,
                bande_avant_px=BANDE_AVANT_PX,
                bande_apres_px=BANDE_APRES_PX,
            )
        except ValueError as exc:
            self._signaler("Ligne invalide", str(exc), grave=True)
            self.ligne = None

    def _ouvrir_session(self) -> None:
        """Passe en « analyse lancée » : traitement actif, commandes actives.

        Ce passage est isolé parce que l'état de la lecture est à DEUX
        drapeaux — `_en_analyse` (le timer tourne) et `_session_ouverte` (une
        analyse a été démarrée et n'est pas finie). Les laisser se remplir
        séparément à chaque site d'appel est exactement ce qui les fait
        diverger ; ils ne sont donc écrits que d'ici, de `arreter` et de
        `_reset_analyse`.
        """
        self._en_analyse = True
        self._session_ouverte = True
        self.btn_pause.setText("Pause")

    def _basculer_pause(self) -> None:
        if self._en_analyse:
            self._timer_traitement.stop()
            self._en_analyse = False
            self.btn_pause.setText("Reprendre")
            self.label_statut.setText("En pause.")
        elif self._session_ouverte and self.compteur is not None:
            self._timer_traitement.start()
            self._ouvrir_session()
            self.label_statut.setText("Analyse en cours…")
        self._maj_boutons()

    def arreter(self) -> None:
        """Ferme la session d'analyse et rend la main à l'opérateur.

        Le récapitulatif s'affiche ici, et pas seulement en fin de lecture : un
        opérateur qui coupe au bout de trente secondes — parce que la scène a
        changé, ou qu'il a assez de matière — doit retrouver ses chiffres sans
        avoir à laisser la vidéo tourner jusqu'au bout. Il s'affiche donc sur
        TOUT arrêt d'une analyse engagée, qui que soit l'appelant.

        Le drapeau est lu AVANT la remise à zéro : `arreter()` est aussi
        appelé au chargement d'une vidéo et lors d'un changement de réglage,
        cas où aucune analyse n'a eu lieu et où il n'y a rien à résumer.
        """
        analyse_engagee = self._session_ouverte
        self._timer_traitement.stop()
        self._timer_affichage.stop()
        # Fin d'analyse : la ligne redevient déplaçable, bande comprise.
        if self.ligne is not None:
            self.ligne.deverrouiller()
        self._rafraichir_zone_deplacement()
        self._en_analyse = False
        self._session_ouverte = False
        self.btn_pause.setText("Pause")
        self._maj_boutons()
        if analyse_engagee:
            self._afficher_recap()

    def _afficher_recap(self) -> None:
        """Remplit et DÉVOILE le récapitulatif de fin d'analyse.

        Un `Resultat` impossible à construire ne fait pas tomber la fenêtre :
        un compteur de terrain ne se ferme pas parce qu'un résumé n'a pas pu
        s'écrire. Le panneau reste alors masqué, et le journal dit pourquoi.
        """
        if self.compteur is None:
            return
        try:
            resultat = self.compteur.resultat(
                self.config.modele, self.frames, self.frames / self._fps()
            )
        except TypeError:
            # Un compteur de test peut ne pas accepter la durée vidéo : le
            # récapitulatif est un bonus d'affichage, pas une condition de
            # fonctionnement de la fenêtre.
            log.warning("récapitulatif indisponible : signature inattendue")
            return
        except Exception as exc:  # noqa: BLE001 — l'affichage ne doit rien casser
            log.warning("récapitulatif indisponible : %s", exc)
            return
        self.recap.afficher(resultat)
        self.recap.setVisible(True)

    def _maj_boutons(self) -> None:
        """Une seule source de vérité pour l'activation des boutons.

        Passé par des `setEnabled` dispersés, cet état divergeait : le bouton
        « Lancer » restait actif après le chargement d'une vidéo, et « Pause »
        cliquable sans session ouverte.
        """
        video = self.cap is not None
        pret = video and self.ligne is not None
        self.btn_video.setEnabled(not self._en_analyse)
        self.btn_ligne.setEnabled(video and not self._en_analyse)
        self.btn_lancer.setEnabled(pret and not self._en_analyse)
        self.btn_pause.setEnabled(self._session_ouverte)
        self.btn_stop.setEnabled(self._session_ouverte)

    # -- Boucle de lecture ------------------------------------------------

    def _tick(self) -> None:
        """Traite UN LOT de frames : détection groupée, comptage séquentiel.

        La séparation d'avec `_afficher` reste entière — et prend encore plus
        de sens ici : un lot de 4 frames produit 4 résultats, seuls le
        dernier et les compteurs sont poussés à l'écran. Les événements des
        frames intermédiaires ne sont pas perdus : `_dernier_evenement_affiche`
        les rattrape depuis l'historique du compteur, écrit précisément pour
        des frames traitées mais non affichées.
        """
        if self.cap is None or self.compteur is None:
            self._timer_traitement.stop()
            return

        fps = self._fps()
        lot = []
        while len(lot) < TAILLE_LOT_DETECTION:
            ok, img = self.cap.read()
            if not ok:
                break
            lot.append((img, self.frames, self.frames / fps))
            self.frames += 1

        if not lot:
            self.arreter()
            self._afficher_bilan()
            return

        # Repli : un compteur de test n'a pas forcément `traiter_lot` — même
        # logique que le repli côté détecteur dans `Compteur.traiter_lot`.
        if hasattr(self.compteur, "traiter_lot"):
            resultats = self.compteur.traiter_lot(lot)
        else:
            resultats = [
                self.compteur.traiter_frame(img, index, ts) for img, index, ts in lot
            ]
        self._resultat_courant = resultats[-1]

    def _afficher(self) -> None:
        """Pousse la dernière frame traitée vers l'écran et les compteurs.

        Le compteur et l'image sont mis à jour ENSEMBLE : l'opérateur doit voir
        le chiffre correspondant à la scène qu'il regarde, pas un chiffre qui
        a déjà pris deux secondes d'avance.
        """
        resultat = self._resultat_courant
        if resultat is None or resultat.frame_index == self._index_affiche:
            return
        self._index_affiche = resultat.frame_index

        if resultat.evenements:
            self._flash = FLASH_FRAMES
        flash = self._flash > 0
        if self._flash > 0:
            self._flash -= 1

        # La méthode met à jour `self._boites_comptees` (état interne) ET
        # retourne les positions courantes à dessiner. L'état interne reste
        # donc intact ; c'est le retour qui part dans l'overlay.
        boites = self._boites_comptees_a_afficher(resultat)
        self.maj_compteurs(resultat.total, resultat.presents, self.frames)
        self.video.definir_image(
            dessiner(
                resultat.image,
                resultat,
                self.ligne,
                afficher_ids=False,
                flash=flash,
                boites_comptees=boites,
            )
        )

    def _boites_comptees_a_afficher(
        self, resultat
    ) -> list[tuple[int, float, float, float, float]]:
        """Met à jour l'état interne et retourne les boîtes vertes à dessiner.

        **La boîte verte SUIT la personne, elle ne reste pas sur la ligne.**
        L'événement de comptage porte la position du franchissement ; mais le
        track, lui, continue d'être suivi tant que la personne reste dans la
        bande de détection (`_purger_franchis` ne touche pas au tracker). On
        stocke donc le `track_id`, pas une boîte figée, et on cherche sa
        position COURANTE dans `resultat.tracks` à chaque frame : la boîte
        verte colle à la tête qui avance.

        **Quand disparaît-elle ?** Quand la personne sort de la bande après la
        ligne, elle n'est plus détectée : le tracker la lâche après
        `survie_max` frames sans détection, donc elle disparaît de
        `resultat.tracks` et la boîte avec elle. C'est exactement le geste que
        l'opérateur décrit — détecté dans la bande, vert au franchissement,
        plus de boîte à la sortie.

        Deux bornes, chacune rattrape la défaillance de l'autre :
        - **La vie du track** est la règle normale : boîte tant que la personne
          est suivie. C'est elle qui fait suivre la boîte ET qui la fait
          disparaître à la sortie de bande.
        - **La durée** (`DUREE_BOITE_COMPTEE_FRAMES`) est le filet : si une
          personne s'arrête dans la bande après la ligne, son track reste
          vivant — la boîte verte ne doit pas rester affichée indéfiniment.
        - **Le nombre** (`TAILLE_MAX_BOITES_COMPTES`) est le filet absolu : si
          rien ne purge, la liste ne peut pas croître sans fin.
        """
        suivis = {t.track_id: t for t in getattr(resultat, "tracks", None) or []}

        # Rattraper les événements non encore transformés en boîte verte.
        # L'affichage peut sauter des frames (vitesse lente) : un événement
        # produit sur une frame NON affichée ne serait jamais vu si on ne
        # lisait que `resultat.evenements`, qui ne porte que la dernière frame
        # traitée. L'historique du compteur, lui, garde TOUT — on en consomme
        # un cran de plus à chaque affichage.
        if self.compteur is not None:
            historique = getattr(self.compteur, "evenements", None) or []
            if len(historique) < self._dernier_evenement_affiche:
                # Nouvelle analyse : l'historique a été vidé, on repart de zéro
                # au lieu de re-créer des boîtes déjà vues.
                self._dernier_evenement_affiche = 0
            for ev in historique[self._dernier_evenement_affiche :]:
                if ev.track_id is not None:
                    self._boites_comptees.append((ev.frame, ev.track_id))
            if len(historique) > self._dernier_evenement_affiche:
                # Un ou plusieurs comptes NOUVEAUX viennent d'être rattrapés :
                # le flash s'arme même si la frame affichée n'en porte aucun.
                # En analyse par lot, un compte tombe souvent sur une frame
                # intermédiaire jamais affichée — sans cet armement ici, le
                # flash ne s'allumerait plus du tout.
                self._flash = FLASH_FRAMES
            self._dernier_evenement_affiche = len(historique)
        else:
            # Pas de compteur (test, ou fenêtre construite seule) : on retombe
            # sur les événements de la frame affichée.
            for ev in getattr(resultat, "evenements", None) or []:
                if ev.track_id is not None:
                    self._boites_comptees.append((ev.frame, ev.track_id))

        # On filtre par la vie du track, pas en purgeant sur place : la liste
        # retournée est celle que l'overlay dessine, et une liste mutée pendant
        # que l'overlay la parcourt serait une course.
        vivantes: list[tuple[int, int]] = []
        for frame_comptage, track_id in self._boites_comptees:
            if track_id not in suivis:
                continue  # la personne a quitté la bande : le track est mort
            age = resultat.frame_index - frame_comptage
            if age > DUREE_BOITE_COMPTEE_FRAMES:
                continue  # filet de durée : personne arrêtée dans la bande
            vivantes.append((frame_comptage, track_id))

        # La plus récente d'abord : un dépassement du plafond sacrifie alors les
        # plus anciennes, celles que l'opérateur a déjà eu le temps de voir.
        vivantes.sort(key=lambda boite: boite[0], reverse=True)
        self._boites_comptees = vivantes[:TAILLE_MAX_BOITES_COMPTES]

        # Les positions COURANTES, à la frame affichée.
        return [
            (frame, *suivis[track_id].bbox)
            for frame, track_id in self._boites_comptees
        ]

    # -- Compteurs et bilan ----------------------------------------------

    def maj_compteurs(self, total: int, presents: int, frames: int) -> None:
        """Met à jour le gros chiffre et le détail sous le compteur."""
        self.compteur_total = int(total)
        self.presents = int(presents)
        self.frames = int(frames)
        # `str(int)` : le chiffre est lu de loin, un « 42.0 » se lirait mal et
        # une largeur changeante ferait Sauter l'alignement.
        self.label_compteur.setText(str(int(total)))
        # Barre de progression : frames TRAITÉES rapportées au total annoncé.
        # Elle avance à la cadence d'affichage (~15 i/s), pas de traitement —
        # largement suffisant pour lire « où on en est ».
        if self._nb_frames_video > 0:
            self.barre_progression.setValue(min(int(self.frames), self._nb_frames_video))
        # `presents` et `frames` sont enregistrés mais plus affichés : le
        # second alimente encore l'avertissement « 0 compté », qui raisonne en
        # secondes de VIDÉO, et le premier le bilan de fin d'analyse.
        self._maj_avertissement_sens(int(total), int(frames))

    # -- Avertissement « 0 compté » --------------------------------------

    def _maj_avertissement_sens(self, total: int, frames: int) -> None:
        """Rappelle de vérifier le sens quand le compteur reste bloqué à zéro.

        C'est le SEUL réglage qui produise un zéro fiable : le sens par défaut
        vaut +1, et la vidéo de référence donne 315 personnes en `sens=-1`
        contre 1 en `sens=+1`. Un opérateur qui laisse le défaut voit une vidéo
        défiler normalement, des boîtes de détection s'afficher, et un zéro
        qui ne bouge pas — rien ne lui dit que le cortège est compté à
        l'envers. D'où le rappel.

        **Déclencheur volontairement simple** : `total == 0` après au moins
        `SEUIL_AVERTISSEMENT_SENS_S` secondes de vidéo ANALYSÉE. La variante
        « et le sens vient d'être changé » a été écartée : le cas le plus
        fréquent est précisément l'opérateur qui n'y touche pas, donc cette
        condition supprimerait l'avertissement dans le cas pour lequel il
        existe. Le seuil de 3 s suffit à écarter le démarrage d'analyse et le
        risque d'alerter sur une scène où, simplement, personne n'est encore
        passé.

        Le rappel est une fois par analyse, jamais par frame : `_afficher` est
        appelé 25 fois par seconde, et un message qui clignote n'est plus un
        message. Il disparaît dès que `total` repart, et à chaque nouvelle
        analyse.
        """
        if not self._session_ouverte or total > 0:
            self._avertissement_sens_affiche = False
            self.label_avertissement.setVisible(False)
            return
        if self._avertissement_sens_affiche:
            return
        # Le seuil est en SECONDES DE VIDÉO, pas en frames affichées : ralentir
        # la présentation ne doit pas retarder ni hâter l'avertissement, qui
        # parle d'un temps vécu par l'opérateur.
        if frames / self._fps() < SEUIL_AVERTISSEMENT_SENS_S:
            return
        self._avertissement_sens_affiche = True
        self.label_avertissement.setText(AVERTISSEMENT_SENS)
        self.label_avertissement.setVisible(True)
        log.info("avertissement « 0 compté » : vérifier le sens de la ligne")

    def _afficher_bilan(self) -> None:
        """Bilan de fin de lecture, dans la barre de statut.

        Les chiffres sont calculés ici à partir du `Resultat`, sans passer par
        `compteur.rapport` : c'est un résumé d'une ligne, pas le récapitulatif
        complet — celui-ci est dessiné par `PanneauRecap`, appelé par
        `arreter()`.

        L'export a été supprimé, donc ce texte ne propose plus rien à
        exporter : il annonce le résultat et s'arrête là.
        """
        if self.compteur is None:
            return
        resultat = self.compteur.resultat(
            self.config.modele, self.frames, self.frames / self._fps()
        )
        minutes = max(1e-9, resultat.secondes / 60.0)
        par_minute = resultat.total / minutes
        self.label_statut.setText(
            f"Terminé — {resultat.total} personnes, "
            f"{par_minute:.1f}/min, "
            f"max {resultat.presents_max} présents, "
            f"{resultat.presents_moyen:.1f} en moyenne."
        )

    # -- Fermeture --------------------------------------------------------

    def _liberer_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def closeEvent(self, event) -> None:  # noqa: N802 (API Qt)
        """Libère la capture en sortant.

        Sous Windows, une `VideoCapture` non libérée garde le fichier ouvert :
        la vidéo ne peut plus être déplacée ni supprimée tant que l'application
        tourne.
        """
        self._timer_traitement.stop()
        self._timer_affichage.stop()
        self._liberer_capture()
        super().closeEvent(event)


def _marquer_point(img, point: tuple[int, int]):
    """Cercles sur un point en cours de placement (le geste n'est pas fini)."""
    cv2 = _cv2()
    x, y = point
    sortie = img.copy()
    cv2.drawMarker(
        sortie,
        (int(x), int(y)),
        (60, 200, 255),
        cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2,
    )
    return sortie


def _cv2():
    import cv2

    return cv2
