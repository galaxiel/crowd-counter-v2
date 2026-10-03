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
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from compteur.compteur import Compteur
from compteur.config import Config
from compteur.detecteur import Detecteur, charger_modele
from compteur.ligne import Ligne
from compteur.tracker import Tracker
from interface.overlay import dessiner
from interface.panneau_reglages import PanneauReglages
from interface.style import appliquer_style  # noqa: F401  (ré-export pour main.py)
from interface.widgets_video import WidgetVideo

log = logging.getLogger(__name__)

#: Vitesses de présentation proposées. La dernière, « max », ne limite rien :
#: on affiche chaque frame traitée dès qu'elle est prête.
VITESSE_MAX = "max"
VITESSE_PAR_DEFAUT = "max"
VITESSE_LENTE = "0.25×"

#: Cadence de référence quand la vidéo n'annonce pas la sienne (codec exotique,
#: conteneur mal écrit). 25 i/s est le standard des caméras de surveillance.
FPS_REPLI = 25.0

#: Durée du flash rouge de comptage, en frames AFFICHÉES. Le flash sert à
#: repérer d'un coup d'œil l'instant d'un comptage ; le compter en frames
#: traitées le rendrait subliminal dès qu'on ralentit l'affichage.
FLASH_FRAMES = 3

#: Filtres du sélecteur de fichier, par ordre de préférence.
FILTRES_VIDEOS = (
    "Vidéos (*.mp4 *.avi *.mkv *.mov *.m4v *.webm);;Tous les fichiers (*)"
)

#: Dossier proposé par défaut à l'export.
DOSSIER_EXPORT_DEFAUT = "sortie"


class FenetrePrincipale(QMainWindow):
    """La fenêtre : vidéo à gauche, compteur et réglages à droite."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Compteur de manifestation")
        self.resize(1400, 850)

        self.config: Config = Config.defauts()
        self.cap = None
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
        #: Nom du modèle effectivement chargé dans `self.detecteur`.
        self._modele_charge = ""
        self._flash = 0
        #: Dernier `FrameResult` traité, en attente d'affichage.
        self._resultat_courant = None
        #: Index du dernier résultat déjà affiché, pour ne pas redessiner.
        self._index_affiche = -1
        #: Position du marqueur de premier point, en pixels image.
        self._apercu_point: tuple[int, int] | None = None

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

        self.video = WidgetVideo()
        self.video.setMinimumWidth(760)
        racine.addWidget(self.video, stretch=3)

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

        self.label_details = QLabel("Présents : 0   •   Frames : 0")
        self.label_details.setObjectName("sous_titre")
        self.label_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        colonne.addWidget(self.label_details)

        self.label_statut = QLabel("Charge une vidéo pour commencer.")
        self.label_statut.setObjectName("sous_titre")
        self.label_statut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_statut.setWordWrap(True)
        self.label_statut.setMinimumHeight(56)
        colonne.addWidget(self.label_statut)

        self.panneau = PanneauReglages(self.config)
        colonne.addWidget(self.panneau, stretch=1)

        self.choix_vitesse = QComboBox()
        self.choix_vitesse.addItems(
            ["0.25×", "0.5×", "1×", "2×", "4×", VITESSE_MAX]
        )
        self.choix_vitesse.setCurrentText(VITESSE_PAR_DEFAUT)
        self.choix_vitesse.setToolTip(
            "Vitesse de présentation.\n"
            "Ralentit l'affichage seul : le comptage, lui, reste à la vitesse "
            "maximale de la machine."
        )
        colonne.addWidget(self.choix_vitesse)

        barre = QHBoxLayout()
        self.btn_video = QPushButton("Charger la vidéo")
        self.btn_ligne = QPushButton("Tracer la ligne")
        self.btn_ligne.setToolTip(
            "Clique ensuite deux points sur l'image, en haut et en bas."
        )
        self.btn_lancer = QPushButton("Lancer")
        self.btn_lancer.setObjectName("primaire")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_export = QPushButton("Exporter")
        for b in (
            self.btn_video,
            self.btn_ligne,
            self.btn_lancer,
            self.btn_pause,
            self.btn_stop,
            self.btn_export,
        ):
            barre.addWidget(b)
        colonne.addLayout(barre)

        # Traitement : intervalle 0, donc « aussi vite que possible ». C'est
        # le GPU qui fixe la cadence de cette minuterie, jamais le curseur de
        # vitesse de présentation.
        self._timer_traitement = QTimer(self)
        self._timer_traitement.setInterval(0)
        # Affichage : intervalle réglé par `_appliquer_vitesse`.
        self._timer_affichage = QTimer(self)
        self._timer_affichage.setInterval(0)

    def _connecter(self) -> None:
        self.btn_video.clicked.connect(self._on_charger_video)
        self.btn_ligne.clicked.connect(self._on_tracer_ligne)
        self.btn_lancer.clicked.connect(self.lancer)
        self.btn_pause.clicked.connect(self._basculer_pause)
        self.btn_stop.clicked.connect(self.arreter)
        self.btn_export.clicked.connect(self._on_exporter)
        self.video.clic.connect(self._on_clic_video)
        self.panneau.config_modifiee.connect(self._sur_config)
        self.choix_vitesse.currentTextChanged.connect(self._appliquer_vitesse)
        self._timer_traitement.timeout.connect(self._tick)
        self._timer_affichage.timeout.connect(self._afficher)

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
        self._appliquer_vitesse()

        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = self._fps()
        self.label_statut.setText(
            f"{p.name} — {nb_frames} frames à {fps:.0f} i/s.\n"
            "Trace la ligne, puis lance."
        )
        self._maj_boutons()
        return True

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
        self.label_statut.setText(
            f"Ligne posée de ({self.ligne.p1[0]:.0f}, {self.ligne.p1[1]:.0f}) "
            f"à ({self.ligne.p2[0]:.0f}, {self.ligne.p2[1]:.0f}). "
            "Le sens se change dans les réglages ; la flèche verte indique "
            "le sens compté. Lance quand tu es prêt."
        )
        return True

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
        self.config = config
        if self.ligne is not None:
            try:
                self.ligne = Ligne(
                    self.ligne.p1,
                    self.ligne.p2,
                    epaisseur=config.epaisseur_bande,
                    sens=config.sens,
                    hysteresis=config.frames_hysteresis,
                )
            except ValueError as exc:
                # Réglage devenu impossible (épaisseur 0, sens nul) : on garde
                # l'ancienne ligne plutôt que de laisser le moteur sans ligne,
                # qui ne compterait plus personne.
                log.warning("réglage de ligne refusé : %s", exc)
        if self.compteur is not None:
            self.compteur.config = config
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
                    charger_modele(self.config.modele), self.config
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

        self.compteur.ajuster_ligne(self.ligne)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.compteur_total = 0
        self.presents = 0
        self.frames = 0
        self._resultat_courant = None
        self._index_affiche = -1
        self._flash = 0
        self._ouvrir_session()
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
        """Ferme la session d'analyse et rend la main à l'opérateur."""
        self._timer_traitement.stop()
        self._timer_affichage.stop()
        self._en_analyse = False
        self._session_ouverte = False
        self.btn_pause.setText("Pause")
        self._maj_boutons()

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
        self.btn_export.setEnabled(self.compteur is not None)

    # -- Boucle de lecture ------------------------------------------------

    def _tick(self) -> None:
        """Traite UNE frame. Ne dessine rien : c'est `_afficher` qui décide.

        La séparation est ce qui rend la vitesse de présentation inoffensive.
        `_afficher` sera rappelle moins souvent, ou plus souvent, sans que la
        cadence de détection, de tracking et de comptage soit touchée.
        """
        if self.cap is None or self.compteur is None:
            self._timer_traitement.stop()
            return

        ok, img = self.cap.read()
        if not ok:
            self.arreter()
            self._afficher_bilan()
            return

        fps = self._fps()
        self._resultat_courant = self.compteur.traiter_frame(
            img, self.frames, self.frames / fps
        )
        self.frames += 1

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

        self.maj_compteurs(resultat.total, resultat.presents, self.frames)
        self.video.definir_image(
            dessiner(
                resultat.image,
                resultat,
                self.ligne,
                afficher_ids=False,
                flash=flash,
            )
        )

    def _appliquer_vitesse(self, _texte: str | None = None) -> None:
        """Règle la cadence d'AFFICHAGE. Ne touche jamais au traitement.

        L'intervalle est l'écart entre deux images présentées : à 0,25x sur une
        vidéo à 25 i/s, l'opérateur voit une image toutes les 160 ms. Les
        frames traitées entre-temps ne sont pas mises en file d'attente — on
        saute simplement les images intermédiaires, comme un lecteur vidéo en
        lecture lente. Le décompte, lui, a avancé de 4 à chaque image affichée.
        """
        self._timer_affichage.setInterval(self._intervalle_affichage())

    def _intervalle_affichage(self) -> int:
        texte = self.choix_vitesse.currentText()
        if texte == VITESSE_MAX:
            return 0
        try:
            multiplicateur = float(texte.replace("×", "").replace("x", ""))
        except ValueError:
            log.warning("vitesse illisible : %r", texte)
            return 0
        if multiplicateur <= 0:
            return 0
        return max(0, int(round(1000.0 / (self._fps() * multiplicateur))))

    # -- Compteurs et bilan ----------------------------------------------

    def maj_compteurs(self, total: int, presents: int, frames: int) -> None:
        """Met à jour le gros chiffre et le détail sous le compteur."""
        self.compteur_total = int(total)
        self.presents = int(presents)
        self.frames = int(frames)
        # `str(int)` : le chiffre est lu de loin, un « 42.0 » se lirait mal et
        # une largeur changeante ferait Sauter l'alignement.
        self.label_compteur.setText(str(int(total)))
        self.label_details.setText(
            f"Présents : {int(presents)}   •   Frames : {int(frames)}"
        )

    def _afficher_bilan(self) -> None:
        """Bilan de fin de lecture, sans dépendre de `compteur.rapport`.

        Les statistiques sont calculées ici à partir du `Resultat` : la fenêtre
        ne doit pas dépendre d'un module livré par une autre tâche pour
        afficher un résumé.
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
            f"{resultat.presents_moyen:.1f} en moyenne.\n"
            "Pense à exporter le décompte."
        )

    # -- Export ------------------------------------------------------------

    def _on_exporter(self) -> None:
        dossier = QFileDialog.getExistingDirectory(
            self, "Dossier de sortie", DOSSIER_EXPORT_DEFAUT
        )
        if dossier:
            self.exporter(dossier)

    def exporter(self, dossier: str) -> bool:
        """Écrit le CSV et le JSON du décompte dans ``dossier``.

        Le module `compteur.rapport` est importé À L'INTÉRIEUR de la méthode :
        s'il n'est pas encore livré, l'export refuse en displaying pourquoi au
        lieu de faire tomber la fenêtre entière sur un clic.
        """
        if self.compteur is None or self.video_finiment is None:
            self._signaler(
                "Rien à exporter", "Lance d'abord une analyse sur une vidéo."
            )
            return False

        try:
            from compteur.rapport import (
                chemins_par_defaut,
                ecrire_csv,
                ecrire_json,
            )
        except ImportError as exc:
            self._signaler(
                "Export indisponible",
                f"Le module d'export n'est pas disponible ({exc}).",
            )
            return False

        try:
            resultat = self.compteur.resultat(
                self.config.modele, self.frames, self.frames / self._fps()
            )
            chemins = chemins_par_defaut(str(self.video_finiment), dossier)
            ecrire_csv(resultat, chemins["csv"])
            ecrire_json(resultat, chemins["json"])
        except Exception as exc:  # noqa: BLE001 — un disque plein ne doit pas fermer l'app
            self._signaler("Export impossible", str(exc), grave=True)
            return False

        self.label_statut.setText(
            f"Exporté :\n{chemins['csv'].name}\n{chemins['json'].name}\n"
            f"(dossier {dossier})"
        )
        return True

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
