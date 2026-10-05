r"""Démarrage de la version 2 : vérifier, installer si besoin, DÉMARCHER.

**La règle qui gouverne tout ce module : l'application démarre TOUJOURS.**

Aucune fenêtre qui refuse de s'ouvrir. Un logiciel de terrain qui, faute de
réseau, affiche « téléchargement impossible » et se ferme laisse l'opérateur
devant un double-clic qui ne fait rien, sans même savoir pourquoi. Le repli
est donc systématique :

1. torch déjà dans NOTRE cache → démarrage immédiat, sans rien demander ;
2. sinon, GPU NVIDIA présent → installation CUDA, avec un bouton
   « Continuer sur CPU » ;
3. l'installation échoue → on démarre quand même, sans torch. La fenêtre
   s'ouvre, l'opérateur peut charger une vidéo, tracer sa ligne ; seule
   l'analyse est indisponible, et l'indicateur le dit.

Le troisième point est ce qui distingue cette version de la précédente : la v1
ne démarrait pas du tout sans torch, puisqu'il l'avait embarqué.

**DEUX barres, pas une.** L'utilisateur vit deux attentes de nature
différente : le téléchargement (des octets qui arrivent, une vitesse, un temps
restant) puis l'installation (des fichiers qui se décompressent). Une barre
unique qui mouille pendant la décompression — deux minutes sans rien montrer —
se lit comme un blocage, et l'utilisateur tue le processus. C'est exactement le
cas qu'on cherche à éviter.

L'installation tourne dans un `QThread` et la fenêtre reste vivante : sans
cela, Qt affiche un écran blanc, ce qui se lit encore comme un plantage.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import NamedTuple

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from compteur import telechargement as tl

log = logging.getLogger(__name__)


class EtatTorch(NamedTuple):
    """Ce que la fenêtre doit afficher dans son indicateur.

    Renvoyé par `preparer()` et stocké par la fenêtre. `cuda` est le fait
    observé — pas le réglage, pas le souhait : c'est ce qui distingue «
    l'utilisateur a demandé CUDA » de « le calcul tournera réellement sur le
    GPU ».
    """

    #: Version de torch, ou chaîne vide si absent.
    version: str
    #: Vrai si le calcul se fait réellement sur un GPU NVIDIA.
    cuda: bool
    #: Nom de la carte, ou chaîne vide.
    gpu: str
    #: Message d'erreur si une tentative a échoué, chaîne vide sinon.
    probleme: str = ""

    @property
    def disponible(self) -> bool:
        return bool(self.version)


def _formater(octets: int) -> str:
    """`2602753280` -> `2,4 Go`. Une unité lisible, pas un nombre d'octets.

    L'utilisateur compare cette taille à ce qui reste sur son disque, et il ne
    fait pas cette conversion de tête.
    """
    if octets <= 0:
        return "taille inconnue"
    valeur = float(octets)
    for unite in ("o", "Ko", "Mo", "Go"):
        if valeur < 1024 or unite == "Go":
            # VIRGULE décimale, comme partout ailleurs dans l'interface : un
            # point décimal dans une fenêtre française se lit comme une faute
            # de frappe, et fait douter de l'outil entier.
            return f"{valeur:.1f}".replace(".", ",") + f" {unite}"
        valeur /= 1024
    return f"{valeur:.1f}".replace(".", ",") + " Go"


def _duree(secondes: float) -> str:
    """`95` -> `1 min 35`. Vide si on n'a pas d'estimation."""
    if secondes <= 0:
        return ""
    if secondes < 60:
        return f"{int(secondes)} s"
    minutes = int(secondes // 60)
    reste = int(secondes % 60)
    if minutes < 60:
        return f"{minutes} min {reste:02d}"
    return f"{minutes // 60} h {minutes % 60:02d}"


class _Worker(QObject):
    """Installe dans un thread, parle au monde par signaux.

    Un signal Qt est le seul moyen sûr de toucher l'interface depuis un thread :
    appeler `QLabel.setText` directement depuis le thread d'installation est un
    crash aléatoire, en général sous charge — c'est-à-dire au moment précis où
    la barre bouge le plus et où une mise à jour est réellement nécessaire.
    """

    #: Émis à chaque point d'avancement (téléchargement ou installation).
    progression = Signal(object)
    #: Émis une fois, à la fin : `True` si l'installation a réussi.
    fini = Signal(bool)

    def __init__(self, peripherique: str, dossier: pathlib.Path) -> None:
        super().__init__()
        self.peripherique = peripherique
        self.dossier = dossier
        self._annule = threading.Event()
        #: Mis à jour par la fenêtre : « Continuer sur CPU ».
        self.basculer_cpu = threading.Event()

    def annuler(self) -> None:
        self._annule.set()

    def _progression(self, p: tl.Progression) -> None:
        if self._annule.is_set():
            raise tl.Annulation("annulé par l'utilisateur")
        # L'utilisateur a demandé le CPU en cours de route : on bascule. Le
        # paquet CUDA déjà téléchargé reste dans le cache d'`uv`, il ne sert
        # simplement plus.
        if self.basculer_cpu.is_set() and self.peripherique == "cuda":
            log.info("bascule vers CPU demandée pendant l'installation CUDA")
            self.peripherique = "cpu"
            self.basculer_cpu.clear()
        self.progression.emit(p)

    def run(self) -> None:
        try:
            ok = tl.installer_torch(
                self.peripherique, self._progression, self.dossier
            )
        except Exception as exc:  # noqa: BLE001 — un thread ne laisse rien fuir
            log.exception("installation terminée sur une exception : %s", exc)
            ok = False
        self.fini.emit(ok)


class _BarrePhase(QGroupBox):
    """Une phase = un titre, une barre, une ligne de détail.

    Le détail n'est pas un luxe : « Téléchargement de CUDA — 1,2 Go / 2,4 Go —
    34 % » ne dit pas si ça avance à 2 Mo/s ou à 200 Mo/s, et ces deux
    situations demandent des décisions opposées à l'utilisateur (attendre, ou
    passer sur le CPU).
    """

    def __init__(self, titre: str, parent=None) -> None:
        super().__init__(titre, parent)
        racine = QVBoxLayout(self)
        racine.setContentsMargins(8, 8, 8, 8)
        self.barre = QProgressBar()
        self.barre.setRange(0, 100)
        self.barre.setValue(0)
        self.barre.setTextVisible(True)
        racine.addWidget(self.barre)
        self.detail = QLabel("")
        self.detail.setObjectName("sous_titre")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        racine.addWidget(self.detail)

    def maj(self, p: tl.Progression) -> None:
        """Met à jour barre et détail à partir d'une progression."""
        self.barre.setValue(p.pourcent)
        morceaux = [p.texte]

        if p.total <= 0:
            # Total inconnu : on montre le compte seul, et surtout PAS un
            # pourcentage inventé. Une barre qui avance sans base se remarque.
            if p.fait > 0:
                morceaux.append(_formater(p.fait))
            self.detail.setText(" — ".join(morceaux))
            return

        if p.phase == tl.PHASE_TELECHARGEMENT:
            morceaux.append(
                f"{_formater(p.fait)} / {_formater(p.total)} — {p.pourcent} %"
            )
            if p.vitesse > 0:
                morceaux.append(f"{_formater(p.vitesse)}/s")
            reste = _duree(p.secondes_restantes)
            if reste:
                morceaux.append(f"reste {reste}")
        else:
            # Phase 2 : `uv` ne dit pas combien d'octets il écrit, on compte
            # des paquets. Le dire évite que l'utilisateur cherche une
            # contrepartie en Go qui n'existe pas.
            morceaux.append(f"{p.fait} / {p.total} paquets — {p.pourcent} %")

        self.detail.setText(" — ".join(morceaux))

    def finie(self, texte: str) -> None:
        self.barre.setValue(100)
        self.detail.setText(texte)


class DialogueInstallation(QDialog):
    """Fenêtre d'installation en deux phases, avec repli CPU à portée de bouton.

    Le bouton « Continuer sur CPU » est là pour une raison précise : la
    version CUDA pèse 2,4 Go et la CPU 120 Mo. Sur une connexion mobile, ou un
    disque presque plein, attendre 2,4 Go pour.constants — ou échouer — est un
    mauvais échange. Le repli est proposé PENDANT le téléchargement, pas
    seulement après un échec.
    """

    def __init__(self, gpu: str, taille: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Préparation du moteur de calcul")
        self.setModal(True)
        self.setMinimumWidth(560)

        racine = QVBoxLayout(self)

        self.label_titre = QLabel(
            f"Installation de CUDA — {gpu or 'carte NVIDIA'}"
        )
        self.label_titre.setObjectName("compteur_titre")
        racine.addWidget(self.label_titre)

        self.label_detail = QLabel(
            f"{_formater(taille)} environ, dans {tl.dossier_cache()}. Rien à "
            f"installer à la main et aucun droit administrateur demandé : tout "
            f"se fait automatiquement. Si le téléchargement est interrompu, il "
            f"reprendra où il s'est arrêté."
        )
        self.label_detail.setWordWrap(True)
        racine.addWidget(self.label_detail)

        # -- Phase 1 : téléchargement. Visible dès l'ouverture.
        self.phase_telechargement = _BarrePhase("Phase 1 — Téléchargement")
        racine.addWidget(self.phase_telechargement)

        # -- Phase 2 : installation. Masquée jusqu'à ce que `uv` commence
        #    vraiment à décompresser : montrer une barre à 0 % pendant que le
        #    téléchargement court donnerait l'impression que le logiciel
        #    attend déjà l'installation.
        self.phase_installation = _BarrePhase("Phase 2 — Installation")
        self.phase_installation.setVisible(False)
        racine.addWidget(self.phase_installation)

        self.label_etat = QLabel("Connexion au serveur…")
        racine.addWidget(self.label_etat)

        boutons = QHBoxLayout()
        boutons.addStretch(1)
        self.btn_cpu = QPushButton("Continuer sur CPU (120 Mo, plus lent)")
        self.btn_cpu.setToolTip(
            "Abandonne la version CUDA et installe une version beaucoup plus "
            "légère, qui calcule sur le processeur. Le comptage est identique ; "
            "seule la vitesse change."
        )
        self.btn_attendre = QPushButton("Attendre la fin de l'installation")
        self.btn_attendre.setObjectName("primaire")
        boutons.addWidget(self.btn_cpu)
        boutons.addWidget(self.btn_attendre)
        racine.addLayout(boutons)

        self._phase_courante = tl.PHASE_TELECHARGEMENT

    def maj(self, p: tl.Progression) -> None:
        """Aiguille la progression vers la bonne phase."""
        if p.phase != self._phase_courante:
            # Bascule : la phase 1 se fige complète, la phase 2 apparaît.
            self._phase_courante = p.phase
            if p.phase == tl.PHASE_INSTALLATION:
                self.phase_telechargement.finie("Téléchargement terminé")
                self.phase_installation.setVisible(True)
                self.btn_cpu.setEnabled(False)
                self.btn_cpu.setText("Trop tard — installation en cours")

        if p.phase == tl.PHASE_TELECHARGEMENT:
            self.phase_telechargement.maj(p)
        else:
            self.phase_installation.maj(p)
        self.label_etat.setText(p.texte)


def preparer(parent=None, dossier: pathlib.Path | None = None) -> EtatTorch:
    """Assure la présence de torch, puis renvoie l'état à afficher.

    Renvoie un `EtatTorch` et ne lève JAMAIS. Si torch est absent, l'état
    retourné le dit, et la fenêtre s'ouvre quand même.
    """
    dossier = pathlib.Path(dossier or tl.dossier_cache())

    # 1. Déjà là ? On ne demande rien, et surtout on ne réinstalle pas.
    if tl.torch_installe(dossier):
        tl.ajouter_au_sys_path(dossier)
        cuda = _cuda_reelle()
        return EtatTorch(tl.version_torch(), cuda, tl.nom_gpu() if cuda else "")

    # 2. Choix de la variante. Le GPU est demandé SANS torch : c'est
    #    `nvidia-smi` qui répond, pas `torch.cuda`.
    gpu = tl.nom_gpu()
    cuda = bool(gpu)
    variante = "cuda" if cuda else "cpu"
    log.info("torch absent de notre cache — installation %s", variante.upper())

    dialogue = DialogueInstallation(gpu, tl.taille_attendue(cuda), parent)

    resultat: list[bool] = []
    worker = _Worker(variante, dossier)
    thread = QThread()
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.progression.connect(dialogue.maj)
    worker.fini.connect(resultat.append)
    worker.fini.connect(thread.quit)
    thread.finished.connect(dialogue.accept)

    def _basculer_cpu() -> None:
        """L'utilisateur a choisi le CPU : on bascule sans fermer la fenêtre."""
        log.info("repli CPU demandé par l'utilisateur")
        worker.basculer_cpu.set()
        dialogue.btn_cpu.setEnabled(False)
        dialogue.btn_cpu.setText("Bascule vers CPU en cours…")
        dialogue.label_titre.setText("Installation de la version CPU")

    dialogue.btn_cpu.clicked.connect(_basculer_cpu)
    dialogue.btn_attendre.clicked.connect(dialogue.accept)

    thread.start()
    dialogue.exec()
    thread.quit()
    thread.wait(30000)

    # 3. Verdict, quel qu'il soit : l'application démarre.
    ok = bool(resultat) and resultat[0]
    if ok:
        tl.ajouter_au_sys_path(dossier)
        cuda = _cuda_reelle()
        return EtatTorch(tl.version_torch(), cuda, tl.nom_gpu() if cuda else "")

    probleme = tl.dernier_message()
    if not probleme:
        probleme = (
            "Le moteur de calcul n'a pas pu être installé. L'analyse de la "
            "vidéo ne sera pas possible ; le reste du logiciel fonctionne. "
            "Relancez le logiciel pour réessayer."
        )
    log.warning("torch indisponible : %s", probleme)

    # Le message va dans le JOURNAL, pas dans une boîte modale : on est avant
    # la construction de la fenêtre principale, et un « OK » modal ici
    # bloquerait le démarrage que cette fonction est censée garantir. Il sera
    # affiché dans la barre de statut, visible sans clic.
    if parent is not None and _fenetre_visible(parent):
        QMessageBox.warning(
            parent,
            "Moteur de calcul indisponible",
            probleme
            + "\n\nVous pouvez tout de même ouvrir l'application. "
            "Relancez-la plus tard pour retenter l'installation.",
        )
    return EtatTorch("", False, "", probleme)


def _fenetre_visible(parent) -> bool:
    """Vrai si `parent` est une fenêtre affichée.

    Même règle que dans `interface.app._signaler` : sous `offscreen`, ou dans
    les tests, une boîte modale serait un blocage définitif.
    """
    try:
        return bool(parent.isVisible())
    except AttributeError:
        return False


def _cuda_reelle() -> bool:
    """Vrai si le torch importé voit réellement un GPU.

    Passe par `compteur.detecteur.cuda_disponible`, qui interroge
    `device_count()` et non `is_available()` : sur cette machine, avec
    `CUDA_VISIBLE_DEVICES=""`, le second renvoie `True` quand aucun GPU
    n'est visible. L'indicateur afficherait alors « CUDA » pour un calcul
    fait sur le processeur — exactement le mensonge que cet indicateur existe
    pour éviter.
    """
    try:
        from compteur.detecteur import cuda_disponible

        return cuda_disponible()
    except Exception as exc:  # noqa: BLE001 — torch cassé ne bloque pas le GUI
        log.info("état CUDA illisible : %s", exc)
        return False


def texte_indicateur(etat: EtatTorch) -> str:
    """Le texte affiché dans la barre de statut.

    Les trois formes demandées :

    - `CUDA 2.14.1 — NVIDIA GeForce RTX 4070 SUPER`
    - `CUDA absent — téléchargement nécessaire`
    - `CPU uniquement (torch 2.14.1)`
    """
    if not etat.disponible:
        if etat.probleme:
            return "Moteur de calcul absent — voir le message ci-dessus"
        if etat.gpu:
            return "CUDA absent — téléchargement nécessaire"
        return "Moteur de calcul absent — téléchargement nécessaire"
    if etat.cuda:
        base = f"CUDA {etat.version}"
        return f"{base} — {etat.gpu}" if etat.gpu else base
    return f"CPU uniquement (torch {etat.version})"