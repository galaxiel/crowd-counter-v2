r"""Tests du mode dégradé et de la fenêtre d'installation (version 2).

C'est la promesse centrale de la v2 : l'application démarre TOUJOURS. Une
régression ici ne se voit pas dans les tests de comptage — elle se voit sur la
machine d'un opérateur, au premier lancement, sans réseau, quand la fenêtre ne
s'ouvre pas.

On y vérifie aussi les DEUX barres. C'est une demande explicite : une seule
barre qui mouline pendant la décompression se lit comme un blocage, et
l'utilisateur tue le processus.

Aucun test n'installe torch : `installer_torch` est remplacé par un faux. Un
test qui télécharge 2,4 Go n'est pas un test.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("PySide6")

from interface import demarrage  # noqa: E402
from interface.demarrage import (  # noqa: E402
    DialogueInstallation,
    EtatTorch,
    _formater,
    texte_indicateur,
)
from compteur import telechargement as tl  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Signal:
    def __init__(self) -> None:
        self._slots: list = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def disconnect(self, slot=None) -> None:
        if slot is None:
            self._slots.clear()
        elif slot in self._slots:
            self._slots.remove(slot)

    def _emettre(self) -> None:
        for slot in list(self._slots):
            slot()

    def emit(self, *args) -> None:
        self._emettre()


class _BoutonFactice:
    """Remplace un `QPushButton` sans boucle d'événements.

    `_basculer_cpu()` de `preparer` s'y connecte avec `.clicked.connect(...)`,
    ce qui marche sur un objet dont `.clicked` est une liste de slots.
    """

    def __init__(self) -> None:
        self.clicked = _Signal()
        self.setEnabled_calls: list[bool] = []
        self.texts: list[str] = []

    def setEnabled(self, actif: bool) -> None:
        self.setEnabled_calls.append(actif)

    def setText(self, texte: str) -> None:
        self.texts.append(texte)

    def click(self) -> None:
        self.clicked._emettre()


class _LabelFactice:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def setText(self, texte: str) -> None:
        self.texts.append(texte)

    def text(self) -> str:
        return self.texts[-1] if self.texts else ""


class _DialogueFactice:
    """Fenêtre d'installation qui ne s'ouvre pas.

    `preparer` appelle `dialogue.exec()`, qui attend un clic. Sous `pytest`,
    personne ne clique et le test resterait bloqué indéfiniment. Ce faux
    dialogue exécute le vrai code de `preparer` — thread, signaux, connexion
    des boutons — mais `exec()` renvoie tout de suite.
    """

    instances: list["_DialogueFactice"] = []

    def __init__(self, gpu, taille, parent=None) -> None:
        self.gpu = gpu
        self.taille = taille
        self.btn_cpu = _BoutonFactice()
        self.btn_attendre = _BoutonFactice()
        self.label_titre = _LabelFactice()
        self.label_detail = _LabelFactice()
        self.label_etat = _LabelFactice()
        self.exec_calls = 0
        self.progression_recues: list = []
        type(self).instances.append(self)

    def maj(self, p) -> None:
        self.progression_recues.append(p)

    def accept(self) -> None:
        pass

    def exec(self) -> int:
        self.exec_calls += 1
        return 0


@pytest.fixture
def application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def sans_dialogue(monkeypatch):
    """Remplace la vraie fenêtre d'installation par un faux dialogue."""
    _DialogueFactice.instances.clear()
    monkeypatch.setattr(demarrage, "DialogueInstallation", _DialogueFactice)
    return _DialogueFactice


# ---------------------------------------------------------------------------
# 1. La règle : démarrer quand même
# ---------------------------------------------------------------------------


def test_preparer_demarre_meme_sans_torch(tmp_path, monkeypatch, sans_dialogue):
    """Torch absent et installation impossible : un état, sans lever.

    `preparer` est appelé par `main.py` AVANT la construction de la fenêtre.
    Une exception ici tue le processus et l'utilisateur se retrouve devant un
    double-clic sans effet. C'est le test le plus important du fichier.
    """
    monkeypatch.setattr(demarrage.tl, "installer_torch", lambda *a, **k: False)
    monkeypatch.setattr(demarrage.tl, "torch_installe", lambda d=None: False)
    monkeypatch.setattr(demarrage.tl, "nom_gpu", lambda: "")
    monkeypatch.setattr(demarrage.tl, "taille_attendue", lambda cuda: 0)

    etat = demarrage.preparer(None, tmp_path)

    assert etat.version == ""
    assert etat.probleme, "un échec sans explication n'aide personne"


def test_preparer_ne_reinstalle_pas_torch_deja_present(
    tmp_path, monkeypatch, sans_dialogue
):
    """Torch déjà là : AUCUNE installation, et l'état reflects la CUDA.

    Le test utile sur une machine qui a déjà CUDA ailleurs : le logiciel
    détecte ce qu'il a et ne propose pas 2,4 Go à télécharger. C'est le cas le
    plus fréquent au deuxième lancement, et le plus visible par l'utilisateur
    s'il échoue.
    """
    appele = {"n": 0}

    def _interdit(*a, **k):  # pragma: no cover — ne doit jamais être atteint
        appele["n"] += 1
        raise AssertionError("torch est déjà installé : ne réinstalle rien")

    monkeypatch.setattr(demarrage.tl, "torch_installe", lambda d=None: True)
    monkeypatch.setattr(demarrage.tl, "installer_torch", _interdit)
    monkeypatch.setattr(demarrage.tl, "ajouter_au_sys_path", lambda *a, **k: True)
    monkeypatch.setattr(demarrage.tl, "version_torch", lambda: "2.14.1+cu126")
    monkeypatch.setattr(demarrage, "_cuda_reelle", lambda: True)
    monkeypatch.setattr(
        demarrage.tl, "nom_gpu", lambda: "NVIDIA GeForce RTX 4070 SUPER"
    )

    etat = demarrage.preparer(None, tmp_path)

    assert appele["n"] == 0
    assert etat.version.startswith("2.14.1")
    assert etat.cuda is True
    assert not _DialogueFactice.instances, "aucune fenêtre d'installation"


def test_un_echec_n_empeche_pas_la_fenetre_d_afficher(
    tmp_path, monkeypatch, sans_dialogue
):
    """Le chemin complet jusqu'à l'affichage : jamais d'exception.

    On va jusqu'à la fenêtre principale, parce que c'est ça qui compte : une
    fenêtre qui s'ouvre sur un compteur à zéro vaut mieux qu'une fenêtre qui ne
    s'ouvre pas.
    """
    from PySide6.QtWidgets import QApplication

    from interface.app import FenetrePrincipale

    monkeypatch.setattr(demarrage.tl, "installer_torch", lambda *a, **k: False)
    monkeypatch.setattr(demarrage.tl, "torch_installe", lambda d=None: False)
    monkeypatch.setattr(demarrage.tl, "nom_gpu", lambda: "")
    monkeypatch.setattr(demarrage.tl, "taille_attendue", lambda cuda: 0)

    app = QApplication.instance() or QApplication([])
    etat = demarrage.preparer(None, tmp_path)

    fenetre = FenetrePrincipale()
    fenetre.definir_etat_torch(etat)

    assert fenetre.label_torch.text(), "l'indicateur doit dire quelque chose"
    assert fenetre.label_compteur.text() == "0", "la fenêtre est utilisable"
    fenetre.close()


# ---------------------------------------------------------------------------
# 2. Le repli CPU, disponible pendant toute la phase 1
# ---------------------------------------------------------------------------


def test_le_repli_cpu_bascule_la_variante_en_cours_de_telechargement():
    """Le bouton « Continuer sur CPU » doit changer la variante EN COURS.

    Le worker télécharge le paquet CUDA déjà entamé quand l'utilisateur clique.
    Sans bascule au vol, le bouton ne servirait qu'au PROCHAIN lancement — ce
    qui est justement le moment où l'utilisateur a le moins envie d'attendre.
    """
    worker = demarrage._Worker("cuda", pathlib.Path("."))
    assert worker.peripherique == "cuda"

    vues: list = []
    worker.progression.connect(vues.append)

    worker.basculer_cpu.set()
    worker._progression(tl.Progression(
        phase=tl.PHASE_TELECHARGEMENT, texte="x", fait=0, total=0, vitesse=0, restant=-1
    ))

    assert worker.peripherique == "cpu", "la variante doit basculer sur CPU"
    assert vues, "la progression doit continuer à être remontée à la fenêtre"


def test_le_bouton_cpu_est_present_des_le_debut():
    """Le repli doit être proposed AVANT que quoi que ce soit ne rate.

    Un bouton qui n'apparaît qu'après un échec est un bouton inutile : à ce
    moment-là, l'utilisateur a déjà attendu.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    dialogue = DialogueInstallation("NVIDIA GeForce RTX 4070 SUPER", 2_602_755_161)
    assert dialogue.btn_cpu.isVisibleTo(dialogue) or dialogue.btn_cpu.isEnabled()
    assert "CPU" in dialogue.btn_cpu.text()
    dialogue.close()


# ---------------------------------------------------------------------------
# 3. Les DEUX barres
# ---------------------------------------------------------------------------


def test_la_deuxieme_barre_est_masquee_tant_que_le_telechargement_court(
    application,
):
    """Afficher une barre d'installation à 0 % pendant le téléchargement ment.

    L'utilisateur verrait deux barres, dont une figée à zéro, et en déduirait
    que le logiciel bloque. Elle n'apparaît qu'à la bascule.
    """
    dialogue = DialogueInstallation("RTX 4070 SUPER", 2_602_755_161)
    assert dialogue.phase_telechargement.isVisibleTo(dialogue)
    assert not dialogue.phase_installation.isVisibleTo(dialogue)


def test_le_passage_en_phase_2_fige_la_premiere_barre(application):
    """La phase 1 doit se terminer à 100 %, jamais rester en suspens.

    Sinon l'utilisateur voit « 90 % » figé pendant toute l'installation, et ne
    sait pas si le téléchargement est fini ou bloqué.
    """
    dialogue = DialogueInstallation("RTX 4070 SUPER", 2_602_755_161)
    dialogue.maj(
        tl.Progression(
            tl.PHASE_INSTALLATION, "Décompression des fichiers", 1, 13, 0, -1
        )
    )
    assert dialogue.phase_telechargement.barre.value() == 100
    assert dialogue.phase_installation.isVisibleTo(dialogue)
    # Le repli CPU disparaît : il est trop tard pour changer de variante.
    assert dialogue.btn_cpu.isEnabled() is False


def test_la_phase_1_affiche_octets_vitesse_et_reste(application):
    """Le détail de la phase 1 doit porter la vitesse et le temps restant.

    « 1,2 Go / 2,4 Go — 50 % » ne dit pas si ça avance à 2 Mo/s ou 200 Mo/s, et
    ces deux situations demandent des décisions opposées.
    """
    dialogue = DialogueInstallation("RTX 4070 SUPER", 2_602_755_161)
    dialogue.maj(
        tl.Progression(
            tl.PHASE_TELECHARGEMENT,
            "Téléchargement de torch",
            fait=1_301_377_580,
            total=2_602_755_161,
            vitesse=20 * 1024**2,
            restant=1_301_377_581,
        )
    )
    detail = dialogue.phase_telechargement.detail.text()
    # `int()` tronque : 1301377580 / 2602755161 = 49,998 %, donc 49 %.
    assert "49 %" in detail
    assert "Mo/s" in detail
    assert "reste" in detail


def test_un_total_inconnu_n_invente_pas_de_pourcentage(application):
    """Sans total, on n'affiche AUCUN pourcentage.

    La préparation de Python n'a pas de total. Un pourcentage calculé sur une
    base fausse se remarque immédiatement, et une barre qui avance sans base
    inspire moins confiance qu'une barre qui ne bouge pas.
    """
    dialogue = DialogueInstallation("RTX 4070 SUPER", 2_602_755_161)
    dialogue.maj(
        tl.Progression(
            phase=tl.PHASE_TELECHARGEMENT,
            texte="Préparation de Python",
            fait=0,
            total=0,
            vitesse=0,
            restant=-1,
        )
    )
    detail = dialogue.phase_telechargement.detail.text()
    assert "%" not in detail
    assert "Préparation de Python" in detail


def test_la_phase_2_compte_des_paquets_et_le_dit(application):
    """`uv` ne dit pas combien d'octets il écrit : on compte des paquets.

    Et on l'affiche en paquets, pour que l'utilisateur ne cherche pas une
    contrepartie en Go qui n'existe pas.
    """
    dialogue = DialogueInstallation("RTX 4070 SUPER", 2_602_755_161)
    dialogue.maj(tl.Progression(
        phase=tl.PHASE_INSTALLATION, texte="Décompression", fait=3, total=13, vitesse=0, restant=-1
    ))
    detail = dialogue.phase_installation.detail.text()
    assert "3 / 13 paquets" in detail
    assert "23 %" in detail


# ---------------------------------------------------------------------------
# 4. Les trois formes de l'indicateur
# ---------------------------------------------------------------------------


def test_l_indicateur_nomme_la_version_et_la_carte():
    """`CUDA 2.14.1 — NVIDIA GeForce RTX 4070 SUPER`, la forme demandée."""
    etat = EtatTorch(version="2.14.1", cuda=True, gpu="NVIDIA GeForce RTX 4070 SUPER")
    assert texte_indicateur(etat) == "CUDA 2.14.1 — NVIDIA GeForce RTX 4070 SUPER"


def test_l_indicateur_annonce_le_telechargement_necessaire():
    """GPU présent mais torch absent : il faut le dire clairement.

    C'est le cas que les deux autres formes ne couvrent pas : l'utilisateur a un
    GPU, torch n'est pas installé, et l'analyse ne marchera pas tant qu'il n'a
    pas téléchargé. « CPU uniquement » serait un mensonge.
    """
    etat = EtatTorch(version="", cuda=False, gpu="NVIDIA GeForce RTX 4070 SUPER")
    assert texte_indicateur(etat) == "CUDA absent — téléchargement nécessaire"


def test_l_indicateur_donne_la_version_en_mode_cpu():
    """`CPU uniquement (torch 2.14.1)` : la version reste visible, elle informe."""
    etat = EtatTorch(version="2.14.1", cuda=False, gpu="")
    assert texte_indicateur(etat) == "CPU uniquement (torch 2.14.1)"


def test_l_indicateur_ne_pretend_jamais_calculer_sur_le_gpu_sans_torch():
    """Aucun texte ne doit nommer la carte quand torch n'est pas installé.

    Le risque n'est pas théorique : `EtatTorch(gpu="RTX 4070")` décrit une
    machine qui a une carte, et il serait tentant d'afficher « CUDA — RTX
    4070 » en ignorant `version`. Ce serait promettre un calcul GPU inexistant.
    """
    etat = EtatTorch(version="", cuda=False, gpu="NVIDIA GeForce RTX 4070 SUPER")
    assert "4070" not in texte_indicateur(etat)


def test_un_probleme_est_signale_dans_le_texte():
    """Après un échec, le texte doit renvoyer vers le message, pas rester muet."""
    etat = EtatTorch(version="", cuda=False, gpu="", probleme="Serveur indisponible.")
    assert "message" in texte_indicateur(etat).lower()


# ---------------------------------------------------------------------------
# 5. Formatage
# ---------------------------------------------------------------------------


def test_les_tailles_sont_lisibles_par_un_humain():
    """`2602753280` -> `2,4 Go`, avec une VIRGULE décimale.

    L'utilisateur compare cette taille à ce qui reste sur son disque ; il ne
    fait pas la conversion de tête, et un point décimal dans une fenêtre
    française se lit comme une faute de frappe.
    """
    assert _formater(2602753280) == "2,4 Go"
    assert _formater(124096268) == "118,3 Mo"  # 124 096 268 / 2^20
    assert _formater(0) == "taille inconnue"