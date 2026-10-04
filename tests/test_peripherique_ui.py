"""Tests d'interface du sélecteur de périphérique et de l'indicateur d'état.

L'indicateur de la barre de statut a une contrainte que les autres labels
n'ont pas : il doit être VRAI. Un « Calcul : CPU » pendant que l'analyse tourne
sur le GPU est pire que pas d'indicateur du tout — c'est exactement le
mensonge que l'opérateur seek à éviter. Ces tests vérifient donc les deux
choses en même temps : ce qui est affiché, et le `device` qui part dans
`charger_modele`.
"""

import pytest

pytest.importorskip("PySide6")

from compteur.config import (  # noqa: E402
    PERIPHERIQUE_AUTO,
    PERIPHERIQUE_CPU,
    PERIPHERIQUE_CUDA,
    Config,
    mode_peripherique,
)
from compteur.detecteur import DEVICE_CUDA, LIBELLE_CPU  # noqa: E402
from interface.app import FenetrePrincipale  # noqa: E402
from interface.panneau_reglages import AIDE, PanneauReglages  # noqa: E402


@pytest.fixture
def application():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def fenetre(application):
    return FenetrePrincipale()


# -- Le panneau ----------------------------------------------------------


def test_le_panneau_propose_les_trois_choix(application):
    """Les trois modes doivent être sélectionnables, pas seulement `auto`."""
    p = PanneauReglages(Config())
    choix = [
        p._widgets["peripherique"].itemText(i)
        for i in range(p._widgets["peripherique"].count())
    ]
    assert len(choix) == 3
    assert any("GPU" in c for c in choix)
    assert any("CPU" in c for c in choix)


def test_le_panneau_refletle_le_reglage_de_la_config(application):
    p = PanneauReglages(Config(peripherique=PERIPHERIQUE_CPU))
    assert p.lire().peripherique == PERIPHERIQUE_CPU


def test_changer_choix_dans_le_panneau_change_la_config(application):
    p = PanneauReglages(Config())
    p.definir_peripherique(PERIPHERIQUE_CUDA)
    assert p.lire().peripherique == PERIPHERIQUE_CUDA
    p.definir_peripherique(PERIPHERIQUE_CPU)
    assert p.lire().peripherique == PERIPHERIQUE_CPU


def test_definir_peripherique_ne_bloque_pas_sur_le_deuxieme_appel(application):
    """Poser deux fois la même valeur ne doit pas émettre une config fausse.

    `setCurrentIndex` sur un index déjà sélectionné n'émet rien ; le setter ne
    doit donc rien fabriquer. Un réglage qui réécrivait sa propre valeur ferait
   INX pendulumer l'interface entre deux reconstructions de modèle.
    """
    p = PanneauReglages(Config())
    p.definir_peripherique(PERIPHERIQUE_CPU)
    avant = p.lire().peripherique
    p.definir_peripherique(PERIPHERIQUE_CPU)
    assert p.lire().peripherique == avant == PERIPHERIQUE_CPU


def test_un_mode_inconnu_ne_casse_pas_la_traduction(application):
    p = PanneauReglages(Config())
    p.definir_peripherique("rocm")
    assert p.lire().peripherique == PERIPHERIQUE_AUTO


def test_appliquer_repousse_le_peripherique(application):
    """Aller-retour : pousser une config doit restituer la même valeur."""
    p = PanneauReglages(Config())
    p.appliquer(Config(peripherique=PERIPHERIQUE_CUDA))
    assert p.lire().peripherique == PERIPHERIQUE_CUDA


def test_un_profil_enregistre_conserve_le_peripherique(application, tmp_path):
    p = PanneauReglages(Config())
    p.definir_peripherique(PERIPHERIQUE_CPU)
    chemin = p.enregistrer_profil("cpu-seul", dossier=tmp_path)
    assert Config.depuis_fichier(chemin).peripherique == PERIPHERIQUE_CPU


# -- L'absence d'explication ---------------------------------------------


def test_le_reglage_na_plus_dexplication(application):
    """Retiré à la demande de l'opérateur : un pavé sur le GPU n'aide personne.

    L'information utile — le périphérique réellement utilisé — est écrite en
    permanence dans la barre de statut, sous le compteur. Une bulle de douze
    lignes qu'on ne voit qu'au survol ne peut que la concurrencer.
    """
    assert "peripherique" not in AIDE


def test_le_panneau_construit_le_reglage_sans_encart_daide(application):
    """Le réglage doit SURVIVRE à la suppression de son explication.

    On n'a retiré que le texte : le choix CPU/GPU reste dans le panneau, avec
    son étiquette, parce que c'est le remède quand un GPU plante sur le terrain.
    """
    p = PanneauReglages(Config())
    w = p._widgets["peripherique"]
    assert w.count() == 3
    assert not w.toolTip()


# -- L'indicateur de la barre de statut ----------------------------------


def test_l_indicateur_existe_des_la_construction(fenetre):
    """Il doit être là SANS avoir cliqué sur quoi que ce soit.

    C'est la réclamation d'origine : l'opérateur veut une réponse à l'écran,
    pas après avoir lancé une analyse.
    """
    assert fenetre.label_peripherique.text().startswith("Calcul : ")


def test_l_indicateur_nomme_la_carte_quand_il_y_en_a_une(fenetre):
    texte = fenetre.label_peripherique.text()
    cuda = fenetre.device_effectif().startswith("cuda")
    if cuda:
        assert "CUDA" in texte
    else:
        assert LIBELLE_CPU in texte


def test_l_indicateur_et_le_device_disent_la_meme_chose(fenetre, monkeypatch):
    """Ce qui est affiché et ce qui est utilisé doivent coïncider."""
    monkeypatch.setattr("compteur.detecteur.cuda_disponible", lambda: False)
    fenetre._maj_peripherique()
    assert LIBELLE_CPU in fenetre.label_peripherique.text()
    assert fenetre.device_effectif() == "cpu"

    monkeypatch.setattr("compteur.detecteur.cuda_disponible", lambda: True)
    monkeypatch.setattr("compteur.detecteur.nom_cuda", lambda: "NVIDIA GeForce RTX 4070 SUPER")
    fenetre._maj_peripherique()
    assert "RTX 4070" in fenetre.label_peripherique.text()
    assert fenetre.device_effectif() == DEVICE_CUDA


def test_l_indicateur_suit_le_reglage_du_panneau(fenetre, monkeypatch):
    """Passer le panneau sur « CPU uniquement » doit changer l'affichage."""
    monkeypatch.setattr("compteur.detecteur.cuda_disponible", lambda: True)
    monkeypatch.setattr("compteur.detecteur.nom_cuda", lambda: "NVIDIA GeForce RTX 4070 SUPER")
    fenetre._maj_peripherique()
    assert fenetre.device_effectif() == DEVICE_CUDA

    fenetre.panneau.definir_peripherique(PERIPHERIQUE_CPU)
    assert fenetre.config.peripherique == PERIPHERIQUE_CPU
    assert LIBELLE_CPU in fenetre.label_peripherique.text()
    assert fenetre.device_effectif() == "cpu"


def test_l_indicateur_survit_aux_changements_de_statut(fenetre):
    """`label_statut` est écrasé à chaque état ; l'indicateur, non.

    C'est la raison d'être d'un label dédié : l'information doit rester lisible
    pendant l'analyse, la pause et les messages d'erreur.
    """
    avant = fenetre.label_peripherique.text()
    fenetre.label_statut.setText("Analyse en cours…")
    assert fenetre.label_peripherique.text() == avant
    fenetre.label_statut.setText("En pause.")
    assert fenetre.label_peripherique.text() == avant


def test_l_indicateur_ne_mentionne_pas_de_device_brut(fenetre):
    """« cuda:0 » à l'écran serait du jargon de développeur."""
    assert "cuda:0" not in fenetre.label_peripherique.text()


def test_changer_de_peripherique_invalide_le_detecteur(fenetre, monkeypatch):
    """Un modèle monté reste sur son périphérique d'origine.

    `Detecteur` ne déplace pas un modèle à chaud : sans ce lâcher, l'écran
    annoncerait « CPU » pendant que l'analyse tourne encore sur le GPU —
    l'inverse exact du problème que l'indicateur sert à résoudre.
    """
    monkeypatch.setattr("compteur.detecteur.cuda_disponible", lambda: True)
    monkeypatch.setattr("compteur.detecteur.nom_cuda", lambda: "NVIDIA GeForce RTX 4070 SUPER")
    fenetre._maj_peripherique()
    fenetre.detecteur = object()
    fenetre._modele_charge = "medium.pt"

    fenetre.panneau.definir_peripherique(PERIPHERIQUE_CPU)
    assert fenetre.detecteur is None
    assert fenetre._modele_charge == ""


def test_un_reglage_ordinaire_ne_recharge_pas_le_modele(fenetre):
    """Le relâchement du détecteur est réservé au périphérique et au modèle."""
    fenetre.detecteur = object()
    fenetre._modele_charge = "medium.pt"
    fenetre.panneau.definir_seuil(0.5)
    assert fenetre.detecteur is not None
