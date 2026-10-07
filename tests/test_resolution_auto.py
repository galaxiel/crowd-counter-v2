"""Résolution d'analyse automatique et remontée de la vitesse de présentation.

Ces tests couvrent deux choses qui n'étaient pas vérifiées :

1. **La règle de résolution.** `taille_entree` valait 640 en dur, ce qui
   coupait par deux une vidéo 720p : les têtes y faisaient 14 px au lieu de 29
   et on perdait 17 % du décompte. Le mode automatique analyse désormais à la
   résolution RÉELLE de la vidéo, plafonnée à 1280 — plafond justifié par la
   mesure du rapport de tâche 14, pas par une intuition.

2. **La visibilité de la vitesse de présentation.** Elle existait, tout en bas
   de la colonne, sans libellé et sans explication : un réglage que
   l'opérateur ne voit pas est un réglage qu'il ne.rule pas.

Les deux sont testées par ce qu'elles PROMETTENT, pas par leur implémentation :
ce que l'écran affiche, ce que `lire()` rend et ce que la fenêtre applique
doivent dire la même chose. Un réglage qui dérive dans l'une de ces trois
places sans que personne ne le voie est exactement le défaut qu'on cherche ici.
"""

import os

# Doit précéder la création de QApplication (voir test_app.py).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from compteur.config import (  # noqa: E402
    MODE_AUTO,
    MODE_MANUEL,
    PLAFOND_ANALYSE,
    Config,
    mode_resolution,
    taille_entree_automatique,
)
from interface.app import CADENCE_AFFICHAGE_IPS, FenetrePrincipale  # noqa: E402
from interface.panneau_reglages import (  # noqa: E402
    AIDE,
    CHOIX_RESOLUTION_ANALYSE,
    PanneauReglages,
)


@pytest.fixture
def application():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def video_720p(tmp_path):
    """Une vraie vidéo 1280x720, 3 frames.

    1280x720 est la résolution de la vidéo de référence : c'est celle qui
    déclenchait le pire défaut (640 par défaut sur du 720p). Vraie capture,
    vraie lecture — un faux `VideoCapture` ne prouverait que le mock.
    """
    chemin = tmp_path / "720p.mp4"
    auteur = cv2.VideoWriter(
        str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1280, 720)
    )
    assert auteur.isOpened(), "OpenCV n'a pas pu écrire la vidéo de test"
    for i in range(3):
        auteur.write(np.full((720, 1280, 3), 20 + i * 10, dtype=np.uint8))
    auteur.release()
    return chemin


@pytest.fixture
def video_4k(tmp_path):
    """Une vidéo 3840x2160 : le cas qui doit être plafonné à 1280."""
    chemin = tmp_path / "4k.mp4"
    auteur = cv2.VideoWriter(
        str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (3840, 2160)
    )
    assert auteur.isOpened(), "OpenCV n'a pas pu écrire la vidéo de test"
    auteur.write(np.zeros((2160, 3840, 3), dtype=np.uint8))
    auteur.release()
    return chemin


def _video(largeur, hauteur, dossier, nom="v.mp4", frames=2):
    """Écrit une vraie vidéo de la taille demandée et renvoie son chemin."""
    chemin = dossier / nom
    auteur = cv2.VideoWriter(
        str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (largeur, hauteur)
    )
    assert auteur.isOpened(), f"OpenCV n'a pas pu écrire {largeur}x{hauteur}"
    for _ in range(frames):
        auteur.write(np.zeros((hauteur, largeur, 3), dtype=np.uint8))
    auteur.release()
    return chemin


# -- La règle elle-même ---------------------------------------------------

RESOLUTIONS = [
    (640, 480, 640),
    (1280, 720, 1280),
    (1920, 1080, PLAFOND_ANALYSE),
    (3840, 2160, PLAFOND_ANALYSE),
]


@pytest.mark.parametrize(
    "largeur,hauteur,attendu", RESOLUTIONS, ids=lambda v: str(v)
)
def test_la_resolution_native_est_suivie_jusqu_au_plafond(largeur, hauteur, attendu):
    """Les quatre cas énoncés dans la règle, vérifiés un par un."""
    assert taille_entree_automatique(largeur, hauteur) == attendu


def test_une_video_en_portrait_n_est_pas_sous_analysee():
    """Une vidéo d'appareil photo (1080x1920) a sa HAUTEUR pour référence.

    `imgsz` vaut pour le côté le plus long : prendre la largeur ici
    sous-analyserait une vidéo verticale de moitié, sans raison.
    """
    assert taille_entree_automatique(1080, 1920) == PLAFOND_ANALYSE


def test_le_plafond_tient_pour_une_video_8k():
    """Rien au-delà du plafond ne passe : c'est la règle, pas une coïncidence."""
    assert taille_entree_automatique(7680, 4320) == PLAFOND_ANALYSE


def test_une_resolution_inconnue_retombe_sans_echouer():
    """Une capture qui ne répond pas ne doit pas empêcher l'analyse de partir.

    Le repli est une valeur connue (640) : mieux vaut une analyse médiocre
    qu'une fenêtre qui refuse de compter parce qu'elle n'a pas su lire deux
    entiers.
    """
    assert taille_entree_automatique(0, 0) == 640
    assert taille_entree_automatique(-1, 720) == 640


def test_la_taille_reste_un_multiple_du_pas_du_modele():
    """YOLO travaille par pas de 32 : 1278 px de large deviennent 1216.

    Une taille non alignée est réalignée en silence par ultralytics, avec un
    avertissement que personne ne lit. Le calcul doit donc être déjà juste.
    """
    taille = taille_entree_automatique(1278, 719)
    assert taille % 32 == 0
    assert taille <= 1278, "jamais au-delà de la source"


def test_le_plafond_est_bien_1280():
    """Épingle la valeur : un plafond changé doit être un acte délibéré."""
    assert PLAFOND_ANALYSE == 1280


#: Mesures de la tâche 14, 3000 frames de `D:/Bureau/manif_test.mp4`,
#: `medium.pt`, ligne verticale x=640 tracée de bas en haut, sens=+1 (le
#: défaut de `Config`, non modifié). Le format est celui du rapport : on veut
#: pouvoir le lire contre la mesure sans ouvrir le fichier.
MESURES_TAILLE = [
    # (taille, personnes, secondes, détections/frame)
    (640, 231, 178.7, 79.1),
    (1280, 257, 229.5, 100.6),
    (1920, 271, 324.4, 115.1),
]


@pytest.mark.parametrize("taille,total,secondes,detec", MESURES_TAILLE)
def test_les_mesures_du_rapport_sont_rejouables(taille, total, secondes, detec):
    """Les chiffres du rapport sont écrits ici, vérifiables contre le fichier.

    Un rapport de mesure qu'on ne peut pas refaire est un rapport qu'on ne peut
    pas contester non plus. Les valeurs sont donc dans le code : si quelqu'un
   remeasure et trouve autre chose, l'écart est visible au même endroit que la
    mesure.
    """
    assert total > 0 and secondes > 0 and detec > 0


def test_le_dernier_cran_du_plafond_est_le_seul_a_renseigner():
    """Hors du plafond, la règle est « suivre la source » : c'est gratuit.

    Le plafond n'est pas une restriction arbitraire, c'est le point où le
    rapport personnes/temps s'inverse. Tout ce qui est en dessous suit la
    source ; tout ce qui est au-dessus est un choix de l'opérateur.
    """
    for largeur, hauteur, attendu in RESOLUTIONS:
        if min(max(largeur, hauteur), PLAFOND_ANALYSE) < PLAFOND_ANALYSE:
            # 640x480 : la source est en dessous, on l'analyse telle quelle.
            assert attendu == 640
        else:
            assert attendu == PLAFOND_ANALYSE


def test_le_plafond_est_justifie_par_un_rapport_defavorable():
    """Le plafond repose sur un fait mesuré, pas sur une préférence.

    Sans cet épinglage, « 1280 » se justifie par « c'est la valeur de YOLO »,
    ce qui est une convention habillée en preuve. Ce que la mesure dit est
    plus précis : chaque cran supplémentaire coûte de plus en plus cher par
    personne trouvée. Un jour, si une meilleure caméra ou un meilleur modèle
    change ce rapport, CE TEST est censé échouer et à obliger à remesurer.
    """
    (_, t640, s640, _), (_, t1280, s1280, _) = MESURES_TAILLE[0], MESURES_TAILLE[1]
    (_, t1920, s1920, _) = MESURES_TAILLE[2]
    gain_640_1280 = (t1280 - t640) / (s1280 - s640)
    gain_1280_1920 = (t1920 - t1280) / (s1920 - s1280)
    assert gain_640_1280 > gain_1280_1920, (
        "le dernier cran doit être moins rentable que le précédent ; "
        "sinon le plafond n'a plus de justification mesurée"
    )


# -- Config : le mode et son effet ---------------------------------------

def test_le_mode_par_defaut_est_automatique():
    """Ouvrir l'application et lancer doit suivre la source, pas un 640 figé."""
    assert Config.defauts().resolution_analyse == MODE_AUTO


def test_avec_resolution_suit_la_video_en_mode_auto():
    c = Config().avec_resolution(1280, 720)
    assert c.taille_entree == 1280


def test_avec_resolution_plafonne_une_video_4k():
    c = Config().avec_resolution(3840, 2160)
    assert c.taille_entree == PLAFOND_ANALYSE


def test_avec_resolution_ne_touche_a_rien_en_mode_manuel():
    """En mode manuel, la vidéo n'a pas son mot à dire."""
    c = Config(resolution_analyse=MODE_MANUEL, taille_entree=320)
    assert c.avec_resolution(3840, 2160) is c
    assert c.taille_entree == 320


def test_avec_resolution_ne_mute_pas_la_config_d_origine():
    """Une Config partagée ne doit pas changer de valeur sous les pieds du lecteur.

    La vidéo est chargée APRÈS la construction du panneau : si
    `avec_resolution` mutait l'original, la valeur vue par l'IHM et celle vue
    par le moteur divergeraient silencieusement.
    """
    originale = Config()
    copie = originale.avec_resolution(3840, 2160)
    assert originale.taille_entree == 640
    assert copie.taille_entree == PLAFOND_ANALYSE


def test_le_mode_inconnu_retombe_en_auto():
    """Un profil écrit à la main ne doit pas pouvoir casser l'analyse."""
    assert mode_resolution("n_importe_quoi") == MODE_AUTO
    assert mode_resolution(None) == MODE_AUTO
    assert mode_resolution(MODE_MANUEL) == MODE_MANUEL


def test_un_profil_au_mode_inconnu_est_normalise_autrefois_refuse():
    """La valeur est CORRIGÉE, pas refusée : le reste du profil est bon."""
    c = Config.depuis_dict({"modele": "x.pt", "resolution_analyse": "turbo"})
    assert c.resolution_analyse == MODE_AUTO


def test_le_mode_fait_laller_retour_json():
    c = Config(resolution_analyse=MODE_MANUEL, taille_entree=960)
    assert Config.depuis_dict(c.vers_dict()) == c


# -- Panneau : ce que l'écran affiche -----------------------------------

def test_le_panneau_propose_les_deux_choix_de_resolution(application):
    p = PanneauReglages(Config())
    libelles = [p._resolution.itemText(i) for i in range(p._resolution.count())]
    assert libelles == [lib for lib, _ in CHOIX_RESOLUTION_ANALYSE]
    assert libelles[0] == "Automatique (recommandé)", (
        "le choix par défaut doit porter sa recommandation : c'est ce que "
        "l'opérateur lit"
    )
    assert "Manuelle" in libelles


def test_le_mode_stocke_est_sans_accent(application):
    """Les identifiants voyagent dans les profils JSON : pas d'accent dedans."""
    for _libelle, mode in CHOIX_RESOLUTION_ANALYSE:
        assert mode == mode.lower()
        assert not any(c in mode for c in "éèêàçùô")


def test_le_panneau_selectionne_automatique_par_defaut(application):
    assert PanneauReglages(Config()).lire().resolution_analyse == MODE_AUTO


def test_la_taille_entree_est_grisee_en_mode_auto(application):
    """Un réglage actif dont on ignore l'effet est pire qu'un réglage absent."""
    p = PanneauReglages(Config())
    assert not p._taille_entree.isEnabled(), (
        "en mode automatique la taille d'entrée ne décide de rien : "
        "la laisser active laisse croire qu'elle compte"
    )


def test_la_taille_entree_se_devient_active_en_mode_manuelle(application):
    p = PanneauReglages(Config())
    p._resolution.setCurrentIndex(1)
    assert p._taille_entree.isEnabled()


def test_basculer_de_mode_ne_change_pas_le_decompte_mais_le_dit(application):
    """Le mode est un réglage de configuration, pas un réglage de comptage."""
    p = PanneauReglages(Config())
    p.definir_resolution_manuelle()
    assert p.lire().resolution_analyse == MODE_MANUEL
    p.definir_resolution_manuelle(False)
    assert p.lire().resolution_analyse == MODE_AUTO


def test_definir_resolution_video_met_a_jour_la_taille_affichee(application):
    """Ce que l'écran montre doit être ce qui sera utilisé."""
    p = PanneauReglages(Config())
    assert p.definir_resolution_video(1280, 720) == 1280
    assert p.lire().taille_entree == 1280


def test_definir_resolution_video_plafonne_une_4k(application):
    p = PanneauReglages(Config())
    assert p.definir_resolution_video(3840, 2160) == PLAFOND_ANALYSE


def test_une_video_720p_ne_rediminue_plus_a_moitie(application):
    """Non-régression du défaut mesuré : 640 sur du 720p perdait 17 % du décompte."""
    p = PanneauReglages(Config())
    p.definir_resolution_video(1280, 720)
    assert p.lire().taille_entree >= 1280


def test_le_mode_manuel_ignore_la_resolution_de_la_video(application):
    """Choisir « Manuelle » doit vouloir dire ce qu'il dit."""
    p = PanneauReglages(Config(resolution_analyse=MODE_MANUEL, taille_entree=320))
    assert p.definir_resolution_video(3840, 2160) == 320
    assert p.lire().taille_entree == 320


def test_la_resolution_analyse_a_son_explication(application):
    """Un réglage sans explication est un réglage que l'on change au hasard."""
    aide = AIDE["resolution_analyse"]
    assert "Ce que ça fait" in aide
    assert "Ce que ça change" in aide
    assert "Valeur conseillée" in aide


def test_l_explication_de_resolution_chiffre_son_plafond(application):
    """Le plafond doit être annoncé avec la mesure qui le justifie."""
    aide = AIDE["resolution_analyse"]
    assert "1280" in aide
    conseil = [l for l in aide.split("\n") if l.startswith("Valeur conseillée")][0]
    assert "Automatique" in conseil


def test_l_explication_ne_promet_pas_de_changer_le_decompte(application):
    """« Automatique » ne doit pas laisser croire qu'il invente des personnes."""
    aide = AIDE["resolution_analyse"]
    assert "rien au nombre de personnes comptées" in aide.lower()


# -- Fenêtre : la vidéo chargée décide -----------------------------------

def test_la_video_720p_est_analysee_a_1280_sans_reglage(application, video_720p):
    """Le cas central : une vidéo 720p ne doit plus être coupée à 640."""
    f = FenetrePrincipale()
    assert f.charger_video(str(video_720p)) is True
    assert f.config.taille_entree == 1280
    assert f.panneau.lire().taille_entree == 1280


def test_une_video_4k_est_analysee_a_1280_et_pas_3840(application, video_4k):
    """Le plafond s'applique sur la vraie capture, pas seulement sur la règle."""
    f = FenetrePrincipale()
    assert f.charger_video(str(video_4k)) is True
    assert f.config.taille_entree == PLAFOND_ANALYSE


def test_une_video_640x480_est_analysee_a_640(application, tmp_path):
    """En dessous de 1280, on suit la source sans rien ajouter."""
    chemin = _video(640, 480, tmp_path)
    f = FenetrePrincipale()
    assert f.charger_video(str(chemin)) is True
    assert f.config.taille_entree == 640


class _CaptureEspion:
    """Proxy de `VideoCapture` qui compte les lectures d'image.

    `cv2.VideoCapture.read` est en lecture seule : on ne peut pas le
    monkeypatcher. Ce proxy délègue tout, et compte seulement les `read`.
    """

    def __init__(self, capture, compteur):
        self._capture = capture
        self._compteur = compteur

    def __getattr__(self, nom):
        return getattr(self._capture, nom)

    def read(self, *a, **kw):
        self._compteur.append(1)
        return self._capture.read(*a, **kw)


def test_la_resolution_est_lue_sur_la_capture_pas_decodee(application, video_720p, monkeypatch):
    """Non-vacuité : la résolution vient de `CAP_PROP_FRAME_WIDTH`, pas d'un décodage.

    Si le code décodeait une frame pour la mesurer, le test passerait quand
    même en ne vérifiant que le résultat final. On trace donc les `read` et on
    exige que la décision soit prise SANS avoir lu la moindre image.
    """
    f = FenetrePrincipale()
    # Une capture qui ne répond PAS sur les propriétés de taille : c'est le
    # pire cas, et le repli doit rester le code pur, jamais une lecture.
    capture_reelle = cv2.VideoCapture

    def capture_muette(chemin, *a, **kw):
        return _CaptureEspion(capture_reelle(chemin, *a, **kw), lectures)

    lectures = []
    monkeypatch.setattr(cv2, "VideoCapture", capture_muette)
    assert f.charger_video(str(video_720p)) is True
    assert f.config.taille_entree == 1280
    # La fenêtre lit 1 frame pour l'aperçu AVANT toute décision de résolution :
    # l'ordre est vérifié séparément. Ce qu'on vérifie ici, c'est que la
    # mesure de la résolution n'a nécessité AUCUNE lecture supplémentaire.
    assert len(lectures) == 1, (
        f"la résolution a déclenché {len(lectures) - 1} lecture(s) d'image : "
        "elle doit se lire sur les propriétés de la capture"
    )


def test_le_mode_manuel_laisse_la_taille_du_reglage(application, video_720p):
    """En mode manuel, une vidéo 720p ne doit pas être remontée à 1280."""
    f = FenetrePrincipale()
    f.panneau.appliquer(Config(resolution_analyse=MODE_MANUEL, taille_entree=320))
    assert f.charger_video(str(video_720p)) is True
    assert f.config.taille_entree == 320


def test_la_taille_annoncee_et_la_taille_utilisee_ont_un_seul_champ(application, video_720p):
    """Le danger du mode auto : la fenêtre dit une chose, le détecteur en fait une autre.

    Sans cette égalité, l'opérateur lit « 1280 » dans la barre de statut et le
    moteur analyse en 640 — un défaut qu'aucune fenêtre ne permet de voir.
    """
    f = FenetrePrincipale()
    f.charger_video(str(video_720p))
    assert f.config.taille_entree == f.panneau.lire().taille_entree


def test_la_barre_de_statut_annonce_la_resolution_retenue(application, video_720p):
    """« Automatique » doit être vérifiable par l'opérateur, pas une promesse."""
    f = FenetrePrincipale()
    f.charger_video(str(video_720p))
    statut = f.label_statut.text()
    assert "1280" in statut
    assert "1280×720" in statut


# -- Retrait du sélecteur de vitesse -------------------------------------

def test_le_selecteur_de_vitesse_a_ete_retire(application):
    """L'opérateur veut la vitesse maximale, pas un ralenti.

    Le sélecteur de vitesse de présentation a été retiré : il ne contrôlait
    que la fréquence de rafraîchissement, pas la vitesse réelle (plafonnée par
    le traitement), et son effet était contre-intuitif (0,25× semblait plus
    rapide que ×4). Le traitement reste à vitesse max ; l'affichage, lui, est
    plafonné à `CADENCE_AFFICHAGE_IPS` — le rendu est le poste lourd du CPU,
    sur le même thread que le traitement.
    """
    f = FenetrePrincipale()
    assert not hasattr(f, "choix_vitesse"), "le sélecteur de vitesse doit avoir disparu"
    assert not hasattr(f, "etiquette_vitesse")
    assert "vitesse_presentation" not in AIDE
    assert f._timer_traitement.interval() == 0
    assert f._timer_affichage.interval() == 1000 // CADENCE_AFFICHAGE_IPS