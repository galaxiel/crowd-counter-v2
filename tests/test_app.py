"""Tests de la fenêtre principale (tâche 10).

Trois choses sont vérifiées ici, par ordre de gravité.

1. **La fenêtre se construit et ne bloque pas.** Aucun modèle n'est chargé à
   l'ouverture : le chargement se fait sur « Lancer », jamais dans
   `__init__`. Un `FenetrePrincipale()` doit donc être instantané même avec
   CUDA actif.

2. **Le tracé de la ligne.** C'est la fonction critique : deux clics posés à la
   souris sur la vidéo deviennent les deux extrémités de la ligne que le moteur
   va comptabiliser. Une erreur ici est INVISIBLE à l'écran — la ligne affichée
   paraît bien posée, c'est le décompte qui est faux. Les tests portent donc sur
   la ligne *source* obtenue, et sur la géométrie qui en découle (verticalité,
   abscisse, côtés de part et d'autre de la bande).

3. **Le découplage traitement / présentation.** Ralentir l'affichage ne doit
   JAMAIS ralentir le traitement : deux minuteries distinctes, et c'est
   vérifié explicitement plutôt que supposé.

Une note sur les boîtes de dialogue : l'interface n'ouvre une `QMessageBox`
MODALE que si la fenêtre est visible et que Qt n'est pas en mode `offscreen`.
Sans cette garde, un test qui provoke une erreur resterait bloqué sur un
« OK » que personne ne peut cliquer.
"""

import os

# Doit précéder la création de QApplication (voir test_widgets_video.py).
# `setdefault` respecte un choix déjà pris dans l'environnement de lancement.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from compteur.types import Evenement, FrameResult  # noqa: E402
from interface.app import AVERTISSEMENT_SENS, FenetrePrincipale  # noqa: E402

# Cadence de la vidéo synthétique : 25 i/s. Les tests d'affichage en dépendent
# (0,25x -> 160 ms entre deux images affichées).
FPS_SYNTHETIQUE = 25.0


@pytest.fixture
def application():
    """Une QApplication unique pour toute la session (Qt n'en autorise qu'une)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def video_synthetique(tmp_path):
    """Une vidéo minuscule (64x48, 5 frames) écrite par OpenCV.

    Vraie vidéo, vraie `VideoCapture` : le test du chargement et de la lecture
    exercise le même chemin de code qu'une vidéo de manifestation, sans
    dépendre d'un fichier de several gigaoctets absent du dépôt.
    """
    chemin = tmp_path / "synthetique.mp4"
    auteur = cv2.VideoWriter(
        str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), FPS_SYNTHETIQUE, (64, 48)
    )
    assert auteur.isOpened(), "OpenCV n'a pas pu écrire la vidéo de test"
    for i in range(5):
        auteur.write(np.full((48, 64, 3), 20 + i * 10, dtype=np.uint8))
    auteur.release()
    return chemin


@pytest.fixture
def fenetre(application):
    return FenetrePrincipale()


def cliquer(widget, x, y):
    """Simule un clic gauche en coordonnées WIDGET."""
    captures = []
    widget.clic.connect(lambda px, py: captures.append((px, py)))
    evenement = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, y),
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(evenement)
    return captures


def avancer(fenetre, n=1):
    """Traite puis affiche ``n`` frames.

    Les deux temps sont volontairement distincts dans le code : c'est ce
    découplage que la vitesse de présentation ne doit pas casser. Les tests
    d'affichage les enchaînent donc explicitement, au lieu d'attendre qu'un
    seul appel fasse les deux.
    """
    for _ in range(n):
        fenetre._tick()
        fenetre._afficher()


class FauxCompteur:
    """Compteur de test : aucun modèle, aucun GPU, un événement en frame 2."""

    def __init__(self) -> None:
        self.ligne = None
        self.appels = 0
        self.evenements = []

    def ajuster_ligne(self, ligne) -> None:
        self.ligne = ligne

    def traiter_frame(self, img, index, ts) -> FrameResult:
        self.appels += 1
        nouveaux = []
        if index == 2:
            ev = Evenement(frame=index, timestamp_s=ts, x=1.0, y=1.0, track_id=7)
            nouveaux.append(ev)
            self.evenements.append(ev)
        return FrameResult(
            image=img,
            total=index + 1,
            presents=2,
            frame_index=index,
            timestamp_s=ts,
            evenements=nouveaux,
        )

    def resultat(self, modele, nb_frames, secondes):
        from compteur.types import Resultat

        return Resultat(
            total=nb_frames,
            evenements=list(self.evenements),
            modele=modele,
            nb_frames=nb_frames,
            presents_max=3,
            secondes=secondes,
        )


# -- Construction --------------------------------------------------------


def test_fenetre_se_construit(application):
    f = FenetrePrincipale()
    assert f.compteur_total == 0
    assert f.presents == 0
    assert f.video_finiment is None
    assert f.ligne is None
    assert f.compteur is None


def test_fenetre_a_le_titre_attendu(application):
    """Le contrôleur lance `python -c ... ; print(f.windowTitle())`."""
    f = FenetrePrincipale()
    assert f.windowTitle() == "Compteur de manifestation"


def test_aucun_chargement_de_modele_a_l_ouverture(application, monkeypatch):
    """Construire la fenêtre ne doit toucher ni torch ni ultralytics.

    CUDA est actif sur cette machine : un chargement de modèle dans `__init__`
    gellerait l'interface avant même d'afficher quoi que ce soit.
    """

    def exploser(*_args, **_kwargs):
        raise AssertionError("le modèle ne doit pas être chargé au démarrage")

    monkeypatch.setattr("interface.app.charger_modele", exploser)
    monkeypatch.setattr("interface.app.Detecteur", exploser)
    FenetrePrincipale()


def test_fenetre_a_le_compteur_et_le_widget_video(application):
    f = FenetrePrincipale()
    assert f.label_compteur.objectName() == "compteur"
    assert f.label_compteur.text() == "0"
    assert f.video is not None


# -- Chargement de la vidéo ---------------------------------------------


def test_charger_video_absente_signale_une_erreur(application, tmp_path):
    f = FenetrePrincipale()
    assert f.charger_video(str(tmp_path / "inexistant.mp4")) is False
    assert f.video_finiment is None


def test_charger_fichier_non_video_signale_une_erreur(application, tmp_path):
    """Un fichier présent mais illisible doit être refusé aussi."""
    faux = tmp_path / "pas_une_video.mp4"
    faux.write_text("ceci n'est pas une vidéo", encoding="utf-8")
    f = FenetrePrincipale()
    assert f.charger_video(str(faux)) is False


def test_video_finiment_renseigne_apres_chargement(application, fenetre, video_synthetique):
    assert fenetre.charger_video(str(video_synthetique)) is True
    assert fenetre.video_finiment == video_synthetique
    assert fenetre.cap is not None


def test_charger_video_affiche_la_premiere_frame(application, fenetre, video_synthetique):
    """L'opérateur doit voir la scène AVANT de tracer sa ligne."""
    fenetre.charger_video(str(video_synthetique))
    assert fenetre.video._image is not None
    assert fenetre.video._image.shape[:2] == (48, 64)


def test_charger_video_reinitialise_le_compteur(application, fenetre, video_synthetique):
    """Une nouvelle vidéo repart de zéro, même après un comptage partiel."""
    fenetre.charger_video(str(video_synthetique))
    fenetre.maj_compteurs(total=42, presents=7, frames=100)
    fenetre.charger_video(str(video_synthetique))
    assert fenetre.compteur_total == 0
    assert fenetre.presents == 0


def test_charger_video_efface_la_ligne_de_la_video_precedente(
    application, fenetre, video_synthetique
):
    """La ligne est en pixels de l'UNE vidéo : la conserver serait compter faux.

    Une ligne gardée après changement de vidéo traverserait l'image au même
    endroit numérique, donc potentiellement à un autre endroit réel. Le défaut
    ne serait visible d'aucune façon à l'écran.
    """
    fenetre.charger_video(str(video_synthetique))
    fenetre.definir_ligne((10.0, 5.0), (10.0, 40.0))
    fenetre.charger_video(str(video_synthetique))
    assert fenetre.ligne is None
    assert not fenetre.btn_lancer.isEnabled()


# -- Boutons : rien de lancable sans vidéo ET sans ligne ------------------


def test_lancer_desactive_tant_qu_aucune_video_nest_chargee(application, fenetre):
    assert not fenetre.btn_lancer.isEnabled()
    assert not fenetre.btn_ligne.isEnabled()
    assert not fenetre.btn_pause.isEnabled()
    assert not fenetre.btn_stop.isEnabled()
    assert not fenetre.btn_export.isEnabled()


def test_lancer_desactive_tant_que_la_ligne_nest_pas_tracee(
    application, fenetre, video_synthetique
):
    fenetre.charger_video(str(video_synthetique))
    assert fenetre.btn_ligne.isEnabled()
    assert not fenetre.btn_lancer.isEnabled()


def test_lancer_actif_apres_une_vide_et_une_ligne(
    application, fenetre, video_synthetique
):
    fenetre.charger_video(str(video_synthetique))
    fenetre._on_tracer_ligne()
    fenetre._on_clic_video(10, 2)
    fenetre._on_clic_video(10, 40)
    assert fenetre.btn_lancer.isEnabled()


def test_retirer_la_ligne_desactive_a_nouveau_lancer(application, fenetre):
    """« Tracer la ligne » réarme : l'ancien tracé ne doit plus rester actif."""
    fenetre.definir_ligne((10.0, 5.0), (10.0, 40.0))
    assert fenetre.btn_lancer.isEnabled() or fenetre.cap is None
    fenetre._on_tracer_ligne()
    assert fenetre.ligne is None
    assert not fenetre.btn_lancer.isEnabled()


# -- Tracé de la ligne : la fonction critique ----------------------------


def test_ligne_invalide_refusee(application, fenetre):
    assert fenetre.definir_ligne((10.0, 10.0), (10.0, 10.0)) is False
    assert fenetre.ligne is None


def test_sens_invalide_refuse(application, fenetre):
    config = fenetre.panneau.lire()
    config.sens = 0  # hors de {+1, -1} : `Ligne` doit refuser
    fenetre.config = config
    assert fenetre.definir_ligne((10.0, 5.0), (10.0, 40.0)) is False


def test_deux_clics_donnent_une_ligne_verticale_a_la_bonne_abscisse(
    application, fenetre
):
    """Le geste réel de l'opérateur, de bout en bout.

    Géométrie connue par construction, donc aucun Repère circulaire : image
    640x480 dans un widget 640x480, facteur 1, décalage nul — les coordonnées
    du clic ET celles de la ligne source sont les mêmes nombres.

    On vérifie ensuite la LIGNE, pas seulement les points : verticale à
    x=300, et surtout les deux personnes voisines de part et d'autre de la
    bande sont classées de côtés OPPOSÉS. C'est cette propriété que le moteur
    utilise pour compter ; une ligne décalée d'un pixel la ferait échouer.
    """
    f = fenetre
    f.video.setMinimumSize(0, 0)
    f.video.resize(640, 480)
    f.video.definir_image(np.zeros((480, 640, 3), dtype=np.uint8))
    assert f.video.facteur_echelle() == pytest.approx(1.0)
    assert f.video.decalage() == (0, 0)

    f._on_tracer_ligne()
    assert cliquer(f.video, 300, 20) == [(300, 20)]
    assert cliquer(f.video, 300, 460) == [(300, 460)]

    assert f.ligne is not None
    assert f.ligne.p1 == (300.0, 20.0)
    assert f.ligne.p2 == (300.0, 460.0)
    assert f.ligne.p1[0] == f.ligne.p2[0], "la ligne doit être verticale"

    # Le point visé est SUR la ligne, à 0 près.
    assert f.ligne.coordonnee_projetee((300, 240)) == pytest.approx(0.0, abs=1e-6)

    # La bande par défaut fait 30 px, donc ±15 px : à ±10 px on est encore
    # DANS la bande, et `point_du_cote` doit y renvoyer 0. Une personne qui
    # n'est pas encore franche du tout n'est comptée ni dans un sens ni dans
    # l'autre — c'est ce qui empêche les oscillations de déclencher un
    # comptage (voir le verrou anti-rebond de `compteur.compteur`).
    assert f.ligne.epaisseur == 30
    assert f.ligne.point_du_cote((290, 240)) == 0
    assert f.ligne.point_du_cote((310, 240)) == 0

    # Juste dehors, de part et d'autre : les côtés sont OPPOSÉS et non nuls.
    # C'est la propriété dont le moteur se sert pour décider d'un franchissement.
    assert f.ligne.point_du_cote((280, 240)) != 0
    assert f.ligne.point_du_cote((320, 240)) != 0
    assert f.ligne.point_du_cote((280, 240)) != f.ligne.point_du_cote((320, 240))

    # Et le franchissement du bon côté vers l'autre est retenu une fois.
    assert f.ligne.a_traverse((280, 240), (320, 240)) is True


def test_deux_clics_avec_reduction_donnent_la_bonne_ligne_source(
    application, fenetre
):
    """Image 1280x960 réduite de moitié dans un widget 640x480.

    Un clic au quart de la largeur doit viser le pixel source 320, pas 160 :
    c'est le régime où la ligne part le plus facilement à côté.
    """
    f = fenetre
    f.video.setMinimumSize(0, 0)
    f.video.resize(640, 480)
    f.video.definir_image(np.zeros((960, 1280, 3), dtype=np.uint8))
    assert f.video.facteur_echelle() == pytest.approx(0.5)

    f._on_tracer_ligne()
    assert cliquer(f.video, 160, 40) == [(320, 80)]
    assert cliquer(f.video, 160, 440) == [(320, 880)]

    assert f.ligne.p1 == (320.0, 80.0)
    assert f.ligne.p2 == (320.0, 880.0)


def test_clic_sans_arme_ne_pose_aucun_point(application, fenetre, video_synthetique):
    """Un clic parasite ne doit pas commencer un tracé.

    Pendant une manifestation, un clic malencontreux au milieu d'un réglage
    décalerait la ligne du premier point venu : le défaut serait invisible et le
    décompte faux.
    """
    f = fenetre
    f.charger_video(str(video_synthetique))
    f._on_clic_video(10, 2)
    assert f.ligne is None
    assert f._points_ligne == []


def test_premier_point_pose_ne_fait_pas_de_ligne(application, fenetre):
    f = fenetre
    f.video.setMinimumSize(0, 0)
    f.video.resize(320, 240)
    f.video.definir_image(np.zeros((240, 320, 3), dtype=np.uint8))
    f._on_tracer_ligne()
    cliquer(f.video, 100, 20)
    assert f.ligne is None
    assert f._points_ligne == [(100.0, 20.0)]


def test_changer_le_sens_reconstruit_la_ligne_sans_perdre_les_points(
    application, fenetre
):
    """Le réglage du sens doit agir sur la ligne déjà tracée, pas la perdre."""
    f = fenetre
    f.definir_ligne((300.0, 20.0), (300.0, 460.0))
    assert f.ligne.sens == 1
    f.panneau._sens.setCurrentIndex(1)  # « Après → avant »
    assert f.config.sens == -1
    assert f.ligne.sens == -1
    assert f.ligne.p1 == (300.0, 20.0)
    assert f.ligne.p2 == (300.0, 460.0)


def test_changer_l_epaisseur_de_bande_reconstruit_la_ligne(application, fenetre):
    f = fenetre
    f.definir_ligne((300.0, 20.0), (300.0, 460.0))
    f.panneau.definir_epaisseur_bande(80)
    assert f.ligne.epaisseur == 80


# -- Compteurs -----------------------------------------------------------


def test_compteur_se_met_a_jour(application):
    f = FenetrePrincipale()
    f.maj_compteurs(total=42, presents=7, frames=100)
    assert f.compteur_total == 42
    assert f.presents == 7
    assert f.frames == 100
    assert f.label_compteur.text() == "42"
    # Les compteurs de « présents » et de « frames » ne sont PLUS affichés :
    # les valeurs restent enregistrées ci-dessus, seul le libellé a disparu.
    assert not hasattr(f, "label_details")


def test_compteur_affiche_les_chiffres_en_entier(application, fenetre):
    """Le gros chiffre est lu de loin : il ne doit jamais afficher de décimale."""
    fenetre.maj_compteurs(total=1234, presents=0, frames=0)
    assert fenetre.label_compteur.text() == "1234"


# -- Boucle de lecture ----------------------------------------------------


def test_tick_traite_une_frame_et_incremente_le_compteur(
    application, fenetre, video_synthetique
):
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.compteur = FauxCompteur()
    f._ouvrir_session()
    avancer(f, 1)
    assert f.compteur.appels == 1
    assert f.frames == 1
    assert f.compteur_total == 1
    avancer(f, 2)
    assert f.compteur.appels == 3
    assert f.compteur_total == 3


def test_tick_ne_traite_pas_si_pas_de_video(application, fenetre):
    """Sans capture, `_tick` doit s'arrêter proprement, pas lever."""
    f = fenetre
    f.compteur = FauxCompteur()
    f._ouvrir_session()
    f._tick()
    assert not f._timer_traitement.isActive()


def test_fin_de_video_met_en_pause_et_affiche_le_bilan(
    application, fenetre, video_synthetique
):
    """Une vidéo de 5 frames : au bout, la lecture s'arrête et propose l'export."""
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.compteur = FauxCompteur()
    f._ouvrir_session()
    for _ in range(10):
        avancer(f)
    assert f._en_analyse is False
    assert not f._timer_traitement.isActive()
    assert f.btn_export.isEnabled()
    assert "Terminé" in f.label_statut.text()


def test_flash_de_comptage_s_allume_puis_s_eteint(application, fenetre, video_synthetique):
    """Le flash d'alerte aide l'opérateur à voir le groupe qui passe."""
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.compteur = FauxCompteur()
    f._ouvrir_session()
    avancer(f, 3)  # l'événement arrive en frame 2 -> flash armé
    assert f._flash > 0
    avancer(f, 6)
    assert f._flash == 0


def test_pause_et_reprise(application, fenetre, video_synthetique):
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.compteur = FauxCompteur()
    f._ouvrir_session()
    f._timer_traitement.start()
    f._basculer_pause()
    assert f._en_analyse is False
    assert not f._timer_traitement.isActive()
    assert f.btn_pause.text() == "Reprendre"
    f._basculer_pause()
    assert f._en_analyse is True
    assert f._timer_traitement.isActive()
    assert f.btn_pause.text() == "Pause"


def test_arreter_ne_plante_pas_sans_video(application, fenetre):
    fenetre.arreter()
    assert fenetre._en_analyse is False


def test_fermer_la_fenetre_libere_la_video(application, fenetre, video_synthetique):
    """Une `VideoCapture` non libérée tient le fichier ouvert sous Windows."""
    f = fenetre
    f.charger_video(str(video_synthetique))
    capture = f.cap
    f.close()
    assert not capture.isOpened()


# -- Vitesse de présentation : ne touche jamais au traitement --------------


def test_vitesse_max_laisse_le_traitement_a_zero(application, fenetre, video_synthetique):
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.choix_vitesse.setCurrentText("max")
    assert f._timer_traitement.interval() == 0
    assert f._timer_affichage.interval() == 0


def test_vitesse_lente_ne_ralentit_pas_le_traitement(
    application, fenetre, video_synthetique
):
    """0,25x à 25 i/s = une image affichée toutes les 160 ms.

    Le timer de TRAITEMENT, lui, reste à 0 ms : c'est le GPU qui fixe la
    cadence, pas le curseur de vitesse. Un seul timer (celui du plan) aurait
    ralenti les deux, donc le comptage lui-même.
    """
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.choix_vitesse.setCurrentText("0.25×")
    assert f._timer_affichage.interval() == 160
    assert f._timer_traitement.interval() == 0


def test_vitesse_1x_affiche_a_la_cadence_de_la_video(
    application, fenetre, video_synthetique
):
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.choix_vitesse.setCurrentText("1×")
    assert f._timer_affichage.interval() == 40  # 1000 / 25


def test_vitesse_2x_accelere_l_affichage_sans_rien_casser(
    application, fenetre, video_synthetique
):
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.choix_vitesse.setCurrentText("2×")
    assert f._timer_affichage.interval() == 20
    assert f._timer_traitement.interval() == 0


def test_affichage_ne_redessine_pas_la_meme_frame_deux_fois(
    application, fenetre, video_synthetique
):
    """À 0,25x, l'affichage saute des frames traitées au lieu de les rejouer.

    Le compteur affiché doit correspondre à l'image affichée : les deux sont
    mis à jour au même moment, sinon l'opérateur voit un chiffre qui avance
    devant la scène qu'il regarde.
    """
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.compteur = FauxCompteur()
    f._ouvrir_session()
    vues = []
    f.video.image_changee.connect(lambda img: vues.append(f.compteur_total))
    for _ in range(3):
        f._tick()
        f._afficher()
    assert len(vues) == 3
    assert vues == [1, 2, 3]
    # Un appel d'affichage supplémentaire sans frame neuve ne redessine rien.
    f._afficher()
    assert len(vues) == 3


# -- Export ---------------------------------------------------------------


def test_export_sans_video_refuse(application, tmp_path):
    f = FenetrePrincipale()
    assert f.exporter(str(tmp_path)) is False


def test_export_apres_analyse(application, fenetre, tmp_path, video_synthetique):
    """L'export dépend de `compteur/rapport.py`, livré par la tâche 6.

    Tant que ce module n'existe pas, la fenêtre doit le dire et refuser — pas
    lever `ImportError` au clic. Le test s'adapte donc à ce qui est réellement
    livré, plutôt que d'exiger un module d'une autre tâche.
    """
    f = fenetre
    f.charger_video(str(video_synthetique))
    f.compteur = FauxCompteur()
    f._ouvrir_session()
    avancer(f, 10)
    dossier = tmp_path / "sortie"
    dossier.mkdir()

    try:
        import compteur.rapport as rapport
    except ImportError:
        rapport = None

    if rapport is None:
        # Module absent : export refusé, message explicite, aucune exception.
        assert f.exporter(str(dossier)) is False
        assert "rapport" in f.label_statut.text().lower() or "export" in (
            f.label_statut.text().lower()
        )
        return

    assert f.exporter(str(dossier)) is True
    chemins = rapport.chemins_par_defaut(str(video_synthetique), str(dossier))
    assert chemins["csv"].exists()
    assert chemins["json"].exists()
    assert "Exporté" in f.label_statut.text()


# -- Réglages -------------------------------------------------------------


def test_changer_le_modele_ne_plante_pas(application, fenetre):
    """Aucun détecteur chargé : le changement de modèle ne doit rien casser."""
    f = fenetre
    f.panneau._modele.setCurrentText("nano.pt")
    assert f.config.modele == "nano.pt"
    f.panneau._modele.setCurrentText("medium.pt")
    assert f.config.modele == "medium.pt"


def test_changer_le_modele_invalide_le_detecteur(application, fenetre, monkeypatch):
    """Un modèle différent = des poids différents : le détecteur est lâché.

    Sans cela, changer de modèle en cours de route continuerait à compter avec
    l'ancien, et l'interface afficherait le nom du nouveau : le résultat serait
    attribué au mauvais modèle.
    """

    class FauxDetecteur:
        def __init__(self, config):
            self.config = config

    f = fenetre
    f.detecteur = FauxDetecteur(f.config)
    f._modele_charge = "medium.pt"
    f.panneau._modele.setCurrentText("nano.pt")
    assert f.detecteur is None
    assert f.tracker is None


def test_changer_un_reglage_met_a_jour_le_detecteur_vivant(
    application, fenetre, monkeypatch
):
    """Tant que le modèle ne change pas, le détecteur reçoit la nouvelle config."""

    class FauxDetecteur:
        def __init__(self, config):
            self.config = config

    f = fenetre
    f.detecteur = FauxDetecteur(f.config)
    f._modele_charge = f.config.modele
    f.panneau.definir_seuil(0.8)
    assert f.detecteur is not None
    assert f.detecteur.config.seuil_confiance == pytest.approx(0.8)


def test_le_lancer_signale_un_modele_absent_sans_vider_la_video(
    application, fenetre, video_synthetique
):
    """Pas de poids sur le disque : l'erreur est signalée, rien ne plante.

    Le cas est réel — les `.pt` ne sont pas versionnés (cf. `.gitignore`). La
    vidéo et la ligne déjà tracées doivent SURVIVRE à l'échec, sinon
    l'opérateur recommence tout pour un simple fichier manquant.
    """
    f = fenetre
    f.charger_video(str(video_synthetique))
    f._on_tracer_ligne()
    f._on_clic_video(10, 2)
    f._on_clic_video(10, 40)
    # Le modèle se règle DANS LE PANNEAU : `lancer` relit les widgets, pas
    # `self.config`. Un `medium.pt` valide se chargerait ici et l'analyse
    # démarrerait pour de bon — le test prendrait dix secondes sur le GPU.
    f.panneau._modele.setCurrentText("fichier_absent_xyz.pt")

    f.lancer()

    assert f._en_analyse is False
    assert not f._timer_traitement.isActive()
    assert f.compteur is None
    assert f.cap is not None
    assert f.ligne is not None
    assert f.btn_lancer.isEnabled()  # on peut réessayer une fois le modèle là


# -- Avertissement « 0 compté : vérifie le sens » ------------------------
#
# Le cas testé ici est LE piège de l'outil : sur la vidéo de référence,
# `sens=-1` compte 315 personnes et `sens=+1` (le défaut) en compte 1. Un
# opérateur qui laisse le défaut voit la vidéo défiler, les boîtes de
# détection s'afficher, et un zéro qui ne bouge pas. Sans rappel, il n'a
# aucun moyen de distinguer « personne ne traverse » de « tout le monde est
# compté à l'envers ».
#
# Le rappel doit être DISCRET : pas de modale (elle bloquerait un traitement
# en cours), et pas plus d'une fois par analyse (il est rappelé à chaque frame
# affichée, donc 25 fois par seconde).


class CompteurInvisible:
    """Compteur qui ne voit personne : `total` reste à 0 indéfiniment.

    Le constructeur accepte n'importe quoi : ce fake remplace `Compteur` dans
    le test de relance, qui l'instancie avec `(config, detecteur, tracker)`.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.ligne = None

    def ajuster_ligne(self, ligne) -> None:
        self.ligne = ligne

    def traiter_frame(self, img, index, ts) -> FrameResult:
        return FrameResult(
            image=img,
            total=0,
            presents=0,
            frame_index=index,
            timestamp_s=ts,
            evenements=[],
        )

    def resultat(self, modele, nb_frames, secondes):
        from compteur.types import Resultat

        return Resultat(
            total=0,
            evenements=[],
            modele=modele,
            nb_frames=nb_frames,
            presents_max=0,
            secondes=secondes,
        )


def avertissement_visible(fenetre):
    """L'avertissement est-il affiché ?

    On interroge `isHidden()` et non `isVisible()` : sous `offscreen`, et
    pour toute fenêtre jamais `show()`e, `isVisible()` renvoie `False` sur
    TOUS les enfants — y compris un label que l'on vient d'afficher. Le test
    passerait donc toujours, ou échouerait pour la mauvaise raison.
    """
    return fenetre.label_avertissement.isHidden() is False


class FauxDetecteurMinimal:
    """Détecteur sans poids : `lancer` n'a besoin que de l'instancier."""

    def __init__(self, modele, config) -> None:
        self.config = config


@pytest.fixture
def video_longue(tmp_path):
    """100 frames à 25 i/s, soit 4 s de vidéo.

    La vidéo de 5 frames du reste de la suite ne permet pas d'atteindre les
    trois secondes du seuil : l'analyse y finit avant que l'avertissement
    puisse avoir lieu de se déclencher.
    """
    chemin = tmp_path / "longue.mp4"
    auteur = cv2.VideoWriter(
        str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), FPS_SYNTHETIQUE, (64, 48)
    )
    assert auteur.isOpened(), "OpenCV n'a pas pu écrire la vidéo de test"
    for i in range(100):
        auteur.write(np.full((48, 64, 3), 20 + (i % 20) * 10, dtype=np.uint8))
    auteur.release()
    return chemin


def test_avertissement_sens_absent_au_demarrage(application, fenetre):
    """Une fenêtre vide n'affiche aucun rappel : rien n'a encore été compté."""
    f = fenetre
    assert avertissement_visible(f) is False
    assert f.label_avertissement.text() == ""


def test_avertissement_sens_pas_avant_le_seuil_de_trois_secondes(
    application, fenetre, video_longue
):
    """Point 3 : à 2,9 s de vidéo, le rappel n'est pas encore justifié.

    Sans ce délai, l'avertissement apparaîtrait sur les premières secondes de
    n'importe quelle analyse — y compris sur une scène où personne n'est
    encore passé — et deviendrait un bruit que l'opérateur ignore.
    """
    f = fenetre
    f.charger_video(str(video_longue))
    f.compteur = CompteurInvisible()
    f._ouvrir_session()
    avancer(f, 70)  # 2,8 s à 25 i/s
    assert avertissement_visible(f) is False


def test_avertissement_sens_apparait_au_dela_de_trois_secondes(
    application, fenetre, video_longue
):
    """Point 1 : zéro franc après 3 s de vidéo -> le rappel s'affiche."""
    f = fenetre
    f.charger_video(str(video_longue))
    f.compteur = CompteurInvisible()
    f._ouvrir_session()
    avancer(f, 80)  # 3,2 s à 25 i/s
    assert avertissement_visible(f) is True
    assert f.label_avertissement.text() == AVERTISSEMENT_SENS
    assert "sens" in f.label_avertissement.text().lower()


def test_avertissement_sens_disparait_des_que_le_compteur_repart(
    application, fenetre, video_longue
):
    """Point 2 : au premier comptage, le rappel s'efface.

    Un avertissement affiché pendant que le compteur monte serait un mensonge :
    il dirait à l'opérateur que son sens est douteux alors qu'il vient de
    compter quelqu'un.
    """
    f = fenetre
    f.charger_video(str(video_longue))
    f.compteur = CompteurInvisible()
    f._ouvrir_session()
    avancer(f, 80)
    assert avertissement_visible(f) is True
    f.maj_compteurs(total=1, presents=1, frames=81)
    assert avertissement_visible(f) is False


class LabelEspion:
    """_proxy du label qui compte les `setText`.

    Ce n'est pas un test cosmétique : `_afficher` est appelé 25 fois par
    seconde, donc un rappel réécrit à chaque frame serait lu comme un
    clignotement — l'opérateur le verrait clignoter et finirait par l'ignorer.
    On compte donc les ÉCRITURES, pas seulement la présence du texte.
    """

    def __init__(self, label) -> None:
        self._label = label
        self.écritures = 0

    def setText(self, texte: str) -> None:
        self.écritures += 1
        self._label.setText(texte)

    def setVisible(self, visible: bool) -> None:
        self._label.setVisible(visible)

    def isHidden(self) -> bool:
        return self._label.isHidden()

    def text(self) -> str:
        return self._label.text()


def test_avertissement_sens_ne_clignote_pas_a_haque_frame(
    application, fenetre, video_longue
):
    """Une seule fois par analyse, jamais une fois par frame.

    `_afficher` tourne 25 fois par seconde : sans ce verrou, le label serait
    réécrit 25 fois par seconde et l'opérateur le lirait comme un clignotement.
    """
    f = fenetre
    f.charger_video(str(video_longue))
    f.compteur = CompteurInvisible()
    f._ouvrir_session()
    f.label_avertissement = LabelEspion(f.label_avertissement)
    avancer(f, 80)
    écritures_au_declenchement = f.label_avertissement.écritures
    assert écritures_au_declenchement == 1
    avancer(f, 10)
    assert f.label_avertissement.écritures == écritures_au_declenchement
    assert f.label_avertissement.text() == AVERTISSEMENT_SENS
    assert avertissement_visible(f) is True


def test_avertissement_sens_reapparait_a_la_relance(
    application, fenetre, video_longue, monkeypatch
):
    """Relancer une nouvelle analyse réarme le rappel.

    Le scénario « mauvais sens » se corrige en changeant le réglage et en
    RELANCANT : si le rappel ne pouvait pas se redéployer, il resterait muet au
    moment précis où l'opérateur vérifie sa correction.

    Le modèle est remplacé par un faux : sans cela, `lancer` chargerait
    `medium.pt` sur le GPU et le test prendrait dix secondes.
    """
    monkeypatch.setattr("interface.app.charger_modele", lambda *a, **k: object())
    monkeypatch.setattr("interface.app.Detecteur", FauxDetecteurMinimal)
    monkeypatch.setattr("interface.app.Tracker", lambda config: object())
    monkeypatch.setattr("interface.app.Compteur", CompteurInvisible)

    f = fenetre
    f.charger_video(str(video_longue))
    f._on_tracer_ligne()
    f._on_clic_video(10, 2)
    f._on_clic_video(10, 40)
    f.lancer()
    assert f._en_analyse is True
    assert f._avertissement_sens_affiche is False

    avancer(f, 80)
    assert avertissement_visible(f) is True

    # L'opérateur corrige le sens et RELANCE : le rappel doit pouvoir se
    # redéployer au lieu de rester muet sur la nouvelle analyse.
    f.arreter()
    f.lancer()
    assert f._avertissement_sens_affiche is False


def test_avertissement_sens_ne_saffiche_pas_hors_analyse(
    application, fenetre, video_longue
):
    """Pas d'analyse ouverte, pas de rappel — même après 100 frames.

    Sans la garde sur la session, charger une vidéo et faire défiler
    l'aperçu suffirait à afficher un avertissement qui n'a aucun sens : rien
    n'a été compté.
    """
    f = fenetre
    f.charger_video(str(video_longue))
    assert f._session_ouverte is False
    f.maj_compteurs(total=0, presents=0, frames=500)
    assert avertissement_visible(f) is False
