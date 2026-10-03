"""Tests du panneau de réglages (tâche 9).

Le panneau ne fait qu'une chose : traduire l'état des widgets en une `Config`.
Ces tests vérifient donc trois propriétés — la fidélité de la traduction dans
les deux sens, la persistance des profils nommés, et la robustesse quand les
fichiers de modèle manquent (ils ne sont pas versionnés, cf. .gitignore).
"""

import pathlib

import pytest

pytest.importorskip("PySide6")

from compteur.config import Config  # noqa: E402
from interface.panneau_reglages import PanneauReglages, lister_modeles  # noqa: E402


@pytest.fixture
def application():
    """Une QApplication unique pour toute la session (Qt n'aime pas deux fois)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


# -- Lecture ------------------------------------------------------------


def test_lire_renvoie_la_config(application):
    p = PanneauReglages(Config(seuil_confiance=0.4))
    assert p.lire().seuil_confiance == 0.4


def test_lire_produit_une_config_valide(application):
    """La Config rendue doit survivre à un aller-retour JSON complet."""
    p = PanneauReglages(Config())
    lue = p.lire()
    assert isinstance(lue, Config)
    # vers_dict/depuis_dict est exactement le chemin emprunté par un profil.
    assert Config.depuis_dict(lue.vers_dict()) == lue


def test_lire_reflete_tous_les_reglages(application):
    """Chaque widget se retrouve dans la Config rendue."""
    c = Config(
        modele="yolov8n-head.pt",
        seuil_confiance=0.35,
        taille_min_px=3,
        taille_entree=320,
        frames_confirmation=4,
        survie_max=12,
        seuil_matching=0.45,
        epaisseur_bande=55,
        sens=-1,
        frames_hysteresis=5,
    )
    lue = PanneauReglages(c).lire()
    assert lue.modele == c.modele
    assert lue.seuil_confiance == pytest.approx(0.35)
    assert lue.taille_min_px == 3
    assert lue.taille_entree == 320
    assert lue.frames_confirmation == 4
    assert lue.survie_max == 12
    assert lue.seuil_matching == pytest.approx(0.45)
    assert lue.epaisseur_bande == 55
    assert lue.sens == -1
    assert lue.frames_hysteresis == 5


def test_lire_preserve_la_ligne_tracee(application):
    """Changer un réglage ne doit pas effacer la ligne tracée sur la vidéo.

    Régression : le panneau est construit à partir d'une Config qui ne connaît
    pas encore la ligne (l'utilisateur la trace ensuite) ; si `lire()` repartait
    d'une Config neuve, le moindre mouvement de curseur annulerait le
    franchissement que l'utilisateur est en train de placer.
    """
    ligne = (12.0, 40.0, 12.0, 640.0)
    p = PanneauReglages(Config())
    p.definir_seuil(0.7)
    assert p.lire().ligne is None

    p.appliquer(Config(seuil_confiance=0.7, ligne=ligne))
    assert p.lire().ligne == pytest.approx(ligne)


def test_lire_preserve_les_classes_retenues(application):
    p = PanneauReglages(Config(classes_retenues=[0, 2]))
    p.definir_seuil(0.6)
    assert p.lire().classes_retenues == [0, 2]


# -- Curseurs -----------------------------------------------------------


def test_modifier_un_slider_met_a_jour_la_config(application):
    p = PanneauReglages(Config())
    p.definir_seuil(0.65)
    assert p.lire().seuil_confiance == pytest.approx(0.65)


def test_changer_un_curseur_met_a_jour_la_config_emise(application):
    """Le signal `config_modifiee` transporte la nouvelle valeur, pas l'ancienne."""
    p = PanneauReglages(Config())
    captures = []
    p.config_modifiee.connect(captures.append)

    p.definir_seuil(0.5)
    p.definir_taille_min(3)
    p.definir_epaisseur_bande(70)

    assert len(captures) == 3
    assert captures[0].seuil_confiance == pytest.approx(0.5)
    assert captures[1].taille_min_px == 3
    assert captures[2].epaisseur_bande == 70
    # La Config émise est aussi celle que `lire()` rend : pas de dérive.
    assert captures[-1].taille_min_px == p.lire().taille_min_px


def test_un_curseur_inchange_n_emettre_rien(application):
    """Poser la valeur déjà en place ne doit pas réveiller l'application."""
    p = PanneauReglages(Config(seuil_confiance=0.5))
    captures = []
    p.config_modifiee.connect(captures.append)
    p.definir_seuil(0.5)
    assert captures == []


# -- Application d'une config externe ----------------------------------


def test_appliquer_une_config_externe(application):
    p = PanneauReglages(Config())
    p.appliquer(Config(seuil_confiance=0.8, epaisseur_bande=70))
    assert p.lire().seuil_confiance == 0.8
    assert p.lire().epaisseur_bande == 70


def test_appliquer_emet_t_une_seule_fois(application):
    """Charger un profil ne doit pas réveiller l'application 10 fois."""
    p = PanneauReglages(Config())
    captures = []
    p.config_modifiee.connect(captures.append)
    p.appliquer(Config(seuil_confiance=0.8, epaisseur_bande=70))
    assert len(captures) == 1
    assert captures[0].seuil_confiance == 0.8


def test_taille_entree_hors_liste_est_acceptee(application):
    """Une taille d'entrée exotique (512) ne doit pas être écrasée en silence."""
    p = PanneauReglages(Config(taille_entree=512))
    assert p.lire().taille_entree == 512
    p.appliquer(Config(taille_entree=512))
    assert p.lire().taille_entree == 512


def test_construction_ne_emet_t_rien(application):
    """Construire le panneau n'est pas un changement de réglages."""
    captures = []
    p = PanneauReglages(Config())
    p.config_modifiee.connect(captures.append)
    assert captures == []


# -- Profils ------------------------------------------------------------


def test_profil_enregistre_et_recharge(application, tmp_path):
    p = PanneauReglages(Config())
    p.definir_seuil(0.33)
    p.enregistrer_profil("essai", dossier=tmp_path)
    assert (tmp_path / "essai.json").exists()
    p2 = PanneauReglages(Config(seuil_confiance=0.9))
    assert p2.charger_profil("essai", dossier=tmp_path) is True
    assert p2.lire().seuil_confiance == pytest.approx(0.33)


def test_profil_aller_retour_complet(application, tmp_path):
    """Un profil doit restituer exactement les réglages qu'il a figés."""
    depart = PanneauReglages(Config())
    depart.definir_seuil(0.42)
    depart.definir_taille_min(3)
    depart.definir_epaisseur_bande(64)
    ecrit = depart.enregistrer_profil("complet", dossier=tmp_path)
    assert Config.depuis_fichier(ecrit).seuil_confiance == pytest.approx(0.42)

    retour = PanneauReglages(Config())
    assert retour.charger_profil("complet", dossier=tmp_path) is True
    assert retour.lire() == depart.lire()


def test_profil_ecrase_une_version_precedente(application, tmp_path):
    """Réenregistrer sous le même nom écrase, sans empiler de fichier."""
    p = PanneauReglages(Config())
    p.definir_seuil(0.3)
    p.enregistrer_profil("scene", dossier=tmp_path)
    p.definir_seuil(0.8)
    p.enregistrer_profil("scene", dossier=tmp_path)
    assert len(list(tmp_path.glob("scene*.json"))) == 1
    assert Config.depuis_fichier(tmp_path / "scene.json").seuil_confiance == (
        pytest.approx(0.8)
    )


def test_enregistrer_cree_le_dossier_manquant(application, tmp_path):
    cible = tmp_path / "profils"
    assert not cible.exists()
    p = PanneauReglages(Config())
    p.enregistrer_profil("neuf", dossier=cible)
    assert (cible / "neuf.json").exists()


def test_profil_apparait_dans_la_liste(application, tmp_path):
    """Après enregistrement, le nom est proposé dans la liste déroulante."""
    p = PanneauReglages(Config())
    p.enregistrer_profil("dense", dossier=tmp_path)
    assert p.noms_profils() == ["dense"]


def test_charger_profil_inexistant_returns_false(application, tmp_path):
    p = PanneauReglages(Config())
    assert p.charger_profil("inconnu", dossier=tmp_path) is False


def test_charger_profil_corrompu_returns_false(application, tmp_path):
    """Un fichier écrit à la main et invalide ne doit pas faire exploser l'IHM."""
    (tmp_path / "casse.json").write_text("{ pas du json", encoding="utf-8")
    p = PanneauReglages(Config(seuil_confiance=0.9))
    assert p.charger_profil("casse", dossier=tmp_path) is False
    # La config affichée n'a pas bougé : le panneau est toujours cohérent.
    assert p.lire().seuil_confiance == 0.9


def test_profil_avec_cle_inconnue_returns_false(application, tmp_path):
    """Un profil contenant une clé inconnue est refusé, pas tronqué.

    Sans ce refus, `Config.depuis_dict` lèverait une exception ; on veut que la
    liste déroulante reste cohérente plutôt que de charger une config à moitié.
    """
    (tmp_path / "futur.json").write_text(
        '{"modele": "medium.pt", "reglage_de_une_version_future": 1}',
        encoding="utf-8",
    )
    p = PanneauReglages(Config(seuil_confiance=0.9))
    assert p.charger_profil("futur", dossier=tmp_path) is False
    assert p.lire().seuil_confiance == 0.9


def test_fenetre_lissage_va_et_vient(application):
    """`fenetre_lissage` a été ajouté à Config : le panneau le suit sans code
    supplémentaire. Le test échoue bruyamment si le champ disparaît."""
    assert hasattr(Config(), "fenetre_lissage")
    p = PanneauReglages(Config())
    assert p.lire().fenetre_lissage == 10
    p.definir_lissage(25)
    assert p.lire().fenetre_lissage == 25
    p.appliquer(Config(fenetre_lissage=4))
    assert p.lire().fenetre_lissage == 4


def test_profils_par_defaut_vont_dans_config_profils(application, monkeypatch, tmp_path):
    """Sans argument, tout se joue dans config/profils du dépôt."""
    import interface.panneau_reglages as module

    monkeypatch.setattr(module, "DOSSIER_PROFILS", tmp_path / "profils")
    p = PanneauReglages(Config())
    p.enregistrer_profil("defaut")
    assert (tmp_path / "profils" / "defaut.json").exists()
    assert PanneauReglages(Config()).charger_profil("defaut") is True


# -- Signal -------------------------------------------------------------


def test_signal_emis_quand_la_config_change(application):
    p = PanneauReglages(Config())
    captures = []
    p.config_modifiee.connect(captures.append)
    p.definir_seuil(0.5)
    assert len(captures) == 1
    assert captures[0].seuil_confiance == pytest.approx(0.5)


# -- Modèles absents ----------------------------------------------------


def test_lister_modeles_trouve_les_fichiers_presents(tmp_path):
    (tmp_path / "medium.pt").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignoré", encoding="utf-8")
    assert lister_modeles([tmp_path]) == ["medium.pt"]


def test_lister_modeles_dossier_vide_ne_crashe_pas(tmp_path):
    """Aucun .pt sur le disque : on renvoie la liste de repli, sans exception."""
    assert lister_modeles([tmp_path]) == ["medium.pt"]


def test_lister_modeles_dossier_absent_ne_crashe_pas(tmp_path):
    assert lister_modeles([tmp_path / "nulle_part"]) == ["medium.pt"]


def test_panneau_ne_crashe_pas_sans_fichier_de_modele(application, tmp_path, monkeypatch):
    """Le modèle par défaut n'est pas versionné : le panneau doit pouvoir
    s'ouvrir sur une installation fraîche, sans aucun .pt sur le disque.

    Le repli proposé est `medium.pt`, le modèle de tête entraîné sur
    SCUT-HEAD, seul choix par défaut défendable (cf. .superpowers/sdd).
    """
    vide = tmp_path / "modeles"
    vide.mkdir()
    monkeypatch.chdir(tmp_path)
    p = PanneauReglages(Config(), dossiers_modeles=[vide])
    assert p.modeles_proposes() == ["medium.pt"]
    # La Config fournie reste la source de vérité : le panneau ne substitue
    # pas un modèle à l'utilisateur sans qu'il le demande.
    assert p.lire().modele == Config().modele


def test_le_modele_absent_du_disque_reste_saisissable(application, tmp_path, monkeypatch):
    """Un modèle téléchargé à la main reste sélectionnable même s'il est absent
    de la liste (le poste qui l'a téléchargé n'a pas le fichier, celui qui
    l'exécute si)."""
    vide = tmp_path / "modeles"
    vide.mkdir()
    monkeypatch.chdir(tmp_path)
    p = PanneauReglages(Config(modele="custom-head.pt"), dossiers_modeles=[vide])
    assert p.lire().modele == "custom-head.pt"


def test_modele_present_est_propose(application, tmp_path, monkeypatch):
    dossier = tmp_path / "modeles"
    dossier.mkdir()
    (dossier / "medium.pt").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    p = PanneauReglages(Config(modele="medium.pt"), dossiers_modeles=[dossier])
    assert p.lire().modele == "medium.pt"


def test_le_repertoire_courant_est_par_default(application, monkeypatch, tmp_path):
    """Sans argument, le panneau cherche les .pt du répertoire courant."""
    (tmp_path / "nano.pt").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert lister_modeles() == ["nano.pt"]
    assert PanneauReglages(Config(modele="nano.pt")).lire().modele == "nano.pt"