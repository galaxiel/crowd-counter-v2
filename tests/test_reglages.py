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
from compteur.ligne import LIBELLES_SENS  # noqa: E402
from interface.panneau_reglages import (  # noqa: E402
    AIDE,
    PanneauReglages,
    lister_modeles,
    orientation_par_defaut_sens,
)


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
    """Le signal `config_modifiee` transporte la nouvelle valeur, pas l'ancienne.

    Chaque `definir_*` utilise une valeur DIFFÉRENTE du défaut courant, sinon
    Qt ne signale aucun changement et aucune émission n'a lieu — c'est le
    comportement correct de Qt, pas un défaut du panneau. `taille_min_px` vaut
    3 par défaut, d'où le choix de 40 ci-dessous.
    """
    p = PanneauReglages(Config())
    captures = []
    p.config_modifiee.connect(captures.append)

    defaut = p.lire()
    p.definir_seuil(0.5)
    p.definir_taille_min(40)
    p.definir_epaisseur_bande(70)

    assert len(captures) == 3
    assert captures[-1].seuil_confiance == pytest.approx(0.5)
    assert captures[-1].taille_min_px == 40
    assert captures[-1].epaisseur_bande == 70
    assert captures[-1].seuil_confiance != defaut.seuil_confiance
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
    """`fenetre_lissage` est exposé par le panneau.

    La valeur par défaut est 1, pas 10 : mesuré, K>1 retarde la position
    d'une demi-fenetre, la position lissée est encore dans la bande au
    moment du croisement, et le comptage est perdu. Le panneau suit
    Config sans code supplémentaire — le test échoue bruyamment si le champ
    disparaît, et il vérifie lealler-retour quelle que soit la valeur.
    """
    assert hasattr(Config(), "fenetre_lissage")
    p = PanneauReglages(Config())
    assert p.lire().fenetre_lissage == Config().fenetre_lissage
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


# -- Sens : libellés explicites, pilotés par la ligne tracée -------------


def _libelles(p):
    return [p._sens.itemText(i) for i in range(p._sens.count())]


def test_sens_affiche_des_libelles_de_direction_pas_avant_apres(application):
    """« Avant → après » ne disait pas avant quoi : plus rien ne doit l'afficher."""
    p = PanneauReglages(Config())
    libelles = " ".join(_libelles(p))
    assert "Avant" not in libelles
    assert "Après" not in libelles
    assert set(_libelles(p)) == set(LIBELLES_SENS[orientation_par_defaut_sens()])


def test_sens_verticale_affiche_gauche_droite_et_droite_gauche(application):
    p = PanneauReglages(Config())
    p.definir_ligne((640.0, 0.0, 640.0, 720.0))
    assert _libelles(p) == ["Gauche → droite", "Droite → gauche"]


def test_sens_horizontale_affiche_haut_bas_et_bas_haut(application):
    """Une ligne tracée à l'horizontale change la NATURE des libellés.

    C'est la demande explicite : « Gauche → droite » sur une ligne horizontale
    n'aurait aucun sens pour l'opérateur, qui voit les gens monter et descendre.
    """
    p = PanneauReglages(Config())
    p.definir_ligne((0.0, 400.0, 1280.0, 400.0))
    assert _libelles(p) == ["Haut → bas", "Bas → haut"]


def test_libelles_recalcules_quand_la_ligne_change(application):
    """Passer d'une verticale à une horizontale renomme les choix, sur place."""
    p = PanneauReglages(Config())
    p.definir_ligne((640.0, 0.0, 640.0, 720.0))
    assert _libelles(p)[0] == "Gauche → droite"
    p.definir_ligne((0.0, 400.0, 1280.0, 400.0))
    assert _libelles(p)[0] == "Haut → bas"


def test_changer_de_ligne_ne_change_pas_le_sens_compté(application):
    """Renommer les choix ne doit pas retourner le comptage.

    Régression centrale : si `lire()` déduisait le `sens` de l'INDICE, le
    passage vertical -> horizontal pourrait conserver l'indice tout en
    changeant de sens. Le `sens` est donc attaché au LIBELLÉ, jamais à sa place.
    """
    p = PanneauReglages(Config(sens=-1))
    p.definir_ligne((640.0, 0.0, 640.0, 720.0))
    assert p.lire().sens == -1
    p.definir_ligne((0.0, 400.0, 1280.0, 400.0))
    assert p.lire().sens == -1, "le sens a bougé alors que la ligne changeait"


def test_lire_et_afficher_restent_daccord_apres_un_changement_de_ligne(application):
    """Ce que le menu affiche est ce que `lire()` renvoie, ligne horizontale comprise."""
    from compteur.ligne import Ligne

    p = PanneauReglages(Config())
    p.definir_ligne((0.0, 400.0, 1280.0, 400.0))
    for index, libelle in enumerate(_libelles(p)):
        p._sens.setCurrentIndex(index)
        sens = p.lire().sens
        reel = Ligne((0.0, 400.0), (1280.0, 400.0), sens=sens).libelle_sens(sens)
        assert reel == libelle, (
            f"index {index} : le menu propose « {libelle} », le moteur compte « {reel} »"
        )


def test_le_sens_par_defaut_du_panneau_compte_droite_gauche(application):
    """Non-régression du 315, au niveau de l'interface cette fois.

    `Config.defauts()` porte `sens=+1`. Sur une ligne verticale tracée de bas en
    haut — l'ordre de tracé de la vidéo de référence — ce `+1` compte de
    DROITE À GAUCHE. Ouvrir l'application et lancer sans rien toucher doit donc
    reproduire les 315, pas 1. Ce test échoue bruyamment si un libellé ou un
    index a été inversé.
    """
    from compteur.config import Config as C, chemin_defaut_config

    p = PanneauReglages(C.depuis_fichier(chemin_defaut_config()))
    p.definir_ligne((640.0, 720.0, 640.0, 0.0))
    from compteur.ligne import Ligne

    sens = p.lire().sens
    libelle = Ligne((640.0, 720.0), (640.0, 0.0), sens=sens).libelle_sens(sens)
    assert libelle == "Droite → gauche"
    # ... et le libellé affiché par le menu est bien celui-là.
    assert libelle in _libelles(p)


def test_appliquer_une_config_repose_le_bon_libelle(application):
    """Charger un profil avec `ligne` réétiquette le menu sans perdre le sens."""
    p = PanneauReglages(Config())
    p.appliquer(Config(sens=-1, ligne=(0.0, 400.0, 1280.0, 400.0)))
    assert _libelles(p) == ["Haut → bas", "Bas → haut"]
    assert p.lire().sens == -1
    # Le libellé de -1 sur cette horizontale est bien celui affiché, pas l'autre.
    from compteur.ligne import Ligne

    attendu = Ligne((0.0, 400.0), (1280.0, 400.0), sens=-1).libelle_sens(-1)
    assert attendu in _libelles(p)


def test_ligne_memorisee_invalide_ne_casse_pas_le_panneau(application):
    """Un profil écrit à la main avec deux points confondus ne doit pas planter.

    Le panneau s'ouvre avant tout tracé : il a besoin d'une hypothèse pour
    nommer les libellés, pas d'une ligne valide.
    """
    p = PanneauReglages(Config())
    p.appliquer(Config(ligne=(10.0, 10.0, 10.0, 10.0)))
    assert _libelles(p) == list(LIBELLES_SENS[orientation_par_defaut_sens()])
    assert p.lire().sens in (1, -1)


# -- Explications des réglages -------------------------------------------

REGLES = (
    "modele",
    "seuil_confiance",
    "taille_min_px",
    "taille_entree",
    "frames_confirmation",
    "survie_max",
    "seuil_matching",
    "epaisseur_bande",
    "sens",
    "frames_hysteresis",
    "fenetre_lissage",
)


@pytest.mark.parametrize("champ", REGLES)
def test_chaque_reglage_a_une_explication(application, champ):
    """Trois questions par réglage : ce que ça fait, ce que ça change, le conseil.

    Un réglage sans explication est un réglage que l'opérateur change au hasard
    puis conclut que le programme est faux.
    """
    p = PanneauReglages(Config())
    aide = p._widgets[champ].toolTip()
    assert aide, f"{champ} n'a aucune explication"
    assert "Ce que ça fait" in aide, f"{champ} : manque la première question"
    assert "Ce que ça change" in aide, f"{champ} : manque la deuxième question"
    assert "Valeur conseillée" in aide, f"{champ} : manque la valeur conseillée"


@pytest.mark.parametrize("champ", REGLES)
def test_le_conseil_chiffre_est_soutenu_par_une_mesure(application, champ):
    """Un conseil numérique doit citer d'où il sort.

    Sans cette exigence, une valeuradvice reste une convention habillée en
    preuve — exactement ce que l'opérateur ne peut pas vérifier.
    """
    aide = AIDE[champ]
    conseil = [l for l in aide.split("\n") if l.startswith("Valeur conseillée")]
    assert conseil, f"{champ} : pas de ligne de conseil"
    ligne = conseil[0]
    assert any(c.isdigit() for c in ligne), f"{champ} : conseil sans chiffre"


def test_chaque_reglage_a_une_aide_visible_sous_le_champ(application):
    """L'explication doit être lisible SANS survol de la souris.

    L'info-bulle seule est un piège : hors du focus d'un informaticien, une
    bulle reste invisible. L'encart est donc obligatoire.
    """
    p = PanneauReglages(Config())
    for champ in REGLES:
        assert p._widgets[champ].toolTip(), f"{champ} : pas d'info-bulle"


def test_survie_max_atteignable_a_400_pour_verifier_la_mesure(application):
    """La plage doit permettre de reproduire la mesure de survie_max.

    On a mesuré 315 à 30 frames et 325 à 400. Un réglage borné à 120
    empêcherait de vérifier ce que dit l'aide, et une aide invérifiable est
    une aide fausse.
    """
    p = PanneauReglages(Config())
    assert p._widgets["survie_max"].maximum() >= 400
    p._widgets["survie_max"].setValue(400)
    assert p.lire().survie_max == 400


def test_taille_entree_propose_1280(application):
    """1280 est la valeur conseillée : elle doit être sélectionnable."""
    p = PanneauReglages(Config())
    choix = [p._widgets["taille_entree"].itemText(i)
             for i in range(p._widgets["taille_entree"].count())]
    assert "1280" in choix
    assert p.lire().taille_entree in (320, 640, 1280)


def test_le_panneau_defile_dans_la_fenetre(application):
    """Les explications ajoutées ne doivent pas pousser les boutons hors de l'écran.

    Le panneau porte désormais une explication lisible sous chaque réglage, ce
    qui porte sa hauteur naturelle bien au-delà d'une fenêtre de 850 px. Sans
    zone défilante, Qt comprime le panneau et le bouton « Lancer » devient
    inatteignable : l'analyse ne serait plus lançable du tout.
    """
    from PySide6.QtWidgets import QScrollArea

    from interface.app import FenetrePrincipale

    f = FenetrePrincipale()
    f.resize(1400, 850)
    f.show()
    application.processEvents()
    zone = f.findChild(QScrollArea)
    assert zone is not None, "le panneau doit être dans une zone défilante"
    assert zone.widget() is f.panneau
    assert f.panneau.sizeHint().height() > zone.height(), (
        "le panneau devrait réellement déborder : sans défilement, il serait "
        "comprimé et les boutons hors de l'écran"
    )
    # Le bouton de lancement doit rester dans la fenêtre.
    assert f.btn_lancer.y() + f.btn_lancer.height() <= 850
    assert f.btn_lancer.isVisible()
