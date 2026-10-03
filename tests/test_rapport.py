"""Tests de l'export : CSV, JSON, statistiques et indicateurs de fiabilité.

Le module `compteur.rapport` est ce qui transforme un compteur en outil
utilisable par un média : sans lui, le total affiché n'existe que sur l'écran.

Trois exigences gouvernent ces tests :

1. **Reproductibilité** — le JSON porte la config COMPLÈTE employée. Relire
   `config/default.json` ne suffit pas : un export doit dire ce qui a
   réellement tourné, pas ce qui est prévu par défaut au moment de la lecture.

2. **Honnêteté** — sur la vidéo de référence, la détection est fiable (88 têtes
   par frame) mais le comptage rend 0 : le mouvement des personnes est noyé
   dans le bruit du détecteur (déplacement médian 0,00 px/frame, bruit σ=3 px).
   Un « 0 personnes » lu sans contexte est un mensonge journalistique. Le
   rapport doit donc exposer les indicateurs de fiabilité qui permettent à un
   lecteur de voir que le total n'est pas fiable, et NE DOIT PAS produire de
   pourcentage d'erreur automatique : on n'a pas la vérité terrain.

3. **Robustesse** — zéro événement est le cas RÉEL de la vidéo de référence,
   pas un cas limite inventé pour les tests. L'export ne doit pas planter.
"""

import json
from dataclasses import fields

import pytest

from compteur.config import Config
from compteur.rapport import (
    ENTETE_CSV,
    chemins_par_defaut,
    ecrire_csv,
    ecrire_json,
    indicateurs_fiabilite,
    statistiques,
)
from compteur.types import Evenement, Resultat


def resultat_fictif():
    """Trois passages en 10 s, deux frames par personne : le cas « qui marche »."""
    evenements = [
        Evenement(frame=30, timestamp_s=1.0, x=100.0, y=50.0, track_id=1),
        Evenement(frame=60, timestamp_s=2.0, x=110.0, y=50.0, track_id=2),
        Evenement(frame=150, timestamp_s=5.0, x=120.0, y=50.0, track_id=3),
    ]
    return Resultat(
        total=3,
        evenements=evenements,
        config=Config(modele="yolov8n-head.pt", seuil_confiance=0.3),
        modele="yolov8n-head.pt",
        nb_frames=300,
        presents_max=5,
        presents_moyen=2.5,
        secondes=300 / 30,
    )


def resultat_video_de_reference():
    """Le cas réel : 88 détections par frame, 0 franchissement compté."""
    return Resultat(
        total=0,
        evenements=[],
        config=Config(modele="medium.pt", seuil_confiance=0.25, ligne=(556.0, 90.0, 588.0, 654.0)),
        modele="medium.pt",
        nb_frames=300,
        presents_max=88,
        presents_moyen=87.4,
        secondes=300 / 30,
    )


# -- CSV : une ligne par événement, en-tête correcte ----------------------


def test_csv_ecrit_une_ligne_par_evenement(tmp_path):
    p = ecrire_csv(resultat_fictif(), tmp_path / "r.csv")
    lignes = p.read_text(encoding="utf-8").strip().splitlines()
    assert lignes[0] == "frame,timestamp_s,x,y,track_id"
    assert len(lignes) == 4


def test_csv_contenu_aligne_avec_les_evenements(tmp_path):
    """Pas seulement le NOMBRE de lignes : les valeurs doivent suivre.

    Une écriture qui juxtapose les événements dans le désordre, ou qui écrit
    une colonne décalée, passerait le test du compte de lignes tout en
    livrant un CSV faux à celui qui l'ouvre dans un tableur.
    """
    r = resultat_fictif()
    p = ecrire_csv(r, tmp_path / "r.csv")
    lignes = p.read_text(encoding="utf-8").strip().splitlines()[1:]
    for ligne, ev in zip(lignes, r.evenements, strict=True):
        frame, timestamp_s, x, y, track_id = ligne.split(",")
        assert int(frame) == ev.frame
        assert float(timestamp_s) == pytest.approx(ev.timestamp_s, abs=1e-3)
        assert float(x) == pytest.approx(ev.x, abs=0.05)
        assert float(y) == pytest.approx(ev.y, abs=0.05)
        assert int(track_id) == ev.track_id


def test_csv_avec_zero_evenement_est_juste_l_entete(tmp_path):
    """Zéro événement : l'en-tête seule, pas d'exception, pas de ligne fantôme."""
    r = resultat_fictif()
    r.evenements = []
    r.total = 0
    p = ecrire_csv(r, tmp_path / "r.csv")
    lignes = p.read_text(encoding="utf-8").strip().splitlines()
    assert lignes == [ENTETE_CSV]


def test_csv_cree_les_dossiers_manquants(tmp_path):
    """L'interface passe un dossier choisi par l'utilisateur : il peut
    n'exister que d'un niveau, ou pas du tout."""
    cible = tmp_path / "sortie" / "reperes" / "r.csv"
    assert not cible.parent.exists()
    p = ecrire_csv(resultat_fictif(), cible)
    assert p == cible
    assert p.exists()


def test_csv_ecrit_les_octets_attendus(tmp_path):
    """Le contenu du CSV est entièrement ASCII par construction
    (`Evenement.vers_ligne_csv` ne formate que des nombres) : le risque
    d'encodage n'existe donc pas pour lui, et l'assertion porte sur les
    OCTETS, pas sur un décodage. Le vrai risque d'encodage est dans le JSON,
    qui porte des messages français — voir
    `test_json_accents_intacts_en_utf8`."""
    r = resultat_fictif()
    attendu = "\n".join([ENTETE_CSV] + [ev.vers_ligne_csv() for ev in r.evenements])
    brut = ecrire_csv(r, tmp_path / "r.csv").read_bytes()
    assert brut == (attendu + "\n").encode("ascii")


# -- JSON : total, config COMPLÈTE, événements ----------------------------


def test_json_contient_total_et_config(tmp_path):
    p = ecrire_json(resultat_fictif(), tmp_path / "r.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["total"] == 3
    assert d["config"]["modele"] == "yolov8n-head.pt"
    assert d["config"]["seuil_confiance"] == 0.3
    assert len(d["evenements"]) == 3


def test_json_contient_la_config_complete_utilisee(tmp_path):
    """C'est ce qui rend une analyse rejouable.

    Une config tronquée aux deux champs « visibles » ne dit pas quel seuil de
    détection, quelle taille minimale, quelle bande ni quel lissage ont
    servi : le rapport serait alors invérifiable.
    """
    config = Config(
        modele="medium.pt",
        seuil_confiance=0.41,
        taille_min_px=7,
        classes_retenues=[0],
        taille_entree=512,
        frames_confirmation=5,
        survie_max=17,
        seuil_matching=0.27,
        ligne=(1.0, 2.0, 3.0, 4.0),
        epaisseur_bande=44,
        sens=-1,
        frames_hysteresis=3,
        fenetre_lissage=10,
    )
    r = Resultat(
        total=1,
        evenements=[Evenement(frame=1, timestamp_s=0.1, x=1.0, y=2.0, track_id=9)],
        config=config,
        modele="medium.pt",
        nb_frames=30,
        presents_max=2,
        presents_moyen=1.5,
        secondes=1.0,
    )
    d = json.loads(ecrire_json(r, tmp_path / "r.json").read_text(encoding="utf-8"))

    attendus = {f.name for f in fields(Config)}
    assert set(d["config"]) == attendus, "un champ de Config manque dans l'export"
    assert d["config"] == config.vers_dict()
    # Le tuple de la ligne doit être rejouable : une liste de 4 flottants.
    assert d["config"]["ligne"] == [1.0, 2.0, 3.0, 4.0]
    assert Config.depuis_dict(d["config"]) == config


def test_json_documente_le_contexte_de_la_video(tmp_path):
    """Modèle, nombre de frames et durée : leminimum pour situer le chiffre."""
    d = json.loads(
        ecrire_json(resultat_fictif(), tmp_path / "r.json").read_text(encoding="utf-8")
    )
    assert d["modele"] == "yolov8n-head.pt"
    assert d["nb_frames"] == 300
    assert d["secondes"] == pytest.approx(10.0)
    assert d["presents_max"] == 5
    assert d["presents_moyen"] == pytest.approx(2.5)


def test_json_sans_config_ne_plante_pas(tmp_path):
    """`Resultat.config` est optionnel : l'export reste possible sans."""
    r = resultat_fictif()
    r.config = None
    d = json.loads(ecrire_json(r, tmp_path / "r.json").read_text(encoding="utf-8"))
    assert d["config"] is None
    assert d["total"] == 3


def test_json_evenements_sont_complets_et_numeriques(tmp_path):
    """Un événement doit survivre à l'aller-retour JSON, champs par champs."""
    r = resultat_fictif()
    d = json.loads(ecrire_json(r, tmp_path / "r.json").read_text(encoding="utf-8"))
    for ev, brut in zip(r.evenements, d["evenements"], strict=True):
        assert set(brut) == {"frame", "timestamp_s", "x", "y", "track_id"}
        assert brut == {
            "frame": ev.frame,
            "timestamp_s": ev.timestamp_s,
            "x": ev.x,
            "y": ev.y,
            "track_id": ev.track_id,
        }


def test_json_accents_intacts_en_utf8(tmp_path):
    """C'est dans le JSON que vit le risque d'encodage : il porte des messages
    d'avertissement en français, écrits avec `ensure_ascii=False`.

    Un rapport qui sort des « dÃ©faut de comptage » parce qu'il a été écrit en
    latin-1 perd exactement l'information qui existe pour être lue.
    """
    p = ecrire_json(resultat_video_de_reference(), tmp_path / "r.json")
    texte = p.read_bytes().decode("utf-8")  # lève si le fichier n'est pas UTF-8
    assert "défaut de comptage" in texte
    # Les accents sont présents LITTÉRALEMENT, pas échappés en \\uXXXX.
    assert "\\u00e9" not in texte


def test_json_ne_contient_aucun_chiffre_d_erreur(tmp_path):
    """Pas de pourcentage d'erreur : on n'a pas la vérité terrain.

    Un « ± 5 % d'erreur » inventé à partir de la config serait lu par un
    média comme une mesure. Le rapport doit s'interdire ce champ.
    """
    d = json.loads(
        ecrire_json(resultat_fictif(), tmp_path / "r.json").read_text(encoding="utf-8")
    )
    interdits = ("erreur", "precision", "accuracy", "incertitude_pct", "marge")
    cles = set(d) | set(d["statistiques"]) | set(d["fiabilite"])
    assert not [c for c in cles if any(i in c.lower() for i in interdits)], (
        "le rapport ne doit publier aucun taux d'erreur automatique"
    )


# -- Statistiques ----------------------------------------------------------


def test_statistiques_debit():
    s = statistiques(resultat_fictif())
    assert s["total"] == 3
    assert s["duree_s"] == 10.0
    assert s["personnes_par_minute"] == 18.0
    assert s["presents_max"] == 5


def test_statistiques_vide_sans_evenement():
    r = resultat_fictif()
    r.evenements = []
    r.total = 0
    s = statistiques(r)
    assert s["personnes_par_minute"] == 0.0


def test_statistiques_zero_duree_ne_divise_pas_par_zero():
    r = resultat_fictif()
    r.secondes = 0.0
    s = statistiques(r)
    assert s["duree_s"] == 0.0
    assert s["personnes_par_minute"] == 0.0
    assert s["fps_moyen"] is None


def test_statistiques_declares_les_cles_attendues():
    """Contrat avec `tools/comparer_modeles.py` et l'interface : l'ensemble des
    clés promises par le plan doit être là, sinon un appelant casse."""
    s = statistiques(resultat_fictif())
    for cle in (
        "total",
        "duree_s",
        "personnes_par_minute",
        "presents_max",
        "presents_moyen",
        "debit_max_par_minute",
        "modele",
        "nb_frames",
    ):
        assert cle in s, f"clé de statistiques manquante : {cle}"
    # Le module ne doit rien importer : aucun toolkit graphique ici non plus.
    assert set(s["fiabilite"])


def test_debit_max_prend_le_pic_et_ignore_les_bursts_lointains():
    """Le débit maximal est le PIC, pas la moyenne.

    Trois passages à t=1, 2, 3 s puis deux autres à t=100, 101 s : la meilleure
    minute contient les trois premiers, pas les cinq. Sur une vidéo de 200 s,
    la fenêtre est bien la minute complète.
    """
    horodatages = [1.0, 2.0, 3.0, 100.0, 101.0]
    evenements = [
        Evenement(frame=i, timestamp_s=t, x=1.0, y=1.0, track_id=i)
        for i, t in enumerate(horodatages, start=1)
    ]
    r = Resultat(
        total=len(evenements),
        evenements=evenements,
        config=Config(),
        modele="m.pt",
        nb_frames=6000,
        presents_max=3,
        presents_moyen=1.0,
        secondes=200.0,
    )
    s = statistiques(r)
    assert s["debit_max_fenetre_s"] == 60.0
    assert s["debit_max_par_minute"] == 3
    assert s["personnes_par_minute"] == pytest.approx(1.5)


def test_debit_max_sur_burst_ne_setend_pas_une_minute_inexistante():
    """600 passages en 20 s : le rapport publie 600 sur 20 s, PAS 1800 « par
    minute ». Extrapoler supposerait un débit uniforme que rien ne mesure — et
    un lieu qui voit 600 personnes en 20 s n'en verra pas 1800 en une minute."""
    evenements = [
        Evenement(frame=i, timestamp_s=i / 30.0, x=1.0, y=1.0, track_id=i)
        for i in range(600)
    ]
    r = Resultat(
        total=len(evenements),
        evenements=evenements,
        config=Config(),
        modele="m.pt",
        nb_frames=600,
        presents_max=3,
        presents_moyen=1.0,
        secondes=20.0,
    )
    s = statistiques(r)
    assert s["debit_max_fenetre_s"] == 20.0
    assert s["debit_max_par_minute"] == 600
    assert s["personnes_par_minute"] == pytest.approx(1800.0)


def test_debit_max_signale_une_fenetre_plus_courte_que_la_minute():
    """Une vidéo de 10 s ne peut pas produire un « par minute » Observé : on
    publie la fenêtre réellement couverte, jamais une extrapolation."""
    s = statistiques(resultat_fictif())
    assert s["debit_max_fenetre_s"] == 10.0
    assert s["debit_max_par_minute"] == 3


# -- Fiabilité : le cœur honnête de l'export -------------------------------


def test_indicateurs_de_fiabilite_presents():
    """Les trois indicateurs demandés doivent exister, y compris quand ils
    valent None : « non mesuré » est une information, une clé absente n'en
    est pas une."""
    f = indicateurs_fiabilite(resultat_fictif())
    for cle in (
        "nb_tracks_comptes",
        "nb_tracks_vus",
        "duree_vie_moyenne_track_frames",
        "detections_par_frame_moyen",
        "detections_par_frame_max",
    ):
        assert cle in f, f"indicateur de fiabilité manquant : {cle}"


def test_indicateurs_comptent_les_tracks_distincts():
    """Le registre anti-recomptage rend un événement par track : les track_id
    distincts disent combien d'IDENTITÉS ont franchi la ligne."""
    r = resultat_fictif()
    r.evenements.append(Evenement(frame=200, timestamp_s=6.0, x=130.0, y=50.0, track_id=1))
    f = indicateurs_fiabilite(r)
    assert f["nb_tracks_comptes"] == 3
    assert f["evenements_par_track"] == pytest.approx(4 / 3, abs=0.005)


def test_vie_moyenne_d_un_track_non_fabricable():
    """On ne connaît pas l'âge des tracks : `Resultat` ne le transporte pas.

    Un track ne peut être compté qu'une fois (registre anti-recomptage), donc
    un événement ne dit rien de sa durée de vie. Publier une durée de vie
    calculée sur les événements serait inventer une mesure.
    """
    f = indicateurs_fiabilite(resultat_fictif())
    assert f["duree_vie_moyenne_track_frames"] is None
    assert f["nb_tracks_vus"] is None
    avertissements = " ".join(f["avertissements"])
    assert "durée de vie" in avertissements
    assert "Resultat" in avertissements


def test_zero_avec_foule_signale_un_defaut_de_comptage():
    """Le cas réel de la vidéo de référence : 88 personnes en moyenne, 0
    passage. C'est un défaut de mesure, pas une foule vide — le rapport doit
    le dire dans ses mots."""
    f = indicateurs_fiabilite(resultat_video_de_reference())
    messages = " ".join(f["avertissements"])
    assert "0 franchissement" in messages
    assert "défaut de comptage" in messages
    assert "87.4" in messages or "87,4" in messages
    # Aucune contradiction interne : c'est mesuré, c'est juste noir.
    assert f["coherent"] is True


def test_contradiction_detectee_entre_total_et_evenements():
    r = resultat_fictif()
    r.total = 7  # le compteur dit 7, la liste n'en contient que 3
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is False
    assert "divergent" in " ".join(f["avertissements"])


def test_contradiction_detectee_entre_passages_et_presence():
    """Des passages comptés alors que personne n'est présent : l'inverse exact
    du cas de référence, aussi impossible."""
    r = resultat_fictif()
    r.presents_moyen = 0.0
    r.presents_max = 0
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is False


def test_track_id_repete_signale_un_double_comptage():
    """Le verrou anti-recomptage a cédé : le total est gonflé."""
    r = resultat_fictif()
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is True
    r.evenements.append(Evenement(frame=210, timestamp_s=7.0, x=140.0, y=50.0, track_id=2))
    r.total = 4
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is False
    assert "track_id" in " ".join(f["avertissements"])


def test_duree_incoherente_avec_le_nombre_de_frames_signalee():
    """`Resultat.secondes` vient soit de la vidéo, soit du temps de calcul
    selon l'appelant : si les deux ne collent pas, le débit par minute est
    sous-estimé. Le rapport le signale au lieu de publier un faux débit."""
    r = resultat_fictif()
    r.secondes = 120.0  # 300 frames « en » 120 s : 2,5 images/s
    f = indicateurs_fiabilite(r)
    assert "2.5" in " ".join(f["avertissements"]) or "2,5" in " ".join(f["avertissements"])


def test_zero_frame_signale():
    r = resultat_video_de_reference()
    r.nb_frames = 0
    f = indicateurs_fiabilite(r)
    assert f["coherent"] is False
    assert "frame" in " ".join(f["avertissements"])


def test_fiabilite_avec_zero_evenement_ne_plante_pas():
    """Le cas réel : aucun événement, aucune division par zéro."""
    r = resultat_video_de_reference()
    f = indicateurs_fiabilite(r)
    assert f["nb_tracks_comptes"] == 0
    assert f["intervalle_median_evenements_s"] is None
    assert f["frames_par_evenement"] is None
    assert isinstance(f["avertissements"], list)


# -- Export complet sur le cas réel ----------------------------------------


def test_export_complet_sur_zero_evenement(tmp_path):
    """Les deux fichiers, l'un après l'autre, sur le cas réel de la vidéo de
    référence : c'est le chemin que l'interface empruntera au clic."""
    r = resultat_video_de_reference()
    chemins = chemins_par_defaut("D:/videos/manifestation.mp4", tmp_path)
    ecrire_csv(r, chemins["csv"])
    ecrire_json(r, chemins["json"])
    assert chemins["csv"].read_text(encoding="utf-8").strip() == ENTETE_CSV

    d = json.loads(chemins["json"].read_text(encoding="utf-8"))
    assert d["total"] == 0
    assert d["evenements"] == []
    assert d["config"]["ligne"] == [556.0, 90.0, 588.0, 654.0]
    assert d["statistiques"]["personnes_par_minute"] == 0.0
    # La discordance est visible DANS le fichier que le média ouvre.
    assert d["fiabilite"]["detections_par_frame_moyen"] == pytest.approx(87.4)
    assert "défaut de comptage" in " ".join(d["fiabilite"]["avertissements"])


def test_json_expose_la_fiabilite_au_meme_niveau_que_les_stats(tmp_path):
    """Le lecteur ouvre le JSON et doit trouver les indicateurs sans fouiller
    dans `statistiques`."""
    d = json.loads(
        ecrire_json(resultat_fictif(), tmp_path / "r.json").read_text(encoding="utf-8")
    )
    assert d["fiabilite"]["nb_tracks_comptes"] == 3
    assert d["statistiques"]["fiabilite"] == d["fiabilite"]


# -- Chemins ---------------------------------------------------------------


def test_chemins_par_defaut():
    # Le plan d'origine écrivait `d["csv"].endswith(...)`, ce qui suppose une
    # chaîne. Or l'interface fait `chemins['csv'].name` et
    # `chemins['csv'].exists()` : un `str` n'a ni l'un ni l'autre. C'est donc
    # le rapport qui rend des `Path`, et l'assertion qui passe par `str()`.
    d = chemins_par_defaut("D:/videos/manifestation.mp4", "sortie")
    assert str(d["csv"]).endswith("manifestation_head_results.csv")
    assert str(d["json"]).endswith("manifestation_head_results.json")


def test_chemins_par_defaut_sont_des_path_nommees():
    """L'interface affiche `chemins['csv'].name` : il faut bien un `Path`."""
    d = chemins_par_defaut("D:/videos/manifestation.mp4", "sortie")
    for chemin in d.values():
        assert hasattr(chemin, "name") and hasattr(chemin, "exists")
        assert chemin.name.endswith(("_head_results.csv", "_head_results.json"))


def test_chemins_par_defaut_encodent_le_nom_de_la_video(tmp_path):
    d = chemins_par_defaut(str(tmp_path / "mon video.mov"), tmp_path)
    assert d["csv"].name == "mon video_head_results.csv"
    assert d["csv"].parent == tmp_path