"""Non-régression du tracking : le critère d'appariement, pas son implémentation.

Ces tests sont le verrou qui empêche le backend de revenir à une association par
distance de centre. Ils vivent dans un fichier à part, et non dans
`test_tracker.py`, pour que les 21 tests d'origine restent lisibles comme le
contrat de comportement — et restent modifiables quand le contrat change.

Ce que la vidéo de référence impose
-----------------------------------

Une tête y mesure **29 px** et se déplace de **0,69 px** entre deux frames
consécutives. À cette échelle, l'écart entre deux personnes distinctes est du
même ordre que le déplacement d'une seule personne : c'est pourquoi ByteTrack,
qui associe les centres prédits par Kalman, y crée 3388 tracks pour 589 marches
réelles et rend un compteur à zéro.

L'IoU ne dépend pas de l'échelle du déplacement : deux têtes décalées de 0,69 px
se recouvrent à **0,95**, et deux personnes réellement distinctes à 30 px l'une de
l'autre se recouvrent à **0,25**, sous le seuil. Un seuil unique les sépare ; une
distance de centre ne peut pas.

Ce que ces tests verrouillent
-----------------------------

1. Le déplacement minuscule (0,69 px) conserve l'identité — le cas réel de la
   vidéo, celui qui casse ByteTrack.
2. Deux personnes proches (30 px) ne sont pas confondues — le contre-test, pour
   qu'un critère trop permissif soit attrapé aussi.
3. Une grosse personne qui passe devant une tête lointaine **conserve son
   identité**. C'est le cas qui distingue l'IoU d'une distance de centre sans
   qu'aucun réglage ne les sépare : voir `test_la_grosse_personne_conserve_son_id`
   et `test_aucun_seuil_en_pixels_ne_sauve_la_grosse_personne`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from compteur.config import Config
from compteur.tracker import TrackSuivi, Tracker, _BackendIoU, _matrice_iou
from compteur.types import Detection

# Géométrie relevée sur la vidéo : une tête fait 29 px de côté et se déplace de
# 0,69 px entre deux frames consécutives. Ce sont ces deux nombres qui font
# échouer ByteTrack, et les tests ci-dessous doivent rester ancrés dessus.
TAXE_TETE_PX = 29.0
PAS_TETE_PX = 0.69
# Deux personnes réellement distinctes, séparées de 30 px.
ECART_VOISINS_PX = 30.0
# Une personne proche de la caméra : grosse boîte, grands pas.
TAXE_PROCHE_PX = 200.0
PAS_PROCHE_PX = 20.0


def d(x, y, w=TAXE_TETE_PX, h=TAXE_TETE_PX, score=0.9, class_id=0):
    return Detection(
        x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h),
        score=score, class_id=class_id,
    )


# --------------------------------------------------------------------------
# Cas 1 — le déplacement infime : 0,69 px/frame sur une tête de 29 px
# --------------------------------------------------------------------------


def test_deplacement_infime_conserve_le_meme_track():
    """Le cas exact qui tue ByteTrack : la personne ne bouge presque pas.

    Sur 40 frames, une tête décalée de 0,69 px par frame doit rester UN track.
    Une association par distance de centre n'a aucun pouvoir de séparation ici :
    à cette échelle, la distance entre deux personnes distinctes est du même
    ordre que le déplacement d'une seule personne.
    """
    t = Tracker(Config(frames_confirmation=1))
    for frame in range(40):
        tracks = t.mettre_a_jour([d(400.0 + frame * PAS_TETE_PX, 200.0)], frame)
        assert len(tracks) == 1, f"la personne a été perdue à la frame {frame}"

    assert t.nb_tracks_vus() == 1, (
        f"0,69 px/frame a produit {t.nb_tracks_vus()} identités : "
        "c'est exactement la régression ByteTrack"
    )


def test_deplacement_infime_ne_cree_pas_de_track_par_frame():
    """Non-vacuité : on relève les identifiants, pas seulement leur nombre.

    Un backend qui créerait une track neuve par frame passerait le test
    précédent s'il se contentait de vérifier qu'une track existe. Ici on vérifie
    que c'est la MÊME, en gardant tous les identifiants vus.
    """
    t = Tracker(Config(frames_confirmation=1))
    identifiants = set()
    for frame in range(40):
        for track in t.mettre_a_jour(
            [d(400.0 + frame * PAS_TETE_PX, 200.0)], frame
        ):
            identifiants.add(track.track_id)
    assert len(identifiants) == 1


def test_iou_du_deplacement_infime_passe_le_seuil_de_la_configuration():
    """Le recouvrement mesuré, pas seulement le résultat du tracker.

    Ce test fixe la raison physique du choix : 0,95 d'IoU pour 0,69 px de
    déplacement sur une tête de 29 px. Si le critère d'appariement change, cette
    valeur doit rester au-dessus du seuil par défaut.
    """
    avant = [(0.0, 0.0, TAXE_TETE_PX, TAXE_TETE_PX)]
    apres = [(PAS_TETE_PX, 0.0, TAXE_TETE_PX + PAS_TETE_PX, TAXE_TETE_PX)]
    iou = _matrice_iou(np.array(avant, float), np.array(apres, float))

    assert iou[0, 0] == pytest.approx(0.95, abs=0.01)
    assert iou[0, 0] > Config.defauts().seuil_matching, (
        "le déplacement réel de la vidéo doit rester associable avec la "
        "configuration par défaut"
    )


# --------------------------------------------------------------------------
# Cas 2 — le contre-test : deux personnes proches ne doivent PAS être confondues
# --------------------------------------------------------------------------


def test_deux_personnes_proches_ne_sont_pas_confondues():
    """Deux personnes à 30 px, bougeant peu : deux tracks, pas un.

    Contre-test indispensable du précédent : un critère trop permissif
    (association par centre avec un seuil large) fusionnerait les deux. Les deux
    tests ensemble encadrent le critère.
    """
    t = Tracker(Config(frames_confirmation=1))
    for frame in range(20):
        tracks = t.mettre_a_jour(
            [
                d(400.0 + frame * PAS_TETE_PX, 200.0),
                d(400.0 + ECART_VOISINS_PX + frame * PAS_TETE_PX, 200.0),
            ],
            frame,
        )
        assert len(tracks) == 2, (
            f"les deux voisins ont été confondus à la frame {frame}"
        )

    assert t.nb_tracks_vus() == 2


def test_des_voisins_proches_restent_distincts_a_la_vitesse_reelle_de_la_video():
    """Le versant permissif du critère, à la vitesse réelle de la vidéo.

    À 0,69 px/frame, deux personnes à 30 px sont à 44 fois la distance que
    parcourt une seule personne : le risque ici n'est pas la confusion, c'est
    l'inverse — un critère trop LAX qui fusionnerait les deux. Ce test verrouille
    donc le versant permissif.

    Note : au-delà de ~10 px/frame sur une boîte de 29 px, plus aucune stratégie
    n'associe, car l'IoU de la boîte avec elle-même tombe sous `seuil_matching`.
    Ce n'est pas un défaut du critère mais une limite du seuil à cette échelle,
    et cela ne concerne pas la vidéo de référence (0,69 px/frame). Voir §5 du
    rapport de tâche.
    """
    t = Tracker(Config(frames_confirmation=1))
    for frame in range(20):
        tracks = t.mettre_a_jour(
            [
                d(400.0 + frame * PAS_TETE_PX, 200.0),
                d(400.0 + ECART_VOISINS_PX + frame * PAS_TETE_PX, 200.0),
            ],
            frame,
        )
        assert len(tracks) == 2
    assert t.nb_tracks_vus() == 2


# --------------------------------------------------------------------------
# Cas 3 — l'invariance d'échelle, sur une seule personne
# --------------------------------------------------------------------------


def test_meme_deplacement_relatif_meme_resultat_quelle_que_soit_la_taille():
    """Deux personnes qui bougent de 10 % de leur taille restent un track chacune.

    C'est la moitié « isolée » de l'invariance d'échelle : 10 % de 29 px font
    2,9 px, 10 % de 200 px font 20 px. Une distance de centre doit choisir entre
    les deux échelles ; l'IoU ne voit qu'un rapport, identique dans les deux cas.
    """
    for taille in (TAXE_TETE_PX, TAXE_PROCHE_PX):
        pas = 0.10 * taille
        t = Tracker(Config(frames_confirmation=1))
        for frame in range(10):
            tracks = t.mettre_a_jour(
                [d(400.0 + frame * pas, 200.0, taille, taille)], frame
            )
            assert len(tracks) == 1
        assert t.nb_tracks_vus() == 1, (
            f"déplacement de 10 % d'une boîte de {taille} px : "
            f"{t.nb_tracks_vus()} identités"
        )


def test_iou_ne_depend_pas_de_la_taille_des_boites():
    """Le recouvrement de deux boîtes décalées de 10 % ne dépend pas de leur taille.

    0,82 dans les deux cas. Une distance de centre donne 2,9 px et 20 px pour le
    même phénomène.
    """
    valeurs = []
    for taille in (TAXE_TETE_PX, TAXE_PROCHE_PX):
        pas = 0.10 * taille
        avant = np.array([[400.0, 200.0, 400.0 + taille, 200.0 + taille]], float)
        apres = np.array(
            [[400.0 + pas, 200.0, 400.0 + pas + taille, 200.0 + taille]], float
        )
        valeurs.append(_matrice_iou(avant, apres)[0, 0])

    assert valeurs[0] == pytest.approx(valeurs[1], abs=1e-9)
    assert valeurs[0] > Config.defauts().seuil_matching


# --------------------------------------------------------------------------
# Cas 4 — LE cas discriminant : une grosse personne passe devant une tête
# --------------------------------------------------------------------------
#
# C'est le cas qui sépare l'IoU d'une distance de centre SANS QU'AUCUN RÉGLAGE
# PUISSE LES SÉPARER. La géométrie :
#
#   frame 0 : la grosse (200 px) est seule, centrée en (400, 200).
#   frame 1 : la grosse a avancé de 20 px -> centrée en (420, 200).
#             une tête de 29 px APPARAIT, centrée sur (399,5 ; 199,5) — c'est-à-dire
#             sur la position exacte qu'occupait la grosse une frame plus tôt.
#
# Par distance de centre, la grosse est à 0,71 px de la tête et à 20 px de sa
# propre position suivante : elle s'associe donc à la tête. Ce n'est pas un
# réglage, c'est la géométrie — l'erreur est plus petite que le déplacement
# réel, donc AUCUN seuil en pixels ne l'évite.
#
# Par recouvrement, la grosse recouvre sa position suivante à 0,82 et la tête à
# 0,02 : elle s'associe à la bonne personne, et la tête devient un track neuf.


def _scene_grosse_et_tete(frame: int):
    """Détections de la scène décrite ci-dessus."""
    if frame == 0:
        return [d(300.0, 100.0, TAXE_PROCHE_PX, TAXE_PROCHE_PX)]
    return [
        d(
            300.0 + frame * PAS_PROCHE_PX, 100.0,
            TAXE_PROCHE_PX, TAXE_PROCHE_PX,
        ),
        d(385.0, 185.0, TAXE_TETE_PX, TAXE_TETE_PX),
    ]


def test_la_grosse_personne_conserve_son_id():
    """La grosse personne garde son identité malgré la tête sur son ancien centre.

    Non-vacuité : on relève l'identifiant qui suit la GROSSE à chaque frame, et on
    exige qu'il ne change jamais. Compter les tracks ne suffirait pas — un
    backend qui échange les deux identifiants entre les frames 1 et 2 garde le bon
    nombre de tracks tout en perdant la grosse personne. C'est exactement ce que
    fait la distance de centre, et c'est ce que ce test attrape.
    """
    t = Tracker(Config(frames_confirmation=1))
    vus = []
    for frame in range(4):
        tracks = t.mettre_a_jour(_scene_grosse_et_tete(frame), frame)
        gros = [
            x for x in tracks
            if abs(x.center[0] - (400.0 + frame * PAS_PROCHE_PX)) < 1.0
        ]
        assert len(gros) == 1, f"la grosse a disparu à la frame {frame}"
        vus.append(gros[0].track_id)

    assert vus == [vus[0]] * len(vus), (
        f"la grosse personne a changé d'identifiant : {vus}"
    )


def test_la_tete_qui_apparait_est_une_nouvelle_personne():
    """Le contre-test de la scène : la tête tardive est bien une nouvelle track.

    Sans ce test, un backend pourrait « sauver » la grosse en fusionnant les deux
    et en ne renvoyant qu'un seul track — la grosse garderait alors son ID, et
    `test_la_grosse_personne_conserve_son_id` passerait quand même.
    """
    t = Tracker(Config(frames_confirmation=1))
    t.mettre_a_jour(_scene_grosse_et_tete(0), 0)
    tracks = t.mettre_a_jour(_scene_grosse_et_tete(1), 1)

    assert len(tracks) == 2, "la grosse et la tête doivent être deux personnes"
    assert t.nb_tracks_vus() == 2


def test_la_geometrie_du_cas_discriminant_est_bien_celle_decrite():
    """La scène ne dérive pas : les deux rapports sont bien à l'opposé.

    Centre : la tête est 20 fois plus PROCHE que la position suivante (0,71 contre
    20 px). Recouvrement : la position suivante est 40 fois plus recouvrante
    (0,82 contre 0,02). Les deux critères donnent donc des réponses opposées, et
    c'est ce qui rend le test non-trivial.
    """
    grosse = (300.0, 100.0, 500.0, 300.0)
    meme_personne = (320.0, 100.0, 520.0, 300.0)
    tete = (385.0, 185.0, 414.0, 214.0)

    def centre(boite):
        return ((boite[0] + boite[2]) / 2.0, (boite[1] + boite[3]) / 2.0)

    def iou(a, b):
        return _matrice_iou(np.array([a], float), np.array([b], float))[0, 0]

    vers_tete = math.dist(centre(grosse), centre(tete))
    vers_meme = math.dist(centre(grosse), centre(meme_personne))
    assert vers_tete < vers_meme, (
        "la tête doit être plus proche que la position suivante"
    )
    assert vers_meme / vers_tete > 20.0

    assert iou(grosse, meme_personne) > iou(grosse, tete) * 10.0, (
        "le recouvrement doit préférer nettement la position suivante"
    )


# --------------------------------------------------------------------------
# La démonstration par l'exécution : la distance de centre perd la grosse
# --------------------------------------------------------------------------


class BackendDistanceCentre:
    """Backend de substitution : association par distance de centre.

    Il ne varie que sur le CHOIX DU SCORE — l'enveloppe gloutonne sur score
    décroissant, le seuil, la forme de sortie sont identiques à `_BackendIoU`.
    C'est donc la mutation exacte qu'on veut éprouver, exécutable en toute
    rigueur, et non une strawman affaiblie.
    """

    def __init__(self, config: Config, seuil_px: float) -> None:
        self.config = config
        self.seuil_px = seuil_px
        self._suivant = 1
        self._vives: dict[int, tuple] = {}

    def _score(self, track, detection) -> float:
        cx_t, cy_t = (track[0] + track[2]) / 2, (track[1] + track[3]) / 2
        cx_d, cy_d = (
            (detection.x1 + detection.x2) / 2,
            (detection.y1 + detection.y2) / 2,
        )
        distance = math.hypot(cx_t - cx_d, cy_t - cy_d)
        if math.isinf(self.seuil_px):
            # « Le plus proche gagne », sans aucun filtrage : la forme
            # canonique de l'appariement par centre. Un scoreconstant ne
            # conviendrait pas — toutes les paires-valent 1,0, et l'ordre de
            # tri départagerait alors les ex æquo par indice, ce qui ferait
            # gagner le mutant par un accident d'ordre, pas par sa métrique.
            return 1.0 / (1.0 + distance)
        # Score dans ]0, 1] : 1 quand les centres coïncident, 0 dès que la
        # distance atteint le seuil en pixels.
        return max(0.0, 1.0 - distance / self.seuil_px)

    def mettre_a_jour(self, detections: list[Detection]) -> list[TrackSuivi]:
        paires = sorted(
            (
                (self._score(boite, detection), track_id, index)
                for track_id, boite in self._vives.items()
                for index, detection in enumerate(detections)
            ),
            key=lambda p: -p[0],
        )
        apparies: dict[int, int] = {}
        pris: set[int] = set()
        for score, track_id, index in paires:
            if score < self.config.seuil_matching:
                break
            if track_id in apparies or index in pris:
                continue
            apparies[track_id] = index
            pris.add(index)

        for track_id, index in apparies.items():
            self._vives[track_id] = (
                detections[index].x1,
                detections[index].y1,
                detections[index].x2,
                detections[index].y2,
            )
        for index, detection in enumerate(detections):
            if index not in pris:
                self._vives[self._suivant] = (
                    detection.x1, detection.y1, detection.x2, detection.y2,
                )
                self._suivant += 1

        return [
            TrackSuivi(
                track_id=track_id,
                bbox=boite,
                center=((boite[0] + boite[2]) / 2.0, (boite[1] + boite[3]) / 2.0),
                age=0,
                nb_hits=1,
            )
            for track_id, boite in sorted(self._vives.items())
        ]

    def reinitialiser(self) -> None:
        self._suivant = 1
        self._vives.clear()


# Seuils couvrant tout le spectre : serré (n'associe que ce qui ne bouge presque
# pas), intermédiaire, large (fusionne tout ce qui est proche), et l'infini —
# « le plus proche gagne », la forme canonique de l'appariement par centre.
SEUILS_PIXELS = [20.0, 40.0, 100.0, 500.0, float("inf")]


def _suit_la_grosse(backend, frames: int = 4):
    """Identifiants portés par la grosse personne, frame par frame."""
    t = Tracker(Config(frames_confirmation=1), backend=backend)
    vus = []
    for frame in range(frames):
        tracks = t.mettre_a_jour(_scene_grosse_et_tete(frame), frame)
        gros = [
            x for x in tracks
            if abs(x.center[0] - (400.0 + frame * PAS_PROCHE_PX)) < 1.0
        ]
        vus.append(gros[0].track_id if gros else None)
    return vus


@pytest.mark.parametrize("seuil_px", SEUILS_PIXELS, ids=lambda v: f"seuil-{v}px")
def test_la_distance_de_centre_perd_la_grosse_personne_a_tout_seuil(seuil_px):
    """Preuve par l'exécution qu'aucun réglage ne remplace l'IoU.

    On fait tourner la scène de la grosse personne et de la tête sur le mutant
    « distance de centre », à chaque seuil — jusqu'à l'infini, c'est-à-dire
    « le plus proche gagne sans filtrer », la forme canonique de l'appariement
    par centre. La grosse y change d'identifiant à TOUS les seuils, alors que
    l'IoU la garde à tous (test précédent). Il n'existe donc aucun point de
    réglage où les deux stratégies seraient équivalentes : ce n'est pas un
    réglage à trouver, c'est la métrique qui est fausse.

    L'assertion `vus[0] is not None` empêche le test de passer par un vide : un
    mutant qui ne suivrait personne passerait sinon `vus != [vus[0]] * 4` sans
    avoir rien démontré.
    """
    vus = _suit_la_grosse(
        BackendDistanceCentre(Config(frames_confirmation=1), seuil_px)
    )
    assert vus[0] is not None, (
        "la grosse n'a jamais été suivie : le mutant n'est pas rompu, "
        "le test ne prouve rien"
    )
    assert vus != [vus[0]] * len(vus), (
        f"avec un seuil de {seuil_px} px, le mutant a conservé l'identité "
        f"({vus}) : ce seuil-là ferait croire que la distance de centre "
        "remplace l'IoU"
    )


def test_l_iou_tient_la_scene_la_ou_la_distance_de_centre_echoue():
    """Le backend de production sur exactement la même scène : il garde l'identité.

    Symétrique du précédent : on ne se contente pas de constater que la distance
    de centre échoue, on vérifie que le backend livré réussit sur la même scène.
    C'est ce test qui verrouille la correction.
    """
    vus = _suit_la_grosse(None)
    assert vus[0] is not None
    assert vus == [vus[0]] * len(vus), (
        f"l'IoU a perdu la grosse personne : {vus}"
    )


# --------------------------------------------------------------------------
# Cas 5 — les trois règles de l'appariement glouton, prises une par une
# --------------------------------------------------------------------------
#
# Les tests ci-dessus prouvent que l'IoU est le bon CRITÈRE. Les suivants
# prouvent que l'ALGORITHME est correct — sans quoi un critère juste appliqué
# deux fois à la même paire, ou appliqué sans seuil, donnerait encore un
# compteur faux. Un test par règle : la plus recouvrante d'abord, une
# détection et une track ne servent qu'une fois, et rien ne s'associe sous le
# seuil.


def test_la_detection_la_plus_recouvrante_est_privilegiee():
    """Un track faced à deux détections prend celle qu'il recouvre le plus.

    Scénario : la personne se déplace de 2 px entre deux frames. La première
    boîte est exactement sur elle (IoU 1,00), la seconde est déjà décalée
    (IoU 0,92). Le track doit se poser sur la première. C'est le principe même
    du glouton sur score décroissant.
    """
    t = Tracker(Config(frames_confirmation=1))
    t.mettre_a_jour([d(0.0, 0.0, 50.0, 50.0)], 0)
    tracks = t.mettre_a_jour(
        [d(0.0, 0.0, 50.0, 50.0), d(2.0, 0.0, 52.0, 50.0)], 1
    )
    levee = [x for x in tracks if abs(x.center[0] - 25.0) < 0.5]
    assert len(levee) == 1, (
        "le track doit se poser sur la détection la plus recouvrante "
        f"(centres vus : {[round(x.center[0], 1) for x in tracks]})"
    )


def test_une_detection_ne_sert_qu_a_un_seul_track():
    """Deux tracks, une seule détection : un seul prend, l'autre reste en place.

    Scénario : deux personnes qui se chevauchent, dont l'une est momentanément
    occultée au point de n'être plus détectée. Une seule boîte revient. Si le
    track occulté pouvait « voler » celle du track visible, les deux
    identifiants partiraient sur la même personne — le compteur la compterait
    deux fois au prochain passage de ligne.
    """
    t = Tracker(Config(frames_confirmation=1))
    tracks = t.mettre_a_jour(
        [d(0.0, 0.0, 100.0, 100.0), d(20.0, 0.0, 100.0, 100.0)], 0
    )
    assert len(tracks) == 2
    par_position = sorted(tracks, key=lambda x: x.center[0])
    gauche, droite = par_position

    # Une seule détection revient, sur la personne de gauche. Les deux tracks
    # la recouvrent (1,00 et 0,67 — les deux au-dessus du seuil) : c'est la
    # concurrence qui rend le test pertinent. La plus recouvrante gagne, et
    # l'autre reste sur sa dernière position connue.
    tracks = t.mettre_a_jour([d(0.0, 0.0, 100.0, 100.0)], 1)
    vus = {x.track_id: round(x.center[0], 1) for x in tracks}
    assert vus[gauche.track_id] == pytest.approx(50.0, abs=0.5), (
        "le track le plus recouvrant doit prendre la détection"
    )
    assert vus[droite.track_id] == pytest.approx(70.0, abs=0.5), (
        f"le track concurrent a volé la détection de l'autre : {vus}"
    )


def test_rien_ne_s_associe_sous_le_seuil():
    """Une détection sous le seuil ne doit pas être prise, même par défaut.

    Scénario : une personne est suivie, et une boîte apparaît 35 px à côté —
    recouvrement 0,18, très sous le seuil de 0,5. Si le seuil n'était pas
    appliqué, le track sauterait sur cette boîte et la vraie personne
    disparaîtrait. Le seuil est le garde-fou contre les fausses détections,
    autant que contre les détections trop permissives.
    """
    t = Tracker(Config(frames_confirmation=1))
    t.mettre_a_jour([d(0.0, 0.0, 50.0, 50.0)], 0)

    tracks = t.mettre_a_jour([d(35.0, 0.0, 85.0, 50.0)], 1)
    original = [x for x in tracks if abs(x.center[0] - 25.0) < 0.5]
    assert len(original) == 1, (
        "le track d'origine a été abandonné sur une détection sous le seuil : "
        f"centres vus {[round(x.center[0], 1) for x in tracks]}"
    )
    # La fausse détection devient bien un track à part entière : elle n'est pas
    # perdue, elle est simplement comptée comme quelqu'un d'autre.
    assert t.nb_tracks_vus() == 2, (
        "la détection sous le seuil doit créer son propre track, pas être "
        "absorbée par le track existant"
    )


def test_la_detection_la_plus_recouvrante_passe_le_seuil_de_la_configuration():
    """Garde-fou numérique : les scénarios ci-dessus restent sous le seuil réel.

    Si quelqu'un change les géométries de test sans vérifier qu'elles restent
    dans la configuration livrée, les trois tests précédents deviennent
    décorations : ils passeraient pour une raison absente du comportement réel.
    """
    seuil = Config.defauts().seuil_matching
    # 1,00 et 0,67 : les deux pistes de `test_une_detection_ne_sert_qu_a_un_seul_track`
    # doivent rester au-dessus du seuil, sinon ce test ne prouve plus la
    # concurrence mais le refus d'associer.
    assert _matrice_iou(np.array([[0.0, 0.0, 100.0, 100.0]], float),
                        np.array([[0.0, 0.0, 100.0, 100.0]], float))[0, 0] >= seuil
    assert _matrice_iou(np.array([[20.0, 0.0, 120.0, 100.0]], float),
                        np.array([[0.0, 0.0, 100.0, 100.0]], float))[0, 0] >= seuil
    # 0,18 : la piste de `test_rien_ne_s_associe_sous_le_seuil` doit rester
    # SOUS le seuil, sinon ce test ne prouve plus le refus d'associer.
    assert _matrice_iou(np.array([[0.0, 0.0, 50.0, 50.0]], float),
                        np.array([[35.0, 0.0, 85.0, 50.0]], float))[0, 0] < seuil


def test_le_backend_de_production_est_bien_l_iou():
    """Garde-fou : le backend par défaut n'a pas glissé vers une autre stratégie.

    Sans ce test, quelqu'un pourrait remplacer `_BackendIoU` par ByteTrack ou par
    une distance de centre, et les autres tests de ce fichier continueraient de
    passer tant qu'ils sont exécutés contre un faux backend injecté.
    """
    assert isinstance(Tracker(Config(frames_confirmation=1)).backend, _BackendIoU)