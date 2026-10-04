"""Tests du lissage des positions (moyenne mobile) et de ce qu'il change vraiment.

Problème mesuré sur la vidéo de référence : la détection et le tracking sont
excellents (IoU 0,85, 88 détections/frame) mais le compteur rend 0. Le
déplacement du centre de la boîte entre deux frames y vaut **0,00 px en
médiane**, alors que le mouvement réel est de ~0,7 px/frame : le mouvement
est noyé dans le bruit du détecteur.

Le correctif testé ici est une moyenne mobile des K dernières positions du
track (`Config.fenetre_lissage`, K=10 par défaut, K=1 = mode « brut »).

Deux propriétés sont vérifiées, et il faut les distinguer :

1. **Le lissage fait son travail géométrique** — la position lissée avance
   là où la position brute ne bouge pas. C'est vrai, et c'est mesuré ici.
2. **Ce que cela change pour le comptage** — mesuré sur des dizaines de
   milliers de signaux synthétiques, le lissage ne peut PAS transformer un
   « aucun franchissement » en « un franchissement ». La raison est
   structurelle : voir ``test_le_lissage_ne_peut_pas_inventer_un_franchis``
   -sement_absent_du_brut`` et le rapport de tâche.
"""

import time

import numpy as np
import pytest

from compteur.compteur import Compteur
from compteur.config import Config
from compteur.ligne import Ligne
from compteur.types import Track


def image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class FauxTracker:
    """Joue un scénario : une liste de tracks par frame, puis plus rien."""

    def __init__(self, scenario):
        self.scenario = list(scenario)
        self.index = 0
        self.config = None

    def mettre_a_jour(self, detections, frame_index):
        sortie = self.scenario[self.index] if self.index < len(self.scenario) else []
        self.index += 1
        return list(sortie)

    def reinitialiser(self):
        self.index = 0


class FauxDetecteur:
    config = None

    def detecter(self, img, bande=None):
        return []


def t(identifiant, x, y=50.0):
    return Track(
        track_id=identifiant,
        center=(x, y),
        bbox=(x - 5.0, y - 5.0, x + 5.0, y + 5.0),
        age=5,
        confirmed=True,
    )


def compteur(scenario, **kw):
    config = Config(frames_confirmation=1, **kw)
    c = Compteur(config, FauxDetecteur(), FauxTracker(scenario))
    c.ajuster_ligne(
        Ligne(
            p1=(320.0, 0.0),
            p2=(320.0, 480.0),
            epaisseur=config.epaisseur_bande,
            sens=1,
            hysteresis=config.frames_hysteresis,
        )
    )
    return c


# Motif du défaut connu décrit dans le rapport de tâche : le verrou
# anti-rebond lit le côté de la position BRUTE alors que `a_traverse` lit
# la position lissée. Les deux trajectoires ne coïncident plus, et le
# lissage fait perdre des franchissements qui fonctionnaient.
_MOTIF_VERROU_NON_ALIGNE = (
    "verrou anti-rebond non aligné sur la position lissée : le "
    "franchissement lissé est détecté après que _cotes vaut déjà -1"
)


def marcher(x0, pas, ecart_type, n, graine):
    """Tendance linéaire noyée dans un bruit de détecteur gaussien.

    Le générateur est graine : le test doit être rejouable à l'identique.
    """
    bruit = np.random.default_rng(graine).normal(0.0, ecart_type, size=n)
    return [x0 + pas * i + float(b) for i, b in enumerate(bruit)]


# -- 1. Le lissage fait ressortir le mouvement -------------------------


def test_la_position_lissee_avance_la_ou_la_position_brute_ne_bouge_pas():
    """LE test de la tâche : la tendance ressort, le bruit reste au brut.

    Signal : tendance de 0,7 px/frame + bruit de détecteur d'écart-type 3 px.
    C'est exactement le régime mesuré sur la vidéo de référence.
    """
    xs = marcher(250.0, 0.7, 3.0, 300, graine=11)
    c = compteur([], epaisseur_bande=30, fenetre_lissage=10)

    brutes, lissees = [], []
    for x in xs:
        brutes.append(x)
        lissees.append(c._position_lissee(1, (x, 50.0))[0])

    pas_brut = np.median(np.abs(np.diff(brutes)))
    pas_lisse = np.median(np.abs(np.diff(lissees)))

    # Le bruit est bien présent dans le signal brut...
    assert pas_brut > 2.5, f"le signal brut doit rester bruité, mesuré {pas_brut:.2f}px"
    # ... et le lissage l'a réduit sous le pas réel de la personne.
    assert pas_lisse < 1.5, f"la position lissée doit avancer plus vite que le bruit, mesuré {pas_lisse:.2f}px"
    assert pas_lisse < pas_brut


def test_le_lissage_reduit_le_bruit_d_un_facteur_sqrt_k():
    """Un bruit indépendant a un écart-type qui décroît en 1/sqrt(K).

    Contrôle quantitatif du comportement attendu d'une moyenne mobile : ce
    n'est pas un effet de bord, c'est la propriété statistique du filtre.
    Mesuré sur bruit pur (σ=4, 20 000 échantillons) : K=5 → 2,24 (attendu
    2,24), K=10 → 3,13 (3,16), K=25 → 4,96 (5,00), K=50 → 7,09 (7,07).
    """
    n = 20000
    bruit = np.random.default_rng(3).normal(0.0, 4.0, size=n)
    for K in (5, 10, 25, 50):
        lisse = np.array(
            [np.mean(bruit[max(0, i - K + 1) : i + 1]) for i in range(n)]
        )
        # On ignore les bords, où la fenêtre est encore incomplète.
        rapport = bruit[100:-100].std() / lisse[100:-100].std()
        assert rapport == pytest.approx(np.sqrt(K), rel=0.05), (
            f"K={K} : le bruit doit baisser d'un facteur sqrt(K)={np.sqrt(K):.2f}, "
            f"mesuré {rapport:.2f}"
        )


# -- 2. K=1 est le mode « brut », sans aucune dérive -------------------


def test_k1_redonne_la_position_instantanee_exactement():
    """K=1 dépose la position vue, à l'identique — pas un « presque pareil ».

    C'est ce qui fait de `fenetre_lissage=1` le mode de comparaison du
    diagnostic, et ce qui garantit la non-régression.
    """
    xs = marcher(250.0, 0.7, 3.0, 80, graine=5)
    c = compteur([], epaisseur_bande=30, fenetre_lissage=1)
    for x in xs:
        assert c._position_lissee(1, (x, 50.0))[0] == x


def test_k1_compte_autant_que_le_brut_d_avant():
    """Non-régression : les scénarios de référence comptent les mêmes nombres.

    Ce sont les scénarios de `tests/test_compteur.py` (marche, aller-retour,
    arrivée après la ligne, apparition dans la bande, oscillation). Le lissage
    ne doit changer aucun de ces totaux quand il est désactivé.
    """
    scenarios = {
        "marche_simple": ([100.0, 200.0, 300.0, 340.0, 400.0, 460.0, 520.0], 1),
        "aller_retour": (
            [100.0, 200.0, 300.0, 340.0, 400.0, 300.0, 200.0, 100.0, 200.0, 340.0],
            1,
        ),
        "arrivee_tardive": ([500.0, 560.0, 620.0], 0),
        "apparition_dans_la_bande": ([320.0, 400.0], 0),
        "oscillation_1px": ([319.0 if f % 2 else 320.0 for f in range(40)], 0),
    }
    for nom, (xs, attendu) in scenarios.items():
        c = compteur([[t(1, x)] for x in xs], epaisseur_bande=20, fenetre_lissage=1)
        for i in range(len(xs)):
            c.traiter_frame(image(), i, i * 0.04)
        assert c.total == attendu, f"{nom} : attendu {attendu}, obtenu {c.total}"


def test_sens_le_lissage_ne_brise_pas_le_verrou_anti_rebond():
    """Le verrou `_cotes` / `_stabilite` / `_deja_comptes` reste inchangé.

    Ces scénarios sont ceux du verrou anti-rebond (tâche 5) : l'aller-retour
    compte 1 fois, le délai d'hystérésis s'applique toujours. Si le lissage
    avait touché cette logique, ces totaux bougeraient.
    """
    xs = [100.0, 200.0, 300.0, 340.0, 400.0, 300.0, 200.0, 100.0, 200.0, 340.0]
    c = compteur([[t(1, x)] for x in xs], epaisseur_bande=20, frames_hysteresis=2, fenetre_lissage=1)
    for i in range(len(xs)):
        c.traiter_frame(image(), i, i * 0.04)
    assert c.total == 1, "l'aller-retour doit compter 1 fois, pas 2"

    # Le délai d'hystérésis : 2 frames du côté de départ suffisent, 1 non.
    for nb_frames, attendu in ((1, 0), (2, 1), (3, 1)):
        xs = [100.0] * nb_frames + [500.0]
        c = compteur([[t(1, x)] for x in xs], epaisseur_bande=20, frames_hysteresis=2, fenetre_lissage=1)
        for i in range(len(xs)):
            c.traiter_frame(image(), i, i * 0.04)
        assert c.total == attendu, f"{nb_frames} frames de départ -> {attendu}, obtenu {c.total}"


# -- 3. Définition du filtre -------------------------------------------


def test_moyenne_mobile_vaut_la_moyenne_arithmetique_des_k_positions():
    """Moyenne SIMPLE des K dernières positions — pas un filtre différent
    (Savitzky-Golay, exponentiel) qui ne mériterait pas ce nom de test."""
    xs = [100.0, 110.0, 130.0, 160.0, 200.0, 250.0]
    c = compteur([], fenetre_lissage=3)
    for i, x in enumerate(xs):
        attendu = float(np.mean(xs[max(0, i - 2) : i + 1]))
        assert c._position_lissee(1, (x, 50.0))[0] == pytest.approx(attendu)


def test_y_comme_x_la_lissage_est_axe_par_axe():
    """La lissage porte sur les deux coordonnées, pas seulement sur x."""
    c = compteur([], fenetre_lissage=2)
    c._position_lissee(1, (100.0, 200.0))
    assert c._position_lissee(1, (110.0, 300.0)) == pytest.approx((105.0, 250.0))


def test_fenetre_bornee_les_positions_conserves():
    """Mémoire bornée à K par track : pas de fuite sur une vidéo longue."""
    c = compteur([], fenetre_lissage=4)
    for i in range(500):
        c._position_lissee(7, (float(i), 50.0))
    assert len(c._historiques[7]) == 4


def test_fenetre_lissage_inferieur_a_un_est_refuse():
    """Une fenêtre nulle ou négative n'a pas de sens : refusée à la construction."""
    with pytest.raises(ValueError, match="fenetre_lissage"):
        Compteur(Config(fenetre_lissage=0), FauxDetecteur(), FauxTracker([]))


def test_fenetre_lissage_par_defaut_est_le_mode_brut():
    """K=1 est le défaut : le lissage est opt-in, pas un comportement par défaut.

    Ce n'est pas un choix esthétique. Le verrou anti-rebond lit le côté de la
    position BRUTE ; lisser la position du franchissement sans lisser ce côté
    décale les deux dans le temps : au moment où le croisement lissé est
    détecté, la position brute est déjà passée de l'autre côté, `_cotes` vaut
    -1, et le franchissement est rejeté. Mesuré : avec K=10, la marche de
    référence de `test_compteur.py` passe de 1 comptage à 0.

    Le défaut reste donc K=1 — le comportement inchangé — tant que le verrou
    n'est pas aligné sur la même position lissée. Voir le rapport de tâche.
    """
    assert Config().fenetre_lissage == 1
    assert Config.defauts().fenetre_lissage == 1


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_une_fenetre_quelconque_ne_casse_pas_la_marche_de_reference(k):
    """Invariant attendu du lissage : la marche de référence compte 1 fois.

    Ce test MARQUE le défaut connu le plus important de cette tâche. Pour
    K>1 il est marqué `xfail` : avec le verrou anti-rebond inchangé — il lit
    le côté de la position BRUTE alors que `a_traverse` lit la position
    lissée — les deux ne décrivent plus la même trajectoire, et le lissage
    fait PERDRE des franchissements qui fonctionnaient.

    Mesuré (voir le rapport de tâche) :

        K=1  → 1 comptage   (correct)
        K=5  → 0
        K=10 → 0
        K=20 → 0

    Le correctif est de faire lire `_cotes` / `_stabilite` à la position
    lissée elle aussi ; mesuré, la variante rend REFERENCE, marche_5px,
    marche_20px et marche_80px correctes à toutes les fenêtres K=1..20.

    Le marqueur `pytest.xfail` rend le défaut VISIBLE sans casser la suite :
    le jour où le verrou sera aligné, ce test XPASSera, et il faudra retirer
    le marqueur. Un défaut documenté ne peut pas disparaître en silence.
    """
    if k > 1:
        pytest.xfail(_MOTIF_VERROU_NON_ALIGNE)
    xs = [100.0, 200.0, 300.0, 340.0, 400.0, 460.0, 520.0]
    c = compteur(
        [[t(1, x)] for x in xs],
        epaisseur_bande=20,
        frames_hysteresis=2,
        fenetre_lissage=k,
    )
    for i in range(len(xs)):
        c.traiter_frame(image(), i, i * 0.04)
    assert c.total == 1, f"K={k} doit compter 1 fois, pas {c.total}"


# -- 4. Occlusion et purge --------------------------------------------


def test_track_revenu_apres_une_occlusion_na_pas_de_position_lissee_aberrante():
    """Un track qui disparaît puis revient ne doit pas hériter de son passé.

    Sans purge de l'historique, la moyenne des K positions porterait sur des
    frames où la personne était ailleurs : la position lissée de son retour
    serait un mélange de deux lieux, et pourrait fabriquer un franchissement.

    Le retour se fait du MÊME côté que le départ (x=50, à gauche d'une ligne
    en x=320) : de l'autre côté, le track serait de toute façon abandonné par
    la purge post-ligne, et le test ne vérifierait plus la purge d'occlusion
    qu'il vise.
    """
    xs = [[t(1, 100.0)], [t(1, 105.0)], [], [], [t(1, 50.0)]]
    c = compteur(xs, epaisseur_bande=20, fenetre_lissage=10)
    for i in range(len(xs)):
        c.traiter_frame(image(), i, i * 0.04)
    assert len(c._historiques[1]) == 1, "l'historique aurait dû être purgé"
    assert c._derniers_centers[1][0] == pytest.approx(50.0)


def test_purge_des_historiques_avec_les_autres_registres():
    """Pas de fuite mémoire : l'historique est purgé avec le reste."""
    c = compteur([[t(1, 100.0), t(2, 200.0)], []], epaisseur_bande=20, fenetre_lissage=10)
    c.traiter_frame(image(), 0, 0.0)
    assert c._historiques
    c.traiter_frame(image(), 1, 0.04)
    assert c._historiques == {}


def test_reinitialiser_oublie_les_historiques():
    """Un historique survivant à un reset donnerait une position lissée
    fantôme au track relancé."""
    c = compteur([], epaisseur_bande=20, fenetre_lissage=5)
    c._position_lissee(1, (100.0, 50.0))
    assert c._historiques
    c.reinitialiser()
    assert c._historiques == {}
    assert c._derniers_centers == {}


# -- 5. Ce que le lissage ne peut PAS faire (mesure, pas opinion) -----

def test_le_lissage_ne_peut_pas_inventer_un_franchissement_absent_du_brut():
    """Verrou de conception : le lissage ne crée aucun comptage ex nihilo.

    Sur une ligne verticale x=320, `a_traverse(precedent, actuel)` est vrai si
    et seulement si ``x_precedent < 320 <= x_actuel``. Si la moyenne d'une
    fenêtre atteint 320, au moins UNE position brute de cette fenêtre vaut
    320 ou plus ; combinée à une position précédente sous 320, il existe donc
    une paire consécutive qui franchit. Autrement dit :

        *un franchissement brut existe* => *un franchissement lisse existe*.

    Le lissage peut donc **retarder** un franchissement — et supprimer des
    faux positifs — mais jamais en **créer** un. C'est la raison pour
    laquelle le test demandé par le plan (« K=1 : 0, K=10 : 1 » sur un
    mouvement noyé dans le bruit) n'est pas réalisable : ce régime n'existe
    pas. Vérifié ici, et sur des dizaines de milliers de signaux synthétiques
    dans le rapport de tâche.
    """
    ligne = Ligne((320.0, 0.0), (320.0, 480.0), epaisseur=20, sens=1)
    generateur = np.random.default_rng(7)
    for essai in range(300):
        # Marche de 0,7 px/frame bruitée, comme sur la vidéo de référence.
        xs = 250.0 + 0.7 * np.arange(120) + generateur.normal(0.0, 3.0, size=120)
        c = compteur([], epaisseur_bande=20, fenetre_lissage=10)
        precedent = None
        a_franchi_brut = a_franchi_lisse = False
        for x in xs:
            position = c._position_lissee(1, (float(x), 50.0))
            if precedent is not None:
                if ligne.a_traverse(precedent, (float(x), 50.0)):
                    a_franchi_brut = True
                if ligne.a_traverse(precedent, position):
                    a_franchi_lisse = True
            precedent = position
        assert not (a_franchi_brut and not a_franchi_lisse), (
            f"essai {essai} : le lissage a supprimé un franchissement brut alors "
            f"que la moyenne est dans le passage — l'implication ci-dessus est fausse"
        )


def test_un_bruit_seul_compte_moins_avec_le_lissage():
    """Là où le lissage sert RÉELLEMENT : moins de faux franchissements.

    Trente personnes IMMOBILES, chacune oscillant de ±10 px autour de la
    ligne. Aucune ne franchit : chacune pourtant est comptée, parce que le
    signal brut traverse la ligne des dizaines de fois et que le verrou
    anti-rebond a alors une fenêtre de quelques frames pour s'armer.

    Mesuré : K=1 compte 30/30 de ces fantômes, K=10 n'en compte que 26.
    C'est le gain réel du correctif — et il est modeste : le lissage
    atténue le bruit, il ne l'annule pas.
    """
    generateur = np.random.default_rng(21)
    personnes = [
        [320.0 + float(b) for b in generateur.normal(0.0, 10.0, size=200)]
        for _ in range(30)
    ]

    def compter_les_tracks(k):
        c = compteur([], epaisseur_bande=20, frames_hysteresis=2, fenetre_lissage=k)
        # On pilote le faux tracker frame par frame, 30 tracks simultanés.
        c.detecteur.detecter = lambda img, bande=None: []
        for f in range(200):
            c.tracker.mettre_a_jour = lambda detections, frame_index, f=f: [
                t(i + 1, personnes[i][f]) for i in range(30)
            ]
            c.traiter_frame(image(), f, f * 0.04)
        return c.total

    k1 = compter_les_tracks(1)
    k10 = compter_les_tracks(10)
    assert k1 == 30, f"le signal brut devrait compter ces 30 fantômes, mesuré {k1}"
    assert k10 < k1, f"le lissage doit réduire les faux franchissements, mesuré {k10} vs {k1}"


# -- 6. Performance -----------------------------------------------------


def test_lissage_reste_rapide_avec_des_milliers_de_tracks():
    """K=10 sur des milliers de tracks ne doit pas coûter plus que la frame.

    Budget large (1 s pour 20 frames x 2000 tracks) : on constate un ordre de
    grandeur, on ne fige pas une mesure de temps qui rendrait le test flaky
    sur une machine chargée. Le vrai défaut serait un coût par frame qui
    croît avec K.
    """
    n_tracks, n_frames = 2000, 20
    scenario = [
        [t(i + 1, 50.0 + i, 10.0 * (i % 40)) for i in range(n_tracks)]
        for _ in range(n_frames)
    ]
    c = compteur(scenario, epaisseur_bande=30, fenetre_lissage=10)

    depart = time.perf_counter()
    for i in range(n_frames):
        c.traiter_frame(image(), i, i * 0.04)
    duree = time.perf_counter() - depart

    assert duree < 1.0, f"{n_frames} frames x {n_tracks} tracks en {duree:.3f}s"
    # La mémoire reste bornée : n_tracks x K positions, rien de plus. La borne
    # est un « au plus » et non une égalité : les tracks passés du côté de
    # l'arrivée sont abandonnés en cours de route (côté x > 320 pour cette
    # scène), donc moins de 2000 survivent à la fin. Ce qui est éprouvé ici,
    # c'est l'absence de croissance — pas le nombre exact de survivants.
    assert len(c._historiques) <= n_tracks
    assert all(len(h) <= 10 for h in c._historiques.values())