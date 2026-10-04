"""Bande de détection de 200 px — trois propriétés, pas une matrice de cas.

Ce que le rognage apporte, mesuré sur 3000 frames de la vidéo de référence
(ligne verticale x=640, `sens=-1`) : **231 personnes comptées sur l'image
entière, 256 sur une bande de 200 px**. Le gain vient du fait que le tracker
n'a plus à distinguer des dizaines de personnes qui se gênent entre elles
hors de la zone qui compte.

Trois tests seulement, choisis pour ce qu'ils prouvent :

1. le rognage découpe bien l'image, et les coordonnées reviennent dans le
   repère de l'image pleine — sans quoi le décompte est faux sans qu'aucun
   signe n'apaisse à l'écran ;
2. l'abandon des tracks après la ligne ne change RIEN au décompte (mesuré :
   256 avec et sans) ;
3. la bande suit la ligne quand on la déplace — « un seul objet », pas deux.

Le reste — bande débordante, ligne hors cadre, verrouillage — est couvert par
les tests existants de `Ligne` et par l'usage.
"""

import numpy as np
import pytest

from compteur.compteur import Compteur
from compteur.config import BANDE_APRES_PX, BANDE_AVANT_PX, Config
from compteur.detecteur import Detecteur
from compteur.ligne import Ligne
from compteur.types import Detection, Track

LARGEUR, HAUTEUR = 1280, 720


def image():
    return np.zeros((HAUTEUR, LARGEUR, 3), dtype=np.uint8)


def ligne_bandee(x=640.0, sens=1):
    """Ligne verticale en `x`, avec la bande de détection de production."""
    return Ligne(
        p1=(x, 0.0),
        p2=(x, float(HAUTEUR)),
        epaisseur=30,
        sens=sens,
        hysteresis=2,
        bande_avant_px=BANDE_AVANT_PX,
        bande_apres_px=BANDE_APRES_PX,
    )


# -- 1. Le rognage et le retour des coordonnées --------------------------


class ModeleEspion:
    """Faux modèle : note la forme de l'image reçue, renvoie une boîte fixe."""

    def __init__(self, boite=(10.0, 20.0, 40.0, 60.0)):
        self.forme_recue = None
        self.boite = boite
        self.nom = {}

    def __call__(self, img, **kw):
        self.forme_recue = img.shape
        x1, y1, x2, y2 = self.boite

        class Boites:
            xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            conf = np.array([0.9], dtype=np.float32)
            cls = np.array([0], dtype=np.float32)

            def __len__(self):
                return 1

        class Resultat:
            boxes = Boites()

        return [Resultat()]


def test_la_bande_rogne_l_image_et_les_coordonnees_reviennent_justes():
    """Le point qui rend la bande UTILISABLE : les coordonnées sont corrigées.

    Le modèle ne voit que la bande, donc ses boîtes sont relatives à la coupe.
    Sans le décalage ajouté ici, cette boîte atterrirait à x=10 dans l'image
    pleine alors que la personne est à x=650 : le tracker la ferait passer à
    côté de la ligne, et le décompte serait faux SANS RIEN MONTRER à l'écran,
    puisque l'affichage se fait elle aussi en coordonnées d'image pleine.
    C'est la famille de bugs la plus coûteuse du projet : invisible à l'œil,
    fausse au chiffrage.
    """
    modele = ModeleEspion()
    detecteur = Detecteur(modele, Config())
    lgn = ligne_bandee(x=640.0)

    rect = lgn.rect_bande_detection(LARGEUR, HAUTEUR)
    # sens=1 : le cote de DEPART (coordonnee positive) est a gauche de x=640,
    # donc 200 px avant et 100 px apres -> 440..740, soit 300 px au total.
    assert rect == (640 - 200, 0, 640 + 100, HAUTEUR), (
        "la bande doit faire 200 px avant et 100 px apres la ligne"
    )

    detections = detecteur.detecter(image(), rect)

    # Le modèle a bien vu la COUPE, pas l'image entière : c'est le rognage.
    assert modele.forme_recue == (HAUTEUR, BANDE_AVANT_PX + BANDE_APRES_PX, 3), (
        f"le modèle a reçu {modele.forme_recue}, la bande n'a pas été appliquée"
    )
    # Et la boîte est revenue dans le repère de l'image pleine : le modèle a
    # rendu x∈[10,40] DANS LA BANDE, donc x∈[450,480] sur l'image de 1280 —
    # c'est le BORD de la bande qui sert de décalage, pas le centre de l'image.
    d = detections[0]
    assert (d.x1, d.x2) == (450.0, 480.0), "coordonnées x non revenues"
    assert (d.y1, d.y2) == (20.0, 60.0), "coordonnées y non revenues"


def test_sans_bande_le_modele_voit_toute_l_image():
    """Contre-test : `None` doit vraiment significr « pas de rognage ».

    Sans ce contre-test, une bande qui ne serait jamais appliquée passerait le
    test précédent — il ne verrait qu'un chemin de code qui fonctionne.
    """
    modele = ModeleEspion()
    detecteur = Detecteur(modele, Config())
    detecteur.detecter(image(), None)
    assert modele.forme_recue == (HAUTEUR, LARGEUR, 3)


# -- 2. L'abandon après la ligne ne change pas le décompte ----------------


class FauxTracker:
    """Joue un scénario de tracks, puis plus rien."""

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


def _compteur(scenario, bande, purger):
    """Un compteur sur le même scénario, avec ou sans abandon post-ligne."""
    c = Compteur(Config(frames_hysteresis=2), FauxDetecteur(), FauxTracker(scenario))
    c.ajuster_ligne(bande)
    # Désactive la purge pour le témoin « sans abandon ».
    if not purger:
        c._purger_franchis = lambda: None
    for i in range(len(scenario)):
        c.traiter_frame(image(), i, i * 0.04)
    return c


def test_abandon_apres_la_ligne_ne_change_pas_le_decompte():
    """La purge post-ligne doit être GRATUITE. Mesurée : 256 avec et sans.

    Vingt personnes marchent vers une ligne verticale en x=320. Chacune est
    comptée une fois, se retrouve de l'autre côté, et ses entrées de suivi
    deviennent mortes. Avec l'abandon, elles sont effacées dès qu'elles sont
    arrivées ; sans, elles restent jusqu'à ce que le tracker les perde.

    Les deux décomptes doivent être identiques. C'est LA propriété qui justifie
    la purge : si elle changeait un seul comptage, elle ne serait pas gratuite.
    """
    # Vingt personnes, chacune franchissant la ligne à son tour : c'est le
    # volume qui rend la purge significative — le nombre de tracks.highs en
    # mémoire, pas le décompte, qui est identique par construction.
    xs = [100.0, 200.0, 250.0, 280.0, 300.0, 310.0, 340.0, 400.0, 500.0]
    scenario = [
        [t(i, xs[f] + 10.0 * i) for i in range(20)]
        for f in range(len(xs))
    ]

    avec = _compteur(scenario, ligne_bandee(x=320.0), purger=True)
    sans = _compteur(scenario, ligne_bandee(x=320.0), purger=False)

    assert avec.total == sans.total, (
        f"l'abandon post-ligne a changé le décompte : {avec.total} contre {sans.total}"
    )
    assert avec.total > 0, "le scénario doit faire traverser quelqu'un"
    assert avec.total == len(avec._deja_comptes), "un comptage par personne"
    # Le journal anti-recomptage ne doit pas dépendre de la purge : c'est lui,
    # et lui seul, qui empêche un aller-retour de compter deux fois.
    assert avec._deja_comptes == sans._deja_comptes


def test_abandon_presque_la_purge_de_registres_vides():
    """Une fois arrivée, une personne ne doit plus laisser d'état de suivi.

    C'est le bénéfice concret de la purge : les registres ne grossissent plus
    sur une vidéo de dix mille passages. Sans elle, chaque personne garderait
    quatre entrées jusqu'à ce que le tracker l'oublie — c'est-à-dire
    `survie_max` frames de plus.
    """
    xs = [100.0, 200.0, 280.0, 300.0, 310.0, 340.0, 400.0]
    c = _compteur([[t(1, x)] for x in xs], ligne_bandee(x=320.0), purger=True)

    assert c.total == 1
    # La personne est arrivée (x=400 > 320) : plus aucune entrée de suivi.
    assert 1 not in c._derniers_centers
    assert 1 not in c._cotes
    assert 1 not in c._stabilite
    # MAIS elle reste dans le journal anti-recomptage, lui n'est jamais purgé.
    assert 1 in c._deja_comptes


# -- 3. La bande suit la ligne -------------------------------------------


def test_la_bande_suit_la_ligne_quand_on_la_deplace():
    """« Un seul objet » : la bande n'est pas une coordonnée à côté.

    On déplace la ligne de x=640 à x=900 et le rectangle de détection doit
    suivre. Si la bande était stockée comme un rectangle, il faudrait le mettre
    à jour à la main — et le jour où on oublierait, le modèle compterait une
    colonne pendant que l'opérateur en regarde une autre. Aucune erreur à
    l'écran : c'est ce qui rend ce genre d'oubli si grave.
    """
    lgn = ligne_bandee(x=640.0)
    assert lgn.rect_bande_detection(LARGEUR, HAUTEUR) == (440, 0, 740, HAUTEUR)

    assert lgn.deplacer((900.0, 0.0), (900.0, float(HAUTEUR))) is True

    assert lgn.rect_bande_detection(LARGEUR, HAUTEUR) == (700, 0, 1000, HAUTEUR), (
        "la bande n'a pas suivi la ligne"
    )


def test_la_bande_ne_bouge_plus_quand_la_ligne_est_verrouillee():
    """Verrouillé au lancement, donc figé pendant l'analyse.

    Sans cela, un réglage glissé en cours de lecture déplacerait la zone de
    comptage sous les yeux de l'opérateur : le décompte afficherait deux zones
    différentes dans la même vidéo, et le chiffre ne vaudrait plus rien.
    """
    lgn = ligne_bandee(x=640.0)
    lgn.verrouiller()

    assert lgn.deplacer((900.0, 0.0), (900.0, float(HAUTEUR))) is False
    assert lgn.rect_bande_detection(LARGEUR, HAUTEUR) == (440, 0, 740, HAUTEUR)
    assert lgn.p1 == (640.0, 0.0), "un déplacement refusé ne doit rien écrire"

    lgn.deverrouiller()
    assert lgn.deplacer((900.0, 0.0), (900.0, float(HAUTEUR))) is True


# -- Le voile d'affichage --------------------------------------------------


def test_le_voile_assombrit_hors_de_la_bande_et_pas_dedans():
    """L'opérateur voit la colonne qui compte, la scène reste lisible.

    Un voile trop fort (masque noir) ferait perdre le contexte : l'opérateur ne
    verrait plus ce qu'il a placé. Un voile absent ne signalerait rien du tout.
    On vérifie donc les DEUX côtés : assombri dehors, intact dedans.
    """
    from interface.overlay import dessiner
    from compteur.types import FrameResult

    img = np.full((HAUTEUR, LARGEUR, 3), 200, dtype=np.uint8)
    lgn = ligne_bandee(x=640.0)
    # Une boîte de détection posée HORS de la bande : elle doit être
    # assombrie elle aussi, sinon l'opérateur la verrait plus bright que le
    # reste et croirait qu'elle compte.
    r = FrameResult(image=img, detections=[Detection(100, 100, 140, 160, 0.9, 0)])
    sortie = dessiner(img, r, ligne=lgn)

    # x=600 est dans la bande de DÉTECTION (440..740) mais hors de la bande de
    # franchissement (30 px autour de 640) : c'est là qu'on mesure le voile,
    # sans confusion avec l'aperçu translucide que la ligne dessine dessus.
    dans_bande = sortie[50, 600].astype(np.int16).mean()
    hors_bande = sortie[50, 200].astype(np.int16).mean()

    assert dans_bande == pytest.approx(200.0, abs=1), "la bande doit rester intacte"
    assert hors_bande < 160, f"le hors-bande doit être assombri, mesuré {hors_bande}"
    assert hors_bande > 100, f"le voile ne doit pas masquer la scène : {hors_bande}"


# -- La dissymétrie 200 / 100 ---------------------------------------------


def test_la_bande_est_dissymetrique_et_du_bon_cote():
    """200 px du côté d'où viennent les gens, 100 px de l'autre.

    C'est le test qui verrouille le SENS de la dissymétrie, pas seulement son
    existence. Les deux `sens` sont éprouvés parce qu'ils sont le piège de
    cette géométrie : les deux moitiés d'une bande symétrique sont
    interchangeables, celles d'une bande dissymétrique ne le sont PAS.

    Avec `sens=+1`, le côté de départ (coordonnée positive sur la normale) est
    à gauche de la ligne ; avec `sens=-1`, il est à droite. Une implémentation
    qui inverserait les deux largeurs passerait un test qui n'en vérifie qu'un
    des deux sens — et compterait 100 px de présence avant au lieu de 200,
   l'erreur serait symétrique et sans aucun signe à l'écran.
    """
    gauche_avant = ligne_bandee(x=640.0, sens=1).rect_bande_detection(LARGEUR, HAUTEUR)
    droite_avant = ligne_bandee(x=640.0, sens=-1).rect_bande_detection(LARGEUR, HAUTEUR)

    assert gauche_avant == (640 - 200, 0, 640 + 100, HAUTEUR)
    assert droite_avant == (640 - 100, 0, 640 + 200, HAUTEUR)

    # Et la bande fait bien 300 px au total, pas 200 : c'est ce qui distingue
    # « dissymétrique » de « symétrique plus étroite ».
    assert (gauche_avant[2] - gauche_avant[0]) == 300
    assert (droite_avant[2] - droite_avant[0]) == 300
