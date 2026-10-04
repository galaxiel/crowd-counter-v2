"""Réglages du compteur : dataclasses, sérialisation JSON, valeurs par défaut."""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from dataclasses import asdict, dataclass, fields, replace

log = logging.getLogger(__name__)


def _racine() -> pathlib.Path:
    """Dossier racine des données de l'application.

    Deux régimes, et ils n'ont rien en commun :

    - **Depuis les sources** (`python main.py`) : le dépôt, déduit de
      l'emplacement de ce fichier — deux niveaux au-dessus de `compteur/`.
    - **Sous PyInstaller** (`sys.frozen`) : le fichier n'est pas dans le
      dossier de l'application. En mode *onefile*, `__file__` pointe vers le
      dossier temporaire d'extraction (`%TEMP%\\_MEIxxxxxx`), et
      `parent.parent` ne remonte nulle part : `config/default.json` serait
      introuvable et `Config.defauts()` partirait sur un repli silencieux.
      Le dossier d'extraction se lit dans `sys._MEIPASS`.

    `sys._MEIPASS` a la priorité : c'est le seul emplacement garanti par le
    fichier `.spec` (`datas=[("config/default.json", "config")]`). Si un
    `one-folder` le définit aussi, les deux points convergent.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return pathlib.Path(meipass)
        # one-folder sans _MEIPASS : le dossier de l'exécutable, où le .spec
        # dépose `config/` en une collection.
        return pathlib.Path(sys.executable).resolve().parent
    return pathlib.Path(__file__).resolve().parent.parent


RACINE = _racine()

#: Modes de choix de la résolution d'analyse, dans l'ordre d'affichage.
#:
#: - `auto` : on suit la résolution réelle de la vidéo, plafonnée à
#:   `PLAFOND_ANALYSE` ;
#: - `manuelle` : l'opérateur fixe lui-même `taille_entree`.
#:
#: Les deux valeurs sont écrites SANS ACCENT : ce sont des identifiants, pas
#: des libellés. Le texte affiché est dans l'interface.
MODE_AUTO = "auto"
MODE_MANUEL = "manuel"
MODES_RESOLUTION: tuple[str, ...] = (MODE_AUTO, MODE_MANUEL)

#: Modes de choix du périphérique de calcul, dans l'ordre d'affichage.
#:
#: - `auto` : CUDA si la machine en a un, CPU sinon. C'est le défaut, et le
#:   seul mode qui convienne sans connaître le matériel de la machine cible ;
#: - `cuda` : le GPU est EXIGÉ. Sur une machine sans NVIDIA, l'analyse bascule
#:   quand même sur le CPU, avec un message — voir
#:   `detecteur.peripherique_effectif` : un message vaut mieux qu'un crash sur
#:   le terrain, et l'opérateur voit immédiatement pourquoi c'est lent ;
#: - `cpu` : le CPU est EXIGÉ, même quand un GPU est présent. C'est le
#:   remède quand le GPU plante sur une scène particulière.
#:
#: Comme pour la résolution, ce sont des IDENTIFIANTS sans accent : ils
## voyagent dans les profils JSON. Les libellés sont dans l'interface.
PERIPHERIQUE_AUTO = "auto"
PERIPHERIQUE_CUDA = "cuda"
PERIPHERIQUE_CPU = "cpu"
PERIPHERIQUES: tuple[str, ...] = (
    PERIPHERIQUE_AUTO,
    PERIPHERIQUE_CUDA,
    PERIPHERIQUE_CPU,
)

#: Plafond de la résolution d'analyse automatique, en pixels.
#:
#: Le coût marginal devient clairement défavorable au-delà. Mesuré sur 3000
#: frames de la vidéo de référence (cf. rapport de tâche 14) :
#:
#: | taille | compté | temps | détec/frame |
#: |--------|---------|-------|--------------|
#: | 640    | 231     | 179 s | 79,1         |
#: | 1280   | 257     | 230 s | 100,6        |
#: | 1920   | 271     | 324 s | 115,1        |
#:
#: 640 -> 1280 : +26 personnes (+11 %) pour +28 % de temps.
#: 1280 -> 1920 : +14 personnes (+5 %) pour +41 % de temps.
#:
#: 1920 n'est donc PAS inutile : c'est ce que la mesure dit, et elle a été
#: faits pour trancher. Mais le dernier cran coûte deux fois plus cher que le
#: précédent pour deux fois moins de gain, sur une vidéo dont la source est
#: déjà en 1280. C'est le coût d'un détail, pas d'une erreur : l'opérateur
#: qui monte en mode manuel paie ce supplément en connaissance de cause.
PLAFOND_ANALYSE = 1280

#: Profondeur de la BANDE DE DÉTECTION de chaque côté de la ligne, en
#: pixels de l'image native.
#:
#: Le modèle ne tourne pas sur l'image entière mais sur une bande
#: **dissymétrique** : 200 px du côté d'où viennent les gens, 100 px de l'autre.
#: Ce n'est pas un réglage : c'est une propriété géométrique de la ligne, elle
#: se déplace donc AVEC elle.
#:
#: **Pourquoi dissymétrique.** Avant la ligne, le tracker a besoin d'espace pour
#: CONSTRUIRE l'identité de quelqu'un qui approche — plusieurs frames, des
#: images successives, pour associer les boîtes à un track stable. Après la
#: ligne, le compte est fait : `a_traverse` exige un côté de départ strictement
#: positif, cette personne ne peut donc plus jamais compter. Il ne reste qu'à
#: la voir passer, et 100 px suffisent. Les 100 px que la bande symétrique
#: gaspillait de ce côté-là ne servaient à rien.
#:
#: « Avant » et « après » ne sont pas des mots de dessin mais le vocabulaire de
#: `compteur.ligne` : côté de DÉPART (coordonnée positive sur la normale,
#: `point_du_cote == +1`) et côté d'ARRIVÉE (coordonnée négative).
#:
#: Mesuré sur 3000 frames de la vidéo de référence (ligne verticale x=640,
#: `sens=-1`, `imgsz=640`), bande alors symétrique :
#:
#: | bande  | détections/frame | compté |
#: |--------|------------------|--------|
#: | entière | 79,1            | 231    |
#: | 200 px  | 13,2            | 256    |
#: | 300 px  | 20,3            | 254    |
#: | 600 px  | 42,2            | 258    |
#:
#: Le gain vient de la restriction elle-même, pas d'une largeur particulière :
#: hors bande, le tracker s'efforce d'associer des dizaines de personnes qui se
#: gênent entre elles (boîtes fusionnées, identités qui permutent) ; en ne
#: gardant que celles qui approchent de la ligne, il suit moins de cibles mais
#: il les suit bien.
#:
#: La bande ne fait PAS gagner de temps de calcul (71 à 76 s quelle que soit la
#: largeur) : le goulot est ailleurs. Ce n'est pas une optimisation de débit.
BANDE_AVANT_PX = 200
BANDE_APRES_PX = 100

#: Durée d'affichage de la boîte verte d'une personne qui vient d'être
#: comptée, en frames TRAITÉES (pas affichées).
#:
#: 100 px de bande après la ligne à 0,69 px/frame mesuré sur la vidéo de
#: référence : 145 frames, soit ~5 s de vidéo. C'est la durée pendant laquelle
#: la personne reste dans la zone utile après son comptage. La borne est donc
#: le TEMPS DE VISIBILITÉ, pas un clignotement : on veut voir la personne
#: s'éloigner, pas la voir disparaître.
DUREE_BOITE_COMPTEE_FRAMES = 145

#: Facteur d'assombrissement appliqué HORS de la bande de détection, à
#: l'affichage uniquement. L'image reste lisible : c'est un voile, pas un
#: masque. 0,6 = 40 % d'assombrissement.
FACTEUR_VOILE = 0.6

#: Résolution d'analyse retenue quand la vidéo ne dit pas la sienne.
#:
#: Une `VideoCapture` qui ne répond pas sur `CAP_PROP_FRAME_WIDTH` (conteneur
#: flux réseau, capture encore ouverte) ne doit pas laisser le champ
#: vide ni faire échouer l'analyse : on retombe sur la valeur historique, qui
#: est aussi celle du plafond pour une vidéo de surveillance courante.
TAILLE_ANALYSE_REPLI = 640


def chemin_defaut_config() -> pathlib.Path:
    """Chemin du fichier de valeurs par défaut, versionné avec le projet.

    Résolu à chaque appel, et non figé dans `RACINE` à l'import : sous
    PyInstaller, `RACINE` est figé avant que quiconque ne sache dans quel
    dossier on a été décompressé, et les tests figent `sys.frozen` bien après
    l'import du module. Recalculer rend le chemin vérifiable.
    """
    return _racine() / "config" / "default.json"


#: Pas de quantum du modèle. YOLO travaille sur des images de côté multiple
#: de 32 ; une taille qui ne l'est pas est réalignée en silence par
#: ultralytics, avec un avertissement que personne ne lit.
QUANTUM_ANALYSE = 32


def taille_entree_automatique(largeur: int, hauteur: int) -> int:
    """Résolution d'analyse d'une vidéo de `largeur` x `hauteur`, en pixels.

    La règle est « la résolution native de la vidéo, plafonnée à
    `PLAFOND_ANALYSE` », et le plafond est MESURÉ (rapport de tâche 14) :

    | vidéo        | analyse à | justification |
    |--------------|-----------|---------------|
    | 640 x 480    | 640       | sous le plafond, on suit la source |
    | 1280 x 720   | 1280      | +11 % de personnes pour +28 % de temps |
    | 1920 x 1080  | 1280      | +5 % de personnes pour +41 % de temps |
    | 3840 x 2160  | 1280      | même source, coût encore plus lourd |

    Le plafond à 1280 ne pretend pas que 1920 est inutile : 1920 trouve bien
    14 personnes de plus (271 contre 257 sur 3000 frames). Il dit que ce cran
    coûte 41 % de temps de calcul supplémentaire pour 5 % de personnes, et
    qu'au-delà le rapport se dégrade encore.

    **Pourquoi le côté le plus grand, et non la largeur.** `imgsz` vaut pour
    le côté le plus LONG de l'image : c'est lui qu'ultralytics aligne sur la
    taille demandée, l'autre côté suit par le rapport d'aspect. Une vidéo
    d'appareil photo portrait (1080 x 1920) a donc sa hauteur pour référence,
    pas sa largeur ; prendre `max` évite de sous-analyser une vidéo dont
    personne n'a choisi l'orientation.

    **Pourquoi arrondir vers le bas.** Le résultat final ne doit JAMAIS
    dépasser la résolution native : grossir une image n'invente aucun détail,
    ça allonge le calcul. 1278 px de large deviennent donc 1216 (le plus
    grand multiple de 32 qui ne dépasse pas 1278), pas 1280.

    Une résolution absente ou absurde (0, capture qui ne répond pas) retombe
    sur `TAILLE_ANALYSE_REPLI` : mieux vaut une valeur connue et médiocre
    qu'une analyse qui ne démarre pas.
    """
    try:
        largeur = int(largeur)
        hauteur = int(hauteur)
    except (TypeError, ValueError):
        return TAILLE_ANALYSE_REPLI
    if largeur <= 0 or hauteur <= 0:
        return TAILLE_ANALYSE_REPLI
    return min(max(largeur, hauteur), PLAFOND_ANALYSE) // QUANTUM_ANALYSE * QUANTUM_ANALYSE


def mode_resolution(connu: str | None) -> str:
    """Normalise un mode de résolution d'analyse.

    Toute valeur inconnue — `None`, chaîne vide, mode d'une version future —
    retombe sur `MODE_AUTO` : c'est le mode qui ne perd personne, puisqu'il
    suit la source au lieu d'imposer un chiffre. Le repli est journalisé.
    """
    if connu in MODES_RESOLUTION:
        return connu
    if connu is not None:
        log.warning(
            "mode de résolution d'analyse inconnu (%r) : repli sur %r.",
            connu,
            MODE_AUTO,
        )
    return MODE_AUTO


def mode_peripherique(connu: str | None) -> str:
    """Normalise un mode de périphérique de calcul.

    Même contrat que `mode_resolution` : toute valeur inconnue — `None`,
    chaîne vide, mode d'une version future — retombe sur `PERIPHERIQUE_AUTO`,
    parce que c'est le seul mode qui marche partout. Le repli est journalisé.
    """
    if connu in PERIPHERIQUES:
        return connu
    if connu is not None:
        log.warning(
            "mode de périphérique inconnu (%r) : repli sur %r.", connu, PERIPHERIQUE_AUTO
        )
    return PERIPHERIQUE_AUTO


@dataclass
class Config:
    # Détection
    # medium.pt : modèle de tête entraîné sur SCUT-HEAD (foule dense). Sur la
    # vidéo de référence il donne des têtes de 29 px contre 21 px pour
    # yolov8n-head.pt, soit 38 % de marches réelles en plus.
    modele: str = "medium.pt"
    seuil_confiance: float = 0.25
    # Une tête détectée sur cette vidéo fait 20-29 px : un seuil de 20 px
    # filtrerait la moitié des détections. 3 px ne filtre que le bruit.
    taille_min_px: int = 3
    classes_retenues: list[int] | None = None
    taille_entree: int = 640
    # Comment `taille_entree` est choisi.
    #
    # - `MODE_AUTO` : on suit la résolution réelle de la vidéo, plafonnée à
    #   `PLAFOND_ANALYSE`. C'est le défaut, parce que le 640 en dur coupait la
    #   moitié d'une vidéo 720p : les têtes y faisaient 14 px au lieu de 29, et
    #   on y perdait 17 % du décompte (mesuré, cf. rapport de tâche 13).
    # - `MODE_MANUEL` : `taille_entree` est pris au pied de la lettre.
    #
    # `taille_entree` reste le champ EFFECTIF : en mode auto, l'appelant
    # l'écrit avec `avec_resolution()` avant de lancer. Le moteur ne connaît
    # donc qu'un seul chiffre, et `Detecteur` n'a pas à savoir d'où il sort.
    resolution_analyse: str = MODE_AUTO
    # Sur quel périphérique tourne la détection.
    #
    # - `PERIPHERIQUE_AUTO` : CUDA si la machine en a un, CPU sinon.
    # - `PERIPHERIQUE_CUDA` : le GPU est exigé ; sans NVIDIA, l'analyse bascule
    #   sur le CPU AVEC un message — un crash sur le terrain coûte plus cher
    #   qu'une analyse lente qu'on peut au moins regarder.
    # - `PERIPHERIQUE_CPU` : le CPU est exigé même si un GPU est présent.
    #   Remède quand le GPU plante sur une scène particulière.
    #
    # Ce réglage ne change PAS le décompte : il ne change que la vitesse (et,
    # à la précision machine près, rien d'autre).
    peripherique: str = PERIPHERIQUE_AUTO
    # Tracker
    frames_confirmation: int = 3
    survie_max: int = 30
    # Une tete de 29 px qui bouge de 15 px/frame a une IoU de 0.32 avec
    # elle-meme : a 0.5 le tracker perdrait la personne. Sur la video de
    # reference le deplacement est de 0.69 px (IoU 0.95), tres au-dessus.
    seuil_matching: float = 0.3
    # Ligne
    ligne: tuple[float, float, float, float] | None = None
    epaisseur_bande: int = 30
    sens: int = 1
    frames_hysteresis: int = 2
    # Lissage
    # Fenêtre de la moyenne mobile des positions d'un track, en frames. Le
    # mouvement réel d'une personne (~0,7 px/frame sur la vidéo de référence)
    # est noyé dans le bruit du détecteur : sans lissage, le test de
    # franchissement porte sur une position qui ne « bouge » pas d'une frame à
    # l'autre (déplacement médian mesuré : 0,00 px). La moyenne des K
    # dernières positions fait ressortir la tendance — le bruit aléatoire
    # s'annule par moyennage, le mouvement constant se cumule.
    # K=1 = mode « brut » : position instantanée, comportement inchangé.
    # K>1 est VOLONTAIREMENT inactif par défaut. Raison mesurée : le lissage
    # retarde la détection du croisement d'une demi-fenetre, mais le verrou
    # anti-rebond lit toujours le côté de la position BRUTE. Au moment où le
    # croisement lissé est détecté, la position brute est déjà passée de
    # l'autre côté : `_cotes` vaut -1 et `_stabilite` 0, donc le verrou
    # REJETTE. Mesuré : la marche de référence passe de 1 comptage (K=1) à 0
    # (K=5, 10, 20). Pour activer K>1, il faut d'abord faire lire
    # `_cotes`/`_stabilite` à la position lissée elle aussi — voir le rapport
    # de tâche 6, section 4.4.
    fenetre_lissage: int = 1

    def vers_dict(self) -> dict:
        d = asdict(self)
        # JSON n'a pas de tuple : on repasse en liste.
        d["ligne"] = list(self.ligne) if self.ligne is not None else None
        return d

    @classmethod
    def depuis_dict(cls, d: dict) -> "Config":
        connus = {f.name for f in fields(cls)}
        inconnus = set(d) - connus
        if inconnus:
            raise ValueError(f"clés de configuration inconnues : {sorted(inconnus)}")
        d = dict(d)
        if d.get("ligne") is not None:
            d["ligne"] = tuple(float(v) for v in d["ligne"])
        if d.get("classes_retenues") is not None:
            d["classes_retenues"] = [int(v) for v in d["classes_retenues"]]
        if "resolution_analyse" in d:
            # Un profil écrit à la main peut porter n'importe quoi ; on
            # normalise plutôt que de refuser le fichier entier.
            d["resolution_analyse"] = mode_resolution(d["resolution_analyse"])
        if "peripherique" in d:
            d["peripherique"] = mode_peripherique(d["peripherique"])
        return cls(**d)

    def avec_resolution(self, largeur: int, hauteur: int) -> "Config":
        """Copie dont `taille_entree` est la résolution d'analyse effective.

        En mode `MODE_MANUEL`, la copie est identique : l'opérateur a dit
        qu'il savait ce qu'il faisait. En mode `MODE_AUTO`,
        `taille_entree_automatique` tranche à partir de la source.

        Cette méthode ne mute PAS l'original. La résolution d'une vidéo est
        connue au chargement, pas à la construction du réglage : un `Config`
        partagé entre la fenêtre, le panneau et le moteur ne doit pas changer
        de valeur sous les pieds de celui qui le lit. L'appelant remplace sa
        référence par le résultat.

        Résolution inconnue (0, 0) : le repli de
        `taille_entree_automatique` s'applique, et le journal le dit.
        """
        if mode_resolution(self.resolution_analyse) == MODE_MANUEL:
            return self
        return replace(
            self, taille_entree=taille_entree_automatique(largeur, hauteur)
        )

    @classmethod
    def depuis_fichier(cls, chemin: str | pathlib.Path) -> "Config":
        return cls.depuis_dict(json.loads(pathlib.Path(chemin).read_text(encoding="utf-8")))

    def vers_fichier(self, chemin: str | pathlib.Path) -> None:
        p = pathlib.Path(chemin)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.vers_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def defauts(cls) -> "Config":
        """Valeurs par défaut issues du fichier versionné."""
        p = chemin_defaut_config()
        if p.exists():
            return cls.depuis_fichier(p)
        # Repli sur les valeurs du dataclass (cas PyInstaller sans config/).
        # La contrainte « toute valeur par défaut doit lire config/default.json »
        # n'est plus respectée : on le signale plutôt que de démarrer en silence.
        log.warning(
            "config/default.json introuvable (%s) : repli sur les valeurs "
            "par défaut codées en dur, qui peuvent diverger du fichier.",
            p,
        )
        return cls()
