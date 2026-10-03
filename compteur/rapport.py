"""Export des résultats d'une analyse : CSV, JSON, statistiques, fiabilité.

C'est ce qui transforme un compteur en outil utilisable par un média. Sans ce
module, le total affiché n'existe que sur l'écran de la machine qui a lancé
l'analyse.

## Ce que le rapport promet, et ce qu'il refuse de promettre

L'export est tenu à une règle : **ne jamais laisser croire à une précision
qu'il n'a pas.**

Mesures réelles sur la vidéo de référence :

- la détection est fiable — 88 têtes par frame ;
- le tracking aussi — IoU 0,85 ;
- **mais le décompte rend 0.** Le déplacement médian d'une personne entre deux
  frames y vaut 0,00 px, noyé dans un bruit de détecteur de σ = 3 px. Le
  mouvement réel n'est pas séparable du bruit, donc aucun franchissement n'est
  établi.

Un rapport qui publie « 0 personnes » sans dire cela est un mensonge, pas une
mesure. D'où les deux partis pris de ce module :

1. **Aucun taux d'erreur automatique.** On n'a pas la vérité terrain : le
   nombre de personnes réellement passées devant la ligne est inconnu. Un
   « ± 5 % d'erreur » déduit de la config serait lu par un média comme une
   mesure. Il n'y en a pas. Les seuls chiffres publiés sont mesurés.
2. **Des indicateurs de FIABILITÉ explicites.** Le nombre d'identités qui ont
   franchi, les détections moyennes par frame, la cohérence interne des chiffres
   et, le cas échéant, un avertissement en français. Un lecteur qui voit une
   durée de vie de track de 2 frames SAIT que le total n'est pas fiable ; c'est
   exactement l'honnêteté qui manque à un compteur de foule classique.

Quand un indicateur ne peut pas être mesuré à partir du `Resultat` disponible,
la clé vaut ``None`` et un avertissement l'explique. Elle n'est jamais
supprimée, et jamais remplie par une valeur devinée : « non mesuré » est une
information, une clé absente n'en est pas une.

## Répartition des fichiers

- le **CSV** est la liste brute des franchissements (une ligne par événement,
  en-tête fixe) : c'est ce qu'un tableur ouvre ;
- le **JSON** porte tout le reste — contexte, config **complète**, statistiques
  et fiabilité — : c'est ce qui rend l'analyse rejouable et ce qui permet de
    vérifier ce chiffre avant de le publier ;
- `Resultat.config` étant optionnel, l'export fonctionne aussi sans config
  (``"config": null``).
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from statistics import median

from .types import Resultat

#: En-tête du CSV. Volontairement minimal et stable : c'est un contrat avec
#: les tableurs et les scripts qui lisent la sortie.
ENTETE_CSV = "frame,timestamp_s,x,y,track_id"

#: Colonnes d'un événement, dans l'ordre du CSV. Les deux exports doivent
#: porter les mêmes informations : une divergence serait invisible.
COLONNES_EVENEMENT = ("frame", "timestamp_s", "x", "y", "track_id")

#: Fenêtre du pic de débit. Au plus 60 s, et jamais plus que la durée réelle de
#: l'analyse : une vidéo de 10 s ne peut pas produire une minute observée, et
#: une extrapoler supposerait un débit uniforme que rien ne mesure.
FENETRE_DEBIT_S = 60.0

#: En dessous de ce nombre d'images par seconde, la durée enregistrée est
#: suspecte : `Resultat.secondes` vaut soit la durée de la vidéo (interface),
#: soit le temps de calcul écoulé (`analyser_video`). Le second cas donne un
#: débit par minute systématiquement sous-estimé — deux fois moins, si
#: l'analyse a duré deux fois la vidéo.
SEUIL_FPS_PLAUSIBLE = 5.0


# -- Chemins ---------------------------------------------------------------


def chemin_video_nom(chemin_video: str) -> str:
    """Nom de la vidéo sans son extension."""
    return pathlib.Path(chemin_video).stem


def chemins_par_defaut(chemin_video: str, dossier: str | pathlib.Path) -> dict:
    """Les deux chemins d'export déduits du nom de la vidéo et du dossier.

    L'interface appelle cette fonction et affiche ensuite ``chemins['csv'].name``
    à l'utilisateur : les valeurs doivent être des `Path`, pas des chaînes.
    Le dossier n'est pas créé ici — `ecrire_csv` / `ecrire_json` le font, et
    l'utilisateur peut changer d'avis avant d'écrire.
    """
    base = chemin_video_nom(chemin_video)
    d = pathlib.Path(dossier)
    return {
        "csv": d / f"{base}_head_results.csv",
        "json": d / f"{base}_head_results.json",
    }


# -- Écriture --------------------------------------------------------------


def ecrire_csv(resultat: Resultat, chemin: str | pathlib.Path) -> pathlib.Path:
    """Écrit un événement par ligne et rend le chemin écrit.

    Zéro événement produit un fichier réduit à son en-tête : c'est le cas réel
    de la vidéo de référence, pas une erreur. Un fichier à en-tête seul est
    lisible par tous les tableurs et dit exactement la vérité — zéro
    franchissement.
    """
    p = pathlib.Path(chemin)
    p.parent.mkdir(parents=True, exist_ok=True)
    lignes = [ENTETE_CSV]
    lignes.extend(ev.vers_ligne_csv() for ev in resultat.evenements)
    # `newline="\n"` explicite : en mode texte, Python traduit "\n" en CRLF sous
    # Windows. Deux conséquences, et chacune compte — le fichier exporté depuis
    # Windows ne serait pas octet-pour-octet identique à celui exporté depuis
    # Linux (donc pas reproductible d'une machine à l'autre), et un `\r` traînant
    # casse les lecteurs ligne à ligne les plus naïfs.
    p.write_text("\n".join(lignes) + "\n", encoding="utf-8", newline="\n")
    return p


def ecrire_json(resultat: Resultat, chemin: str | pathlib.Path) -> pathlib.Path:
    """Écrit le rapport complet et rend le chemin écrit.

    Le JSON est la source de vérité de l'analyse : il porte la config
    complète, les événements, les statistiques et les indicateurs de
    fiabilité. `ensure_ascii=False` : les messages d'avertissement sont en
    français et doivent rester lisibles.
    """
    p = pathlib.Path(chemin)
    p.parent.mkdir(parents=True, exist_ok=True)
    stats = statistiques(resultat)
    donnees = {
        "total": resultat.total,
        "modele": resultat.modele,
        "nb_frames": resultat.nb_frames,
        "presents_max": resultat.presents_max,
        "presents_moyen": resultat.presents_moyen,
        "secondes": resultat.secondes,
        # Config COMPLÈTE : c'est elle qui permet de rejouer l'analyse. Une
        # config tronquée aux champs « visibles » ne dit pas quel seuil, quelle
        # bande ni quel lissage ont réellement servi.
        "config": resultat.config.vers_dict() if resultat.config else None,
        "evenements": [
            dict(zip(COLONNES_EVENEMENT, (getattr(ev, c) for c in COLONNES_EVENEMENT)))
            for ev in resultat.evenements
        ],
        "statistiques": stats,
        # Répliqué au premier niveau : le lecteur ouvre le JSON et doit trouver
        # les indicateurs de fiabilité sans avoir à fouiller dans
        # « statistiques ». Même objet, pas une seconde mesure.
        "fiabilite": stats["fiabilite"],
    }
    p.write_text(
        json.dumps(donnees, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return p


# -- Statistiques ----------------------------------------------------------


def statistiques(resultat: Resultat) -> dict:
    """Indicateurs de synthèse d'une analyse terminée.

    Aucun chiffre d'erreur ici non plus : on n'a pas la vérité terrain, donc
    le débit est une moyenne arithmétique sur la vidéo analysée, pas une
    prédiction. Les indicateurs de fiabilité sont imbriqués sous ``fiabilite``.

    Les clés ``total``, ``duree_s``, ``personnes_par_minute``, ``presents_max``,
    ``presents_moyen``, ``debit_max_par_minute``, ``modele`` et ``nb_frames``
    constituent le contrat avec `tools/comparer_modeles.py` et l'interface.
    """
    duree = float(resultat.secondes or 0.0)

    if duree > 0.0 and resultat.total:
        debit = resultat.total / duree * 60.0
    else:
        # Vidéo de durée nulle (ou total nul) : pas de débit à annoncer. Le
        # zéro est un « non mesuré », pas une mesure de zéro.
        debit = 0.0

    fenetre = _fenetre_de_reel(resultat)
    pic = _pic_de_debit(resultat.evenements, fenetre)

    return {
        "total": resultat.total,
        "duree_s": round(duree, 2),
        "fps_moyen": round(resultat.nb_frames / duree, 2) if duree > 0.0 else None,
        "personnes_par_minute": round(debit, 1),
        # Le PIC est un compte d'événements sur la fenêtre réellement
        # couverte, pas un débit extrapolé. `debit_max_fenetre_s` dit de
        # combien de secondes ce compte a été observé : en dessous d'une
        # minute, lire ce chiffre comme un « par minute » serait faux.
        "debit_max_par_minute": pic,
        "debit_max_fenetre_s": round(fenetre, 2),
        "presents_max": resultat.presents_max,
        "presents_moyen": round(float(resultat.presents_moyen or 0.0), 2),
        "nb_frames": resultat.nb_frames,
        "modele": resultat.modele,
        "fiabilite": indicateurs_fiabilite(resultat),
    }


def _fenetre_de_reel(resultat: Resultat) -> float:
    """Durée sur laquelle le pic de débit est observable, en secondes.

    ``min(60, durée de l'analyse)`` : au-delà d'une minute, un pic se mesure
    toujours sur une minute ; en dessous, la vidéo entière est la seule
    fenêtre possible et le rapport doit le dire au lieu d'extrapoler.
    """
    return min(FENETRE_DEBIT_S, float(resultat.secondes or 0.0))


def _pic_de_debit(evenements: list, fenetre: float) -> int:
    """Plus grand nombre d'événements dans une fenêtre glissante de `fenetre`.

    Fenêtre glissante et non « première minute de la vidéo » : un pic de
   |POLICE| plus de dix passages en deux secondes doit ressortir, sinon le
    débit maximal ne décrit pas le risque pour un lieu bondé.

    Fenêtre de durée nulle (durée inconnue) : on retient tout, ce qui évite
    une division par zéro sur une vidéo dont la durée n'a pas été relevée.
    """
    if not evenements:
        return 0
    horodatages = sorted(ev.timestamp_s for ev in evenements)
    if fenetre <= 0.0:
        return len(horodatages)
    debut = 0
    pic = 0
    for fin, t in enumerate(horodatages):
        # La fenêtre est [t - fenetre, t] : elle contient l'événement courant
        # et ceux qui le précèdent de moins d'une fenêtre.
        while horodatages[debut] < t - fenetre:
            debut += 1
        pic = max(pic, fin - debut + 1)
    return pic


# -- Fiabilité -------------------------------------------------------------


def indicateurs_fiabilite(resultat: Resultat) -> dict:
    """Ce qui permet à un lecteur de juger le total avant de le publier.

    Les indicateurs se répartissent en deux familles :

    - **ce qui est mesuré** : combien d'identités distinctes ont franchi la
      ligne, combien de détections le détecteur a produites par frame en
      moyenne, si les chiffres du rapport se contredisent ;
    - **ce qui n'est pas mesurable ici** : le nombre total de tracks vus et la
      durée de vie moyenne d'un track. `Resultat` ne transporte que les
      *événements de franchissement*, or le registre anti-recomptage garantit
      au plus un événement par track : un événement ne dit donc rien de la
      durée de vie de l'identité qui l'a produit. Ces clés valent `None` et un
      avertissement l'explique, plutôt que d'être remplies par un nombre
      deviné.

    ``coherent`` ne dit PAS que le comptage est juste : il dit seulement que
    les chiffres du rapport ne se contredisent pas entre eux.
    """
    evenements = list(resultat.evenements)
    track_ids = [ev.track_id for ev in evenements]
    distincts = set(track_ids)
    presents_moyen = float(resultat.presents_moyen or 0.0)
    presents_max = int(resultat.presents_max or 0)
    nb_frames = int(resultat.nb_frames or 0)
    duree = float(resultat.secondes or 0.0)
    total = int(resultat.total or 0)

    avertissements: list[str] = []
    coherent = True

    if nb_frames <= 0:
        coherent = False
        avertissements.append(
            "Aucune frame analysée : tous les chiffres de ce rapport sont vides."
        )

    if total != len(evenements):
        coherent = False
        avertissements.append(
            f"Le total annoncé ({total}) et le nombre de lignes d'événements "
            f"({len(evenements)}) divergent : l'export est incohérent avec "
            "lui-même."
        )

    if total == 0 and presents_moyen > 0.0:
        # Le cas réel de la vidéo de référence. Un « 0 personnes » lu sans
        # cette phrase serait une affirmation fausse : il y avait une foule.
        avertissements.append(
            f"{presents_moyen:.1f} personnes détectées par frame en moyenne, "
            f"0 franchissement compté sur {nb_frames} frames : ce 0 est un "
            "défaut de comptage, pas une foule absente. Le déplacement des "
            "personnes est noyé dans le bruit du détecteur — aucun franchissement "
            "n'est établi. Ce total ne doit pas être publié."
        )

    if total > 0 and presents_moyen <= 0.0:
        coherent = False
        avertissements.append(
            f"{total} franchissements comptés alors qu'aucune personne n'est "
            "présente en moyenne : les deux mesures se contredisent."
        )

    repetes = sorted(tid for tid, n in Counter(track_ids).items() if n > 1)
    if repetes:
        coherent = False
        avertissements.append(
            f"Les track_id {repetes} ont produit plusieurs événements : le "
            "verrou anti-recomptage n'a pas tenu, le total est gonflé."
        )

    if nb_frames > 0 and duree > 0.0:
        fps = nb_frames / duree
        if fps < SEUIL_FPS_PLAUSIBLE:
            coherent = False
            avertissements.append(
                f"{nb_frames} frames en {duree:.0f} s, soit {fps:.1f} images/s : "
                "la durée enregistrée est probablement le temps de calcul et non "
                "la durée de la vidéo. Le débit par minute est sous-estimé d'autant."
            )

    # Ces deux limites sont des propriétés du MODÈLE DE DONNÉES, pas de
    # cette exécution-ci : elles sont donc averties à chaque export, y compris
    # quand le comptage s'est bien passé. Un lecteur qui trouve `null` sans
    # explication croirait à une valeur oubliée.
    avertissements.append(
        "La durée de vie moyenne d'un track n'est pas mesurable depuis "
        "`Resultat` : il ne transporte que les événements de franchissement, "
        "et le verrou anti-recomptage en donne au plus un par track."
    )
    avertissements.append(
        "Le nombre total de tracks vus n'est pas mesurable depuis `Resultat` : "
        "seules les identités ayant franchi la ligne y figurent."
    )

    if nb_frames > 0 and total > 0:
        fenetre = _fenetre_de_reel(resultat)
        if fenetre < FENETRE_DEBIT_S:
            avertissements.append(
                f"Le débit maximal ({_pic_de_debit(evenements, fenetre)}) est un "
                f"compte sur {fenetre:.0f} s, pas sur une minute : c'est un "
                "minimum observé, pas un débit extrapolé."
            )

    return {
        # -- mesuré --
        "nb_tracks_comptes": len(distincts),
        "evenements_par_track": (
            round(len(evenements) / len(distincts), 2) if distincts else None
        ),
        "detections_par_frame_moyen": round(presents_moyen, 2),
        "detections_par_frame_max": presents_max,
        "frames_par_evenement": (
            round(nb_frames / total, 2) if nb_frames > 0 and total > 0 else None
        ),
        "intervalle_median_evenements_s": _intervalle_median(evenements),
        # -- non mesurable depuis Resultat --
        "nb_tracks_vus": None,
        "duree_vie_moyenne_track_frames": None,
        # -- lecture --
        "coherent": coherent,
        "avertissements": avertissements,
    }


def _intervalle_median(evenements: list) -> float | None:
    """Écart médian entre deux franchissements, en secondes.

    Vrai indicateur de terrain — « une personne toutes les 2,5 s au pic » — et
    lui aussi directement mesuré. `None` sous deux événements : une médiane
    sur un seul intervalle se lirait comme une régularité qui n'existe pas.
    """
    if len(evenements) < 2:
        return None
    horodatages = sorted(ev.timestamp_s for ev in evenements)
    ecarts = [b - a for a, b in zip(horodatages, horodatages[1:])]
    return round(median(ecarts), 2)