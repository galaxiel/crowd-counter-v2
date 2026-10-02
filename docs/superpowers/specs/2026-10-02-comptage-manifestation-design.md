# Compteur de manifestation — Design

**Date** : 2026-10-02
**Statut** : validé en discussion, en attente de revue du spec
**Auteur** : Jeff + Hermes

---

## 1. Objectif

Compter le nombre de personnes franchissant une ligne virtuelle, à partir d'un
flux vidéo en provenance d'une **caméra fixe en position surélevée** filmant
une rue. Aucun déplacement de la caméra. Ordre de grandeur attendu : **200 à
300 personnes** au total sur une manifestation, qui défilent de façon espacée
(non compacte).

### Ce qui est explicitement hors périmètre (v1)

- Aucune identification de personne, aucun visages, aucune reconnaissance.
- Aucun mode multi-caméra.
- Aucune application web / serveur / API.
- Aucune densimétrie (hors périmètre : foulues compactes > 5 000 personnes).
- Aucun entraînement de modèle.

## 2. Cas d'usage réel

L'utilisateur cadre lui-même la ligne sur l'image (2 clics), choisit le sens
retenu, lance l'analyse, et **regarde le décompte se faire**. Il doit pouvoir
voir en direct chaque personne détectée (boîtes, identifiants, trajectoires)
pour pouvoir juger par lui-même de la fiabilité du résultat.

## 3. Contraintes de la scène — pourquoi cette architecture

- **Occlusion faible** : personnes espacées, pas de foule compacte.
- **Modèle tête possiblement utile** : l'utilisateur a constaté que le modèle
  « personne » généraliste performait mal sur ses banderoles et fusionnait
  plusieurs personnes. Un modèle détectant les **têtes** pourrait éviter ces
  erreurs, mais **ce n'est pas tranché à la conception**.
- **Point de vue stable** : pas de mouvement de caméra, donc pas de
  stabilisation ni de détection de plan nécessaire.
- **TraficPedestrian dense mais non compact** : le comptage par franchissement
  de ligne + tracking est la méthode correcte.

## 4. Architecture

Le principe directeur : **le moteur de comptage est une bibliothèque sans
interface graphique**. L'IHM n'est qu'une enveloppe. Cela rend le moteur
testable sans écran, réutilisable, et permet d'ajouter plus tard une API web
sans réécrire la logique métier.

```
crowd-counter-v2/
├── compteur/                    # cœur métier — AUCUNE dépendance UI
│   ├── detecteur.py             # chargement modèle YOLO + inférence
│   ├── tracker.py               # ByteTrack (via supervision)
│   ├── ligne.py                 # géométrie de ligne : franchissement + sens
│   ├── compteur.py              # ORCHESTRATEUR (fonction principale)
│   ├── rapport.py               # export CSV / JSON / stats
│   └── config.py                # dataclasses, sérialisation JSON
├── interface/
│   ├── app.py                   # fenêtre principale PySide6
│   ├── widgets_video.py         # affichage vidéo + overlay OpenCV
│   ├── panneau_reglages.py      # sliders, spinboxes, sauvegarde
│   └── style.py                 # feuille de style (dark, lisible)
├── tests/
│   ├── test_ligne.py            # géométrie de franchissement
│   ├── test_compteur.py         # frame synthétique → total attendu
│   └── test_performance.py
├── main.py                      # point d'entrée
├── requirements.txt
├── config/
│   └── default.json             # valeurs par défaut
└── docs/
    └── superpowers/specs/
```

### 4.1 Le moteur : contrat public

```python
# compteur/compteur.py
def analyser_video(
    chemin_video: str,
    config: Config,
    callback_frame: Callable[[FrameResult], None] | None = None,
) -> Resultat
```

`FrameResult` contient : l'image annotée, la liste des détections, la liste des
tracks, le compteur courant, le nombre de personnes présentes.

`Resultat` contient : total, événements de comptage, statistiques, config
utilisée, nom du modèle.

L'interface appelle cette fonction en direct. Les tests l'appellent sans
écran. Une future API web l'appellerait sur un fichier vidéo.

## 5. Algorithme de comptage

Pour chaque frame :

1. **Détection** — YOLO retourne `(x, y, w, h, score, classe_id)`.
2. **Tracking** — ByteTrack associe les détections aux tracks actifs, tous
   ports d'entrée/sortie. Chaque association confirmée a un identifiant unique.
3. **Éligibilité au comptage** — un track n'est compté que si les trois
   conditions sont simultanément remplies :
   - **a.** le track existe depuis au moins `frames_confirmation` frames ;
   - **b.** son point central a traversé la **bande** de la ligne dans le **sens
     sélectionné** ;
   - **c.** il n'a pas déjà été compté.
4. **Comptage** — incrément du total, enregistrement d'un événement
   (frame, timestamp, position, track_id), et un `Event` est émis pour
   l'interface (flash de la ligne, mise à jour du compteur).

### 5.1 Garde-fous contre les faux comptages

- **Sens de traversée obligatoire** — pas seulement « a franchi », mais
  « a franchi dans le sens choisi par l'utilisateur ».
- **Bande de franchissement épaisse** (paramétrable, 5–100 px) — une personne
  qui fait un aller-retour sur la ligne ne compte pas.
- **Hystérésis** — le point central doit rester du côté « départ » pendant
  `frames_hysteresis` frames avant qu'une traversée soit considérée comme
  réelle. Supprime les tremblements du détecteur.

## 6. Réglages (sensibilité)

Trois groupes, modifiables en direct. Chaque paramètre est persisté par profil.

| Groupe            | Paramètre                 | Plage      | Rôle                                            |
|-------------------|---------------------------|------------|-------------------------------------------------|
| **Détection**     | seuil de confiance         | 0.05–0.95  | plus bas → détecte plus, plus de faux positifs |
|                   | taille minimale (px)       | 0–300      | ignore les taches trop loin                     |
|                   | classes retenues           | liste      | si le modèle expose "head" et "person"          |
|                   | taille d'entrée            | 320/640/1280 | compromis précision / vitesse                 |
|                   | **modèle**                 | chemin     | n'importe quel `.pt` présent dans le dossier   |
| **Tracker**       | frames de confirmation     | 1–20       | **réglage de sensibilité principal**            |
|                   | survie max (frames)        | 1–120      | mémoire du tracker                              |
|                   | seuil de matching          | 0.1–0.9    | avidité de l'association                        |
| **Ligne**         | points (2 clics)           | —          | définition de la ligne                          |
|                   | épaisseur de bande (px)    | 5–100      | anti-rebond                                     |
|                   | sens                       | 2 flèches  | direction retenue                               |
|                   | frames d'hystérésis        | 0–10       | anti-flou                                       |
| **Lecture**       | vitesse de présentation    | 0.25×–max  | 0.25×, 0.5×, 1×, 2×, 4×, « max »               |

Les réglages qui modifient un track **en cours d'analyse** s'appliquent
immédiatement. Ceux qui affectent la structure (modèle, taille d'entrée) ne
prennent effet qu'au démarrage d'une analyse.

## 7. Interface

Fenêtre unique, disposition en deux colonnes.

- **Colonne gauche** : le flux vidéo avec overlay. Boîtes de détection, point
  central, identifiant de track, trajectoire courte (dernières N positions),
  ligne et bande de franchissement dessinées en surimpression.
- **Colonne droite, en haut** : compteur cumulé en très gros (lisible à
  distance), puis « personnes présentes maintenant », « tracks actifs ».
- **Colonne droite, en bas** : les réglages, groupés en sections.
- **Barre d'actions** : Charger une vidéo → Tracer la ligne → Lancer / Pause /
  Stop → Exporter.

Éthétique sombre, sobre, à forte lisibilité (application en extérieur,
conditions de luminosité variables).

## 8. Sorties

### Écran
Compteur cumulé, présents maintenant, temps réel, nombre de frames traitées,
taux de dropped frames.

### Exports (à l'arrêt ou manuellement)
- `resultat.csv` — une ligne par événement : `frame, timestamp_s, x, y, track_id`
- `rapport.json` — total, modèle utilisé, config complète employée, statistiques
- `apercu/` — quelques frames annotées (optionnel)

## 9. Validation et mesure d'erreur

Le nombre de personnes est une **estimation**, pas une mesure exacte.

**Le logiciel ne prétend pas fournir un pourcentage d'erreur automatique.** Un
tel chiffre exigerait une vérité terrain inconnue.

En revanche, l'IHM expose un **mode audit** permettant à l'utilisateur de :

1. corriger manuellement un comptage (« cette personne n'a pas été comptée »,
   « ce comptage était un faux positif ») ;
2. voir en temps réel **l'écart entre l'estimation automatique et le total
   corrigé** ;
3. obtenir ainsi un **pourcentage d'erreur mesuré** sur la portion vérifiée.

L'interface affiche en permanence un **indicateur de confiance** de la frame
courante (densité de détection, perte de track, durée de vie moyenne des
tracks) pour signaler les moments peu fiables — pas pour corriger le total.

## 10. Stack technique

- **Python 3.11+**
- **PyTorch** avec support CUDA (GPU NVIDIA requis pour un usage confortable)
- **ultralytics** (YOLO) pour la détection
- **supervision** pour ByteTrack (bibliothèque éprouvée, évite un tracker
  maison)
- **PySide6** pour l'interface
- **opencv-python** pour le traitement d'image et l'overlay
- **numpy**, **Pillow** pour l'affichage
- **PyInstaller** pour la génération de l'exécutable `.exe`

## 11. Découpage de l'implémentation (ordre prévu)

1. Socle : lecture vidéo, boucle d'affichage, structure de config + sauvegarde.
2. Détection seule : visualisation des boîtes, réglage du seuil en direct.
3. Ligne de franchissement : tracé à la souris, calcul de traversée, sens.
4. Tracking : intégration ByteTrack, identifiants, trajectoires.
5. Comptage : total, événements, export CSV/JSON.
6. Panneau de réglages complet.
7. Mode audit et correction manuelle.
8. Packaging `.exe`.

Chaque étape est vérifiable visuellement avant de passer à la suivante.

## 12. Points ouverts / à trancher

- **Modèle tête vs modèle personne** : à trancher empiriquement sur les
  vidéos réelles de l'utilisateur, pas par choix théorique. L'architecture
  ne dépend pas de ce choix (le modèle est un paramètre). Le test comparatif
  fait partie de l'étape 2.
