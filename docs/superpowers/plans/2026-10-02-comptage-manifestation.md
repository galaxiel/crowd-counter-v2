# Compteur de manifestation — Plan d'implémentation

> **Pour les agentsworkers :** SOUS-SKILL REQUISE : utiliser
> `superpowers:subagent-driven-development` (recommandé) ou
> `superpowers:executing-plans` pour implémenter ce plan tâche par tâche.
> Les étapes utilisent la syntaxe à cases (`- [ ]`) pour le suivi.

**Objectif :** construire un compteur de personnes franchissant une ligne
virtuelle, à partir d'une vidéo de caméra fixe en surplomb, avec un moteur de
comptage sans UI et une interface PySide6.

**Architecture :** le cœur métier (`compteur/`) est une bibliothèque sans
dépendance graphique, dont la fonction principale
`analyser_video(chemin, config, callback) -> Resultat` est testable sans
écran. L'interface (`interface/`) est une enveloppe PySide6 qui appelle ce
moteur. Chaque brique (détection, tracking, ligne, comptage) est un module
séparé avec un contrat explicite, ce qui rend chaque étape vérifiable
indépendamment.

**Stack technique :** Python 3.11+, PyTorch (CUDA), ultralytics (YOLO),
supervision (ByteTrack), PySide6, opencv-python, numpy, Pillow, pytest.

**Spec :** `docs/superpowers/specs/2026-10-02-comptage-manifestation-design.md`

## Contraintes globales

- Python 3.11 ou supérieur. Aucune syntaxe antérieure.
- Windows 11, GPU NVIDIA requis pour un usage confortable (le CPU doit rester
  fonctionnel mais lent).
- Le paquet `compteur/` ne doit **jamais** importer PySide6, Tkinter ou tout
  autre module d'interface. Test automatisé à l'appui (Task 1).
- Aucun entraînement de modèle, aucun téléchargement de dataset.
- Le modèle YOLO est un paramètre de configuration, jamais codé en dur. Toute
  valeur par défaut doit lire `config/default.json`.
- Tests avec pytest. Chaque module a ses tests. TDD sur chaque tâche.
- Commits fréquents : un commit par étape franchie, message au format
  conventionnel.
- Interface en français (libellés, messages), code et identifiants en anglais.
- Chaque tâche se termine par un test qui passe et un commit.

---

## Structure des fichiers

```
crowd-counter-v2/
├── compteur/
│   ├── __init__.py          # exports publics
│   ├── types.py             # Detection, Track, FrameResult, Resultat, Evenement
│   ├── config.py            # dataclasses Config + (de)sérialisation JSON
│   ├── ligne.py             # géométrie : bande, sens, franchissement, hystérésis
│   ├── detecteur.py         # chargement YOLO + inférence
│   ├── tracker.py           # wrapper ByteTrack
│   ├── compteur.py          # analyse_video() — orchestrateur
│   └── rapport.py           # export CSV/JSON + statistiques
├── interface/
│   ├── __init__.py
│   ├── style.py             # feuille de style Qt sombre
│   ├── overlay.py           # dessin des boîtes, trajectoires, ligne (OpenCV)
│   ├── widgets_video.py     # widget d'affichage vidéo + clics
│   ├── panneau_reglages.py  # sliders et sauvegarde des profils
│   └── app.py               # fenêtre principale
├── tests/
│   ├── conftest.py
│   ├── test_isolation_ui.py     # Task 1
│   ├── test_types.py
│   ├── test_config.py
│   ├── test_ligne.py
│   ├── test_detecteur.py
│   ├── test_tracker.py
│   ├── test_compteur.py
│   ├── test_rapport.py
│   ├── test_interface_isolation.py
│   └── test_export_video_synthetique.py
├── config/
│   └── default.json
├── main.py
├── requirements.txt
└── docs/
```

### Responsabilités d'un fichier

| Fichier | Une seule responsabilité |
|---|---|
| `compteur/types.py` | Définir les structures de données qui circulent entre modules |
| `compteur/config.py` | Charger/sauvegarder les réglages ; ne pas savoir quoi qu'ils signifient |
| `compteur/ligne.py` | Répondre à « ce point est-il passé de l'autre côté, dans le bon sens ? » |
| `compteur/detecteur.py` | Transformer une image numpy en liste de `Detection` |
| `compteur/tracker.py` | Transformer des détections successives en tracks suivis |
| `compteur/compteur.py` | Orchestrer le pipeline et décider des comptages |
| `compteur/rapport.py` | Écrire des fichiers et calculer des statistiques |
| `interface/overlay.py` | Dessiner sur une image, ne rien décider |
| `interface/widgets_video.py` | Afficher des images, émettre des clics |
| `interface/panneau_reglages.py` | Produire une `Config` depuis des widgets |
| `interface/app.py` | Coller le tout, gérer la boucle de lecture |

---

## Tâche 1 : Socle — types, config, et garantie d'isolation UI

**Fichiers :**
- Créer : `compteur/__init__.py`
- Créer : `compteur/types.py`
- Créer : `compteur/config.py`
- Créer : `config/default.json`
- Créer : `requirements.txt`
- Créer : `tests/test_isolation_ui.py`
- Créer : `tests/test_config.py`
- Créer : `tests/test_types.py`

**Interfaces :**
- Consomme : rien (point de départ).
- Produit :
  - `compteur.types.Detection(x1: float, y1: float, x2: float, y2: float, score: float, class_id: int)`
    avec propriété `center` -> `(float, float)`.
  - `compteur.types.Track(track_id: int, center: tuple[float, float], bbox: tuple[float, float, float, float], age: int, confirmed: bool)`
  - `compteur.types.Evenement(frame: int, timestamp_s: float, x: float, y: float, track_id: int)`
  - `compteur.types.FrameResult(image: np.ndarray, detections: list[Detection], tracks: list[Track], total: int, presents: int, frame_index: int, timestamp_s: float)`
  - `compteur.types.Resultat(total: int, evenements: list[Evenement], config: Config, modele: str, nb_frames: int, presents_max: int)`
  - `compteur.config.Config` (dataclass) avec champs : `modele: str`, `seuil_confiance: float = 0.25`, `taille_min_px: int = 20`, `classes_retenues: list[int] | None = None`, `taille_entree: int = 640`, `frames_confirmation: int = 3`, `survie_max: int = 30`, `seuil_matching: float = 0.5`, `ligne: tuple[float, float, float, float] | None = None`, `epaisseur_bande: int = 30`, `sens: int = 1`, `frames_hysteresis: int = 2`.
    Méthodes : `vers_dict() -> dict`, `depuis_dict(d: dict) -> Config`,
    `depuis_fichier(chemin) -> Config`, `vers_fichier(chemin) -> None`,
    `defauts() -> Config` (lit `config/default.json`).
  - `compteur.config.chemin_defaut_config() -> pathlib.Path`

- [ ] **Étape 1 : Écrire le test d'isolation UI (doit échouer)**

Créer `tests/test_isolation_ui.py` :

```python
"""Le paquet compteur/ ne doit jamais dépendre d'un toolkit graphique."""
import ast
import pathlib

import pytest

LIVRABLES = list(pathlib.Path("compteur").rglob("*.py"))
MODULES_INTERDITS = {"PySide6", "PySide6.QtCore", "tkinter", "PyQt5", "PyQt6", "wx"}


def test_le_paquet_compteur_existe():
    assert (pathlib.Path("compteur") / "__init__.py").exists(), (
        "le paquet compteur/ doit exister dès la tâche 1"
    )


@pytest.mark.parametrize("chemin", LIVRABLES, ids=lambda p: p.name)
def test_aucun_import_graphique_dans_compteur(chemin):
    arbre = ast.parse(chemin.read_text(encoding="utf-8"))
    importes = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            importes.update(a.nom for a in noeud.names)
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            importes.add(noeud.module)
    interdites = importes & MODULES_INTERDITS
    assert not interdites, f"{chemin} importe {interdites}, interdit dans compteur/"
```

- [ ] **Étape 2 : Lancer le test, vérifier qu'il échoue**

```bash
python -m pytest tests/test_isolation_ui.py -v
```

Attendu : ÉCHEC, `FileNotFoundError` sur `compteur/__init__.py`.

- [ ] **Étape 3 : Créer le paquet et les types**

Créer `compteur/__init__.py` :

```python
"""Moteur de comptage — aucune dépendance graphique.

Ce paquet est volontairement utilisable sans écran : les tests, la comparaison
de modèles et une future API web l'importent tous directement.
"""

__all__ = ["types", "config", "ligne", "detecteur", "tracker", "compteur", "rapport"]
```

Créer `compteur/types.py` :

```python
"""Structures de données échangées entre les briques du moteur."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # évite une dépendance circulaire à l'exécution
    from .config import Config


@dataclass
class Detection:
    """Boîte produite par le détecteur, en coordonnées pixels de l'image source."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aire(self) -> float:
        return self.width * self.height


@dataclass
class Track:
    """Identité suivie dans le temps, telle que-maintenue par le tracker."""

    track_id: int
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    age: int
    confirmed: bool


@dataclass
class Evenement:
    """Un comptage effectif, daté et localisé."""

    frame: int
    timestamp_s: float
    x: float
    y: float
    track_id: int

    def vers_ligne_csv(self) -> str:
        return f"{self.frame},{self.timestamp_s:.3f},{self.x:.1f},{self.y:.1f},{self.track_id}"


@dataclass
class FrameResult:
    """Tout ce que l'interface peut afficher pour une frame donnée."""

    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    total: int = 0
    presents: int = 0
    frame_index: int = 0
    timestamp_s: float = 0.0
    indice: int = 0
    evenements: list[Evenement] = field(default_factory=list)


@dataclass
class Resultat:
    """Bilan complet d'une analyse."""

    total: int
    evenements: list[Evenement] = field(default_factory=list)
    config: "Config | None" = None
    modele: str = ""
    nb_frames: int = 0
    presents_max: int = 0
    presents_moyen: float = 0.0
    secondes: float = 0.0
```

- [ ] **Étape 4 : Écrire les tests de types, les faire échouer**

Créer `tests/test_types.py` :

```python
import pytest

from compteur.types import Detection, Evenement


def test_center_est_le_milieu_de_la_boite():
    d = Detection(x1=10, y1=20, x2=30, y2=60, score=0.9, class_id=0)
    assert d.center == (20.0, 40.0)


def test_aire_calculee():
    d = Detection(x1=0, y1=0, x2=10, y2=5, score=0.5, class_id=0)
    assert d.aire == pytest.approx(50.0)


def test_evenement_vers_ligne_csv():
    e = Evenement(frame=42, timestamp_s=1.4, x=100.0, y=200.0, track_id=7)
    assert e.vers_ligne_csv() == "42,1.400,100.0,200.0,7"
```

- [ ] **Étape 5 : Écrire les tests de config, les faire échouer**

Créer `tests/test_config.py` :

```python
import json

from compteur.config import Config, chemin_defaut_config


def test_aller_retour_dict():
    c = Config(modele="yolov8n-head.pt", seuil_confiance=0.4, frames_confirmation=5)
    retour = Config.depuis_dict(c.vers_dict())
    assert retour == c


def test_aller_retour_fichier(tmp_path):
    c = Config(modele="x.pt", epaisseur_bande=55, sens=-1)
    p = tmp_path / "c.json"
    c.vers_fichier(p)
    assert Config.depuis_fichier(p) == c


def test_ligne_none_est_acceptee():
    c = Config(modele="x.pt", ligne=None)
    assert c.depuis_dict(c.vers_dict()).ligne is None


def test_config_par_defaut_existe():
    assert chemin_defaut_config().exists(), "config/default.json doit être versionné"


def test_config_par_defaut_est_valide():
    d = json.loads(chemin_defaut_config().read_text(encoding="utf-8"))
    Config.depuis_dict(d)  # lève si une clé est inconnue ou mal typée
```

- [ ] **Étape 6 : Lancer les tests, vérifier qu'ils échouent**

```bash
python -m pytest tests/test_config.py tests/test_types.py -v
```

Attendu : ÉCHEC sur l'import (`ModuleNotFoundError: No module named 'compteur.config'`).

- [ ] **Étape 7 : Implémenter `compteur/config.py`**

```python
"""Réglages du compteur : dataclasses, sérialisation JSON, valeurs par défaut."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field, fields

RACINE = pathlib.Path(__file__).resolve().parent.parent


def chemin_defaut_config() -> pathlib.Path:
    """Chemin du fichier de valeurs par défaut, versionné avec le projet."""
    return RACINE / "config" / "default.json"


@dataclass
class Config:
    # Détection
    modele: str = "yolov8n-head.pt"
    seuil_confiance: float = 0.25
    taille_min_px: int = 20
    classes_retenues: list[int] | None = None
    taille_entree: int = 640
    # Tracker
    frames_confirmation: int = 3
    survie_max: int = 30
    seuil_matching: float = 0.5
    # Ligne
    ligne: tuple[float, float, float, float] | None = None
    epaisseur_bande: int = 30
    sens: int = 1
    frames_hysteresis: int = 2

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
        return cls(**d)

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
        return cls()
```

- [ ] **Étape 8 : Créer `config/default.json` et `requirements.txt`**

`config/default.json` :

```json
{
  "modele": "yolov8n-head.pt",
  "seuil_confiance": 0.25,
  "taille_min_px": 20,
  "classes_retenues": null,
  "taille_entree": 640,
  "frames_confirmation": 3,
  "survie_max": 30,
  "seuil_matching": 0.5,
  "ligne": null,
  "epaisseur_bande": 30,
  "sens": 1,
  "frames_hysteresis": 2
}
```

`requirements.txt` :

```
torch>=2.4
ultralytics>=8.3
supervision>=0.6
opencv-python>=4.10
numpy>=1.26
Pillow>=10.4
PySide6>=6.7
scipy>=1.13
pytest>=8.0
pyinstaller>=6.10
```

- [ ] **Étape 9 : Lancer tous les tests, vérifier qu'ils passent**

```bash
python -m pytest tests/ -v
```

Attendu : TOUT PASSE (isolation, types, config).

- [ ] **Étape 10 : Commit**

```bash
git add compteur/ config/ tests/ requirements.txt
git commit -m "feat: socle du moteur — types, config, isolation UI garantie"
```

---

## Tâche 2 : Géométrie de la ligne de franchissement

**Fichiers :**
- Créer : `compteur/ligne.py`
- Créer : `tests/test_ligne.py`

**Interfaces :**
- Consomme : `compteur.types.Track` (Tâche 1).
- Produit :
  - `compteur.ligne.Ligne(p1: tuple[float, float], p2: tuple[float, float], epaisseur: int = 30, sens: int = 1, hysteresis: int = 2)`
    Attributs exposés : `.p1`, `.p2`, `.epaisseur`, `.sens`, `.hysteresis`.
  - `.point_du_cote(p: tuple[float, float]) -> int` — retourne `+1` si `p` est du
    côté « avant », `-1` du côté « après », `0` si `p` est dans la bande.
  - `.vecteur_normal() -> tuple[float, float]` — normal unitaire, orientée selon
    `sens` : positive dans la direction de traversée retenue.
  - `.coordonnee_projetee(p: tuple[float, float]) -> float` — position signée
    de `p` sur la normale, en pixels.
  - `.est_dans_la_bande(p: tuple[float, float]) -> bool` — `True` si la
    projection est à moins de `epaisseur / 2` de la ligne.
  - `.contient(p: tuple[float, float]) -> bool` — `True` si `p` est dans le
    rectangle défini par `p1`/`p2` étendu de `epaisseur / 2` (évite de compter
    un croisement à l'infini, loin du segment).
  - `.dessiner(img: np.ndarray) -> np.ndarray` — trace la ligne, la bande et
    une flèche de sens.

**Géométrie retenue (à respecter exactement).** La ligne passe par `p1` et
`p2`. On définit :
- `d = normalize(p2 - p1)` — direction le long de la ligne.
- `n = (-d.y, d.x)` — normale, perpendiculaire.
- `coordonnee_projetee(p) = dot(p - p1, n)` — signée, 0 exactement sur la ligne.
- Le côté « avant » (celui d'où l'on vient) correspond à `coordonnee < 0`
  quand `sens = +1`, et `coordonnee > 0` quand `sens = -1`.
- Le passage est retenu quand la coordonnée projeteé change de signe dans le
  sens indiqué **et** que le déplacement total sur la normale dépasse
  `epaisseur / 2` (on exige qu'on traverse toute la bande, pas qu'on effleure
  la ligne).

- [ ] **Étape 1 : Écrire les tests, tous doivent échouer**

Créer `tests/test_ligne.py` :

```python
import numpy as np
import pytest

from compteur.ligne import Ligne


def ligne_haut_vers_bas(**kw):
    """Ligne verticale x=100, de y=0 à y=1000. Traversée dans le sens +y."""
    return Ligne(p1=(100.0, 0.0), p2=(100.0, 1000.0), **kw)


def test_coordonnee_projetee_nulle_sur_la_ligne():
    l = ligne_haut_vers_bas()
    assert l.coordonnee_projetee((100.0, 500.0)) == pytest.approx(0.0, abs=1e-6)


def test_coordonnee_signee_change_de_cote():
    l = ligne_haut_vers_bas()
    a = l.coordonnee_projetee((50.0, 500.0))
    b = l.coordonnee_projetee((150.0, 500.0))
    assert a * b < 0, "les deux points sont de côtés opposés"


def test_point_du_cote_avant_et_apres():
    l = ligne_haut_vers_bas(sens=1)
    assert l.point_du_cote((50.0, 500.0)) == 1
    assert l.point_du_cote((150.0, 500.0)) == -1


def test_inverser_sens_inverse_les_cotes():
    l = ligne_haut_vers_bas(sens=-1)
    assert l.point_du_cote((50.0, 500.0)) == -1
    assert l.point_du_cote((150.0, 500.0)) == 1


def test_point_dans_la_bande():
    l = ligne_haut_vers_bas(epaisseur=30)
    assert l.est_dans_la_bande((110.0, 500.0)) is True
    assert l.est_dans_la_bande((120.0, 500.0)) is False


def test_point_hors_segment_malgre_proche_de_la_ligne():
    """Une personne à 1 px de la ligne mais à 5000 px du segment n'est pas dessus."""
    l = ligne_haut_vers_bas(epaisseur=30)
    assert l.contient((105.0, 5000.0)) is False
    assert l.contient((105.0, 500.0)) is True


def test_ligne_degenerate_refusee():
    with pytest.raises(ValueError):
        Ligne(p1=(10.0, 10.0), p2=(10.0, 10.0))


def test_vecteur_normal_unitaire():
    l = ligne_haut_vers_bas()
    nx, ny = l.vecteur_normal()
    assert nx * nx + ny * ny == pytest.approx(1.0, abs=1e-6)


def test_sens_invalide_refuse():
    with pytest.raises(ValueError):
        ligne_haut_vers_bas(sens=0)


def test_dessiner_ne_crash_pas_et_ne_mute_pas():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    l = ligne_haut_vers_bas()
    avant = img.copy()
    sortie = l.dessiner(img)
    assert sortie.shape == img.shape
    assert np.array_equal(img, avant), "dessiner() ne doit pas modifier l'image source"
    assert sortie.any(), "la ligne doit être visible"
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_ligne.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'compteur.ligne'`.

- [ ] **Étape 3 : Implémenter `compteur/ligne.py`**

```python
"""Géométrie de la ligne de franchissement.

Tout le raisonnement « cette personne est-elle passée de l'autre côté, dans le
bon sens ? » tient dans ce fichier. Il ne dépend d'aucun autre module métier,
ce qui le rend testable de façon exhaustive.
"""

from __future__ import annotations

import math

import numpy as np


class Ligne:
    """Ligne orientée defining une zone de franchissement.

    La ligne passe par ``p1`` et ``p2``. ``sens`` vaut ``+1`` : on ne compte
    que les passages dans le sens de la normale, ``-1`` : que les passages
    inverses.
    """

    def __init__(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        epaisseur: int = 30,
        sens: int = 1,
        hysteresis: int = 2,
    ) -> None:
        p1 = (float(p1[0]), float(p1[1]))
        p2 = (float(p2[0]), float(p2[1]))
        longueur = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if longueur < 1e-6:
            raise ValueError("les deux points de la ligne sont confondus")
        if sens not in (1, -1):
            raise ValueError("sens doit valoir +1 ou -1")
        if epaisseur < 1:
            raise ValueError("l'épaisseur doit être >= 1")
        if hysteresis < 0:
            raise ValueError("l'hystérésis doit être >= 0")

        self.p1 = p1
        self.p2 = p2
        self.epaisseur = int(epaisseur)
        self.sens = int(sens)
        self.hysteresis = int(hysteresis)

        # Direction le long de la ligne, puis normale (perpendiculaire).
        self._d = ((p2[0] - p1[0]) / longueur, (p2[1] - p1[1]) / longueur)
        self._n = (-self._d[1], self._d[0])
        # La normale est orientée dans le sens de traversée retenu.
        if self.sens < 0:
            self._n = (-self._n[0], -self._n[1])

    # -- Géométrie -------------------------------------------------------

    def vecteur_normal(self) -> tuple[float, float]:
        """Normale unitaire, orientée dans le sens de traversée retenu."""
        return self._n

    def coordonnee_projetee(self, p: tuple[float, float]) -> float:
        """Position signée de ``p`` sur la normale, en pixels (0 = sur la ligne)."""
        dx = float(p[0]) - self.p1[0]
        dy = float(p[1]) - self.p1[1]
        return dx * self._n[0] + dy * self._n[1]

    def est_dans_la_bande(self, p: tuple[float, float]) -> bool:
        return abs(self.coordonnee_projetee(p)) <= self.epaisseur / 2.0

    def parametre_along(self, p: tuple[float, float]) -> float:
        """Position de ``p`` projetée sur l'axe de la ligne (0 à 1 environ)."""
        dx = float(p[0]) - self.p1[0]
        dy = float(p[1]) - self.p1[1]
        return dx * self._d[0] + dy * self._d[1]

    def contient(self, p: tuple[float, float]) -> bool:
        """``p`` est-il dans la bande ET le long du segment (pas à l'infini) ?"""
        if not self.est_dans_la_bande(p):
            return False
        t = self.parametre_along(p)
        longueur = math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
        return -self.epaisseur / 2.0 <= t <= longueur + self.epaisseur / 2.0

    def point_du_cote(self, p: tuple[float, float]) -> int:
        """``+1`` côté d'arrivée, ``-1`` côté de départ, ``0`` dans la bande."""
        c = self.coordonnee_projetee(p)
        if c > self.epaisseur / 2.0:
            return 1
        if c < -self.epaisseur / 2.0:
            return -1
        return 0

    def a_traverse(self, avant: tuple[float, float], apres: tuple[float, float]) -> bool:
        """Le passage de ``avant`` à ``apres`` est-il un franchissement retenu ?

        Trois conditions cumulatives : les deux points sont dans la zone de la
        ligne, le côté a changé, et le déplacement sur la normale dépasse la
        demi-bande (on exige de traverser toute la bande, pas d'effleurer).
        """
        if not self.contient(avant) and not self.contient(apres):
            return False
        c_avant = self.coordonnee_projetee(avant)
        c_apres = self.coordonnee_projetee(apres)
        if c_avant * c_apres > 0:
            return False  # pas de changement de côté
        if abs(c_apres - c_avant) < self.epaisseur:
            return False  # simple tremblement, pas une traversée
        return True

    # -- Rendu -----------------------------------------------------------

    def dessiner(self, img: np.ndarray) -> np.ndarray:
        """Trace ligne, bande et flèche de sens. Ne mute pas ``img``."""
        sortie = img.copy()
        p1 = (int(self.p1[0]), int(self.p1[1]))
        p2 = (int(self.p2[0]), int(self.p2[1]))
        vert = (80, 220, 80)
        jaune = (60, 200, 255)

        # La bande est un quadrilatère épaissi perpendiculairement.
        nx, ny = self._n
        demi = self.epaisseur / 2.0
        quad = np.array(
            [
                [int(self.p1[0] - nx * demi), int(self.p1[1] - ny * demi)],
                [int(self.p2[0] - nx * demi), int(self.p2[1] - ny * demi)],
                [int(self.p2[0] + nx * demi), int(self.p2[1] + ny * demi)],
                [int(self.p1[0] + nx * demi), int(self.p1[1] + ny * demi)],
            ],
            dtype=np.int32,
        )
        cv2 = _cv2()
        cv2.fillPoly(sortie, [quad], (40, 40, 40))
        cv2.polylines(sortie, [quad], True, jaune, 1, cv2.LINE_AA)
        cv2.line(sortie, p1, p2, vert, 2, cv2.LINE_AA)

        # Flèche de sens, au milieu du segment.
        mx = (self.p1[0] + self.p2[0]) / 2.0
        my = (self.p1[1] + self.p2[1]) / 2.0
        pointe = (int(mx + nx * demi * 1.6), int(my + ny * demi * 1.6))
        base = (int(mx), int(my))
        cv2.arrowedLine(sortie, base, pointe, vert, 3, cv2.LINE_AA, tipLength=0.4)
        return sortie


def _cv2():
    import cv2

    return cv2
```

- [ ] **Étape 4 : Lancer les tests, vérifier le passage**

```bash
python -m pytest tests/test_ligne.py -v
```

Attendu : TOUT PASSE (11 tests).

- [ ] **Étape 5 : Commit**

```bash
git add compteur/ligne.py tests/test_ligne.py
git commit -m "feat: géométrie de la ligne — bande, sens, franchissement"
```

---

## Tâche 3 : Détection

**Fichiers :**
- Créer : `compteur/detecteur.py`
- Créer : `tests/test_detecteur.py`

**Interfaces :**
- Consomme : `compteur.types.Detection`, `compteur.config.Config` (Tâche 1).
- Produit :
  - `compteur.detecteur.Detecteur(modele: YOLO, config: Config)`
  - `.detecter(img: np.ndarray) -> list[Detection]` — filtre par
    `seuil_confiance`, `taille_min_px` (côté hauteur ET largeur) et
    `classes_retenues`.
  - `.changer_taille_entree(taille: int) -> None`
  - `.noms_classes() -> dict[int, str]`
  - `compteur.detecteur.charger_modele(chemin: str) -> YOLO` — lève
    `FileNotFoundError` avec un message clair si absent, vérifie une seule fois
    le CUDA et journalise le périphérique retenu.

- [ ] **Étape 1 : Écrire les tests avec un faux détecteur (doivent échouer)**

Créer `tests/test_detecteur.py` :

```python
import numpy as np
import pytest

from compteur.config import Config
from compteur.detecteur import Detecteur
from compteur.types import Detection


class FauxYOLO:
    """Substitue YOLO : renvoie toujours la même liste de détections brutes."""

    def __init__(self, brut: list[tuple]):
        self.brut = brut
        self.taille = 640
        self.appele = 0

    def __call__(self, img, **kwargs):
        self.appele += 1
        return self

    @property
    def results(self):
        return [self]

    def boxes(self):
        return self

    def xyxy(self):
        return [b[:4] for b in self.brut]

    def conf(self):
        return [b[4] for b in self.brut]

    def cls(self):
        return [b[5] for b in self.brut]

    def names(self):
        return {0: "head", 1: "person"}


def image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_filtre_par_seuil_de_confiance():
    brut = [(0, 0, 100, 100, 0.9, 0), (200, 200, 300, 300, 0.1, 0)]
    d = Detecteur(FauxYOLO(brut), Config(seuil_confiance=0.5))
    assert len(d.detecter(image())) == 1


def test_filtre_par_taille_minimale():
    petit = (0, 0, 5, 5, 0.9, 0)
    grand = (0, 0, 100, 100, 0.9, 0)
    d = Detecteur(FauxYOLO([petit, grand]), Config(taille_min_px=30))
    assert len(d.detecter(image())) == 1


def test_filtre_par_classe():
    brut = [(0, 0, 100, 100, 0.9, 0), (200, 200, 300, 300, 0.9, 1)]
    d = Detecteur(FauxYOLO(brut), Config(classes_retenues=[1]))
    assert len(d.detecter(image())) == 1


def test_coordonnes_bien_lues():
    d = Detecteur(FauxYOLO([(10, 20, 110, 220, 0.9, 0)]), Config())
    det = d.detecter(image())[0]
    assert (det.x1, det.y1, det.x2, det.y2) == (10, 20, 110, 220)
    assert det.score == pytest.approx(0.9)
    assert det.class_id == 0


def test_vide_si_aucune_detection():
    d = Detecteur(FauxYOLO([]), Config())
    assert d.detecter(image()) == []


def test_noms_classes_exposes():
    d = Detecteur(FauxYOLO([]), Config())
    assert d.noms_classes() == {0: "head", 1: "person"}


def test_taille_entree_modifiable():
    f = FauxYOLO([])
    Detecteur(f, Config()).changer_taille_entree(1280)
    assert f.taille == 1280
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_detecteur.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'compteur.detecteur'`.

- [ ] **Étape 3 : Implémenter `compteur/detecteur.py`**

```python
"""Détection : image numpy -> liste de boîtes, via YOLO (ultralytics)."""

from __future__ import annotations

import logging

import numpy as np

from .config import Config
from .types import Detection

log = logging.getLogger(__name__)


class Detecteur:
    """Enveloppe mince autour d'un modèle ultralytics.

    Le modèle est injecté pour que les tests n'aient pas besoin de GPU ni de
    poids. En production, utiliser :func:`charger_modele`.
    """

    def __init__(self, modele, config: Config) -> None:
        self.modele = modele
        self.config = config

    def detecter(self, img: np.ndarray) -> list[Detection]:
        resultat = self.modele(
            img,
            conf=self.config.seuil_confiance,
            imgsz=self.config.taille_entree,
            verbose=False,
        )[0]
        boites = resultat.boxes
        if boites is None or len(boites) == 0:
            return []

        xyxy = boites.xyxy().tolist()
        confs = boites.conf().tolist()
        classes = boites.cls().tolist()

        detections: list[Detection] = []
        for (x1, y1, x2, y2), score, class_id in zip(xyxy, confs, classes):
            class_id = int(class_id)
            if self.config.classes_retenues is not None:
                if class_id not in self.config.classes_retenues:
                    continue
            largeur = x2 - x1
            hauteur = y2 - y1
            mini = self.config.taille_min_px
            if largeur < mini or hauteur < mini:
                continue
            detections.append(
                Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    score=float(score), class_id=class_id,
                )
            )
        return detections

    def changer_taille_entree(self, taille: int) -> None:
        self.config.taille_entree = int(taille)

    def noms_classes(self) -> dict[int, str]:
        noms = getattr(self.modele, "names", None)
        if isinstance(noms, dict):
            return dict(noms)
        if noms:
            return {i: str(n) for i, n in enumerate(noms)}
        return {}


def charger_modele(chemin: str):
    """Charge un modèle YOLO depuis le disque, en privilégiant le GPU."""
    import pathlib

    import torch
    from ultralytics import YOLO

    p = pathlib.Path(chemin)
    if not p.exists():
        raise FileNotFoundError(
            f"modèle introuvable : {p}\n"
            "Place le fichier .pt dans le dossier du projet, ou change le "
            "réglage « modele » dans l'interface."
        )
    if p.suffix.lower() not in (".pt", ".onnx"):
        raise ValueError(f"format de modèle non supporté : {p.suffix}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info("périphérique de détection : %s", device)
    if device == "cpu":
        log.warning("GPU non disponible : l'analyse sera très lente.")
    return YOLO(str(p))
```

- [ ] **Étape 4 : Lancer les tests, vérifier le passage**

```bash
python -m pytest tests/test_detecteur.py -v
```

Attendu : TOUT PASSE (7 tests).

- [ ] **Étape 5 : Commit**

```bash
git add compteur/detecteur.py tests/test_detecteur.py
git commit -m "feat: détection YOLO filtrable (seuil, taille, classes)"
```

---

## Tâche 4 : Tracking ByteTrack

**Fichiers :**
- Créer : `compteur/tracker.py`
- Créer : `tests/test_tracker.py`

**Interfaces :**
- Consomme : `compteur.config.Config`, `compteur.types.Detection` (Tâche 1, 3).
- Produit :
  - `compteur.tracker.Tracker(config: Config)`
  - `.mettre_a_jour(detections: list[Detection], frame_index: int) -> list[Track]`
    — met à jour le tracker, retourne les tracks confirmés
    (`frames_confirmation` respectés), bounding box et centre en pixels source.
  - `.reinitialiser() -> None`
  - `.nb_tracks_vus() -> int` — identifiants distincts vus depuis le début.
  - `compteur.tracker.TrackSuivi` — namedtuple interne
    `(track_id, bbox, center, age, nb_hits)`.

- [ ] **Étape 1 : Écrire les tests avec un faux backend (doivent échouer)**

Créer `tests/test_tracker.py` :

```python
import pytest

from compteur.config import Config
from compteur.tracker import Tracker
from compteur.types import Detection


def d(x, y, w=50, h=50, score=0.9, class_id=0):
    return Detection(x1=x, y1=y, x2=x + w, y2=y + h, score=score, class_id=class_id)


def test_detection_isolee_ne_produit_rien_tant_que_non_confirmee():
    t = Tracker(Config(frames_confirmation=3))
    assert t.mettre_a_jour([d(10, 10)], 0) == []


def test_track_confirme_apres_le_nombre_de_frames_demande():
    t = Tracker(Config(frames_confirmation=3))
    assert t.mettre_a_jour([d(10, 10)], 0) == []
    assert t.mettre_a_jour([d(11, 10)], 1) == []
    tracks = t.mettre_a_jour([d(12, 10)], 2)
    assert len(tracks) == 1
    assert tracks[0].confirmed is True


def test_meme_personne_conserve_son_identifiant():
    t = Tracker(Config(frames_confirmation=1))
    premier = t.mettre_a_jour([d(10, 10)], 0)[0]
    second = t.mettre_a_jour([d(12, 11)], 1)[0]
    assert premier.track_id == second.track_id


def test_deux_personnes_distinctes_ont_des_ids_distincts():
    t = Tracker(Config(frames_confirmation=1))
    tracks = t.mettre_a_jour([d(10, 10), d(400, 300)], 0)
    assert len({x.track_id for x in tracks}) == 2


def test_personne_perdue_puis_revue_recompte_comme_nouvelle():
    """Après survie_max frames sans détection, l'ID est réutilisé pour un autre."""
    t = Tracker(Config(frames_confirmation=1, survie_max=2))
    ancien = t.mettre_a_jour([d(10, 10)], 0)[0].track_id
    for i in range(1, 6):
        t.mettre_a_jour([], i)
    t.reinitialiser()
    neuf = t.mettre_a_jour([d(600, 400)], 6)[0].track_id
    assert neuf != ancien


def test_reinitialiser_oublie_tout():
    t = Tracker(Config(frames_confirmation=1))
    t.mettre_a_jour([d(10, 10)], 0)
    t.reinitialiser()
    assert t.nb_tracks_vus() == 0
    assert t.mettre_a_jour([d(10, 10)], 1)[0].track_id == 1


def test_bbox_exposee_en_pixels():
    t = Tracker(Config(frames_confirmation=1))
    track = t.mettre_a_jour([d(10, 20, 60, 80)], 0)[0]
    assert track.bbox == pytest.approx((10.0, 20.0, 60.0, 80.0))
    assert track.center == pytest.approx((35.0, 50.0))
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_tracker.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'compteur.tracker'`.

- [ ] **Étape 3 : Implémenter `compteur/tracker.py`**

```python
"""Tracking : détections frame par frame -> identités suivies (ByteTrack)."""

from __future__ import annotations

import logging
from typing import NamedTuple

from .config import Config
from .types import Detection, Track

log = logging.getLogger(__name__)


class TrackSuivi(NamedTuple):
    """Sortie brute du backend de tracking, avant filtrage."""

    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    age: int
    nb_hits: int


class _BackendSupervision:
    """ByteTrack via la bibliothèque supervision (testée, sans code maison)."""

    def __init__(self, config: Config) -> None:
        import supervision as sv

        self.config = config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.frames_confirmation,
            lost_track_buffer=config.survie_max,
            minimum_matching_threshold=config.seuil_matching,
        )

    def mettre_a_jour(self, detections: list[Detection]) -> list[TrackSuivi]:
        import numpy as np
        import supervision as sv

        if not detections:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
        xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=float)
        conf = np.array([d.score for d in detections], dtype=float)
        cls = np.array([d.class_id for d in detections], dtype=int)
        entree = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        self.tracker.update_with_detections(entree)

        sortie: list[TrackSuivi] = []
        for t in self.tracker.tracks:
            if t.track_id is None or not t.started:
                continue
            x1, y1, x2, y2 = (float(v) for v in t.xyxy)
            sortie.append(
                TrackSuivi(
                    track_id=int(t.track_id),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    age=int(t.age),
                    nb_hits=int(t.detections_count),
                )
            )
        return sortie


class Tracker:
    """Enveloppe du backend : applique les règles de confirmation de la config."""

    def __init__(self, config: Config, backend=None) -> None:
        self.config = config
        self.backend = backend if backend is not None else _BackendSupervision(config)
        self._confirmes: set[int] = set()
        self._vus: set[int] = set()

    def mettre_a_jour(
        self, detections: list[Detection], frame_index: int
    ) -> list[Track]:
        bruts = self.backend.mettre_a_jour(detections)
        self._vus.update(t.track_id for t in bruts)

        tracks: list[Track] = []
        for t in bruts:
            if t.nb_hits < self.config.frames_confirmation:
                continue
            self._confirmes.add(t.track_id)
            tracks.append(
                Track(
                    track_id=t.track_id,
                    center=t.center,
                    bbox=t.bbox,
                    age=t.age,
                    confirmed=True,
                )
            )
        return tracks

    def reinitialiser(self) -> None:
        self.backend = _BackendSupervision(self.config)
        self._confirmes.clear()
        self._vus.clear()

    def nb_tracks_vus(self) -> int:
        return len(self._vus)
```

- [ ] **Étape 4 : Lancer les tests, vérifier le passage**

```bash
python -m pytest tests/test_tracker.py -v
```

Attendu : TOUT PASSE (7 tests). Si supervision n'est pas installé,
`python -m pip install -r requirements.txt` d'abord.

- [ ] **Étape 5 : Commit**

```bash
git add compteur/tracker.py tests/test_tracker.py
git commit -m "feat: tracking ByteTrack avec règles de confirmation configurables"
```

---

## Tâche 5 : Comptage — l'orchestrateur

**Fichiers :**
- Créer : `compteur/compteur.py`
- Créer : `tests/test_compteur.py`

**Interfaces :**
- Consomme : `Detecteur` (T3), `Tracker` (T4), `Ligne` (T2), `Config`, `Types`
  (T1).
- Produit :
  - `compteur.compteur.analyser_video(chemin_video: str, config: Config, callback_frame: Callable[[FrameResult], None] | None = None, detecteur=None, tracker=None) -> Resultat`
    Le callback reçoit un `FrameResult` à chaque frame traitée.
    `detecteur` et `tracker` sont injectables pour les tests.
  - `compteur.compteur.Compteur(config: Config, detecteur, tracker)`
    - `.traiter_frame(img, frame_index, timestamp_s) -> FrameResult`
    - `.total`, `.presents`, `.evenements`
    - `.reinitialiser() -> None`
    - `.ajuster_ligne(ligne: Ligne) -> None`
  - Règle de comptage : un track est compté au premier frame où
    `ligne.a_traverse(dernier_center_connu, center)` est vrai, **une seule fois**
    par `track_id`, et seulement si le track est confirmé.

- [ ] **Étape 1 : Écrire les tests (doivent échouer)**

Créer `tests/test_compteur.py` :

```python
import numpy as np
import pytest

from compteur.compteur import Compteur
from compteur.config import Config
from compteur.ligne import Ligne
from compteur.types import Detection, Track


class FauxDetecteur:
    """Renvoie les détections qu'on lui a programmées, cadre par cadre."""

    def __init__(self):
        self.programme = {}
        self.config = None

    def detecter(self, img):
        return list(self.programme.get(img, []))


class FauxTracker:
    """Assigne des IDs croissants à chaque détection, sans logique de tracking."""

    def __init__(self):
        self._suivant = 1
        self.config = None

    def mettre_a_jour(self, detections, frame_index):
        return [
            Track(
                track_id=self._suivant + i,
                center=((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2),
                bbox=(d.x1, d.y1, d.x2, d.y2),
                age=10,
                confirmed=True,
            )
            for i, d in enumerate(detections)
        ]

    def reinitialiser(self):
        self._suivant = 1

    def nb_tracks_vus(self):
        return self._suivant


def image():
    return np.zeros((100, 200, 3), dtype=np.uint8)


def d(cx, cy, w=20, h=20):
    return Detection(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2,
                     score=0.9, class_id=0)


LIGNE = Ligne(p1=(100.0, 0.0), p2=(100.0, 100.0), epaisseur=20, sens=1)


def compteur(**kw):
    c = Compteur(Config(frames_confirmation=1, **kw), FauxDetecteur(), FauxTracker())
    c.ajuster_ligne(LIGNE)
    return c


def test_traversal_dans_le_sens_compte_une_fois():
    """Un ID stable traversant la ligne de gauche à droite compte exactement 1."""
    c = compteur()
    etat = {"i": 0}
    positions = [(50.0, 50.0), (80.0, 50.0), (150.0, 50.0), (190.0, 50.0)]

    def mu(d, f):
        x = positions[min(etat["i"], len(positions) - 1)]
        etat["i"] += 1
        return [Track(track_id=1, center=(x, 50.0), bbox=(0, 0, 1, 1), age=5,
                      confirmed=True)]

    c._tracker.mettre_a_jour = mu
    for i in range(len(positions)):
        c.traiter_frame(image(), i, i * 0.04)
    assert c.total == 1
    assert c.evenements[0].frame == 2


def test_meme_id_compte_une_seule_fois():
    c = compteur()
    track_fixe = Track(track_id=7, center=(50.0, 50.0), bbox=(0, 0, 1, 1), age=5,
                       confirmed=True)
    c._derniers_centers = {7: (50.0, 50.0)}
    c._tracker.mettre_a_jour = lambda d, f: [track_fixe]
    c.traiter_frame(image(), 0, 0.0)
    assert c.total == 0

    c._derniers_centers[7] = (50.0, 50.0)
    c._tracker.mettre_a_jour = lambda d, f: [
        Track(track_id=7, center=(150.0, 50.0), bbox=(0, 0, 1, 1), age=6, confirmed=True)
    ]
    c.traiter_frame(image(), 1, 0.04)
    assert c.total == 1

    c._tracker.mettre_a_jour = lambda d, f: [
        Track(track_id=7, center=(50.0, 50.0), bbox=(0, 0, 1, 1), age=7, confirmed=True)
    ]
    c.traiter_frame(image(), 2, 0.08)
    assert c.total == 1, "un retour en arrière ne doit pas recompter"


def test_sens_inverse_ne_compte_pas():
    c = Compteur(Config(frames_confirmation=1), FauxDetecteur(), FauxTracker())
    c.ajuster_ligne(Ligne(p1=(100.0, 0.0), p2=(100.0, 100.0), epaisseur=20, sens=-1))
    etat = {"i": 0}
    positions = [(150.0, 50.0), (50.0, 50.0)]  # traversée dans le mauvais sens

    def mu(d, f):
        x = positions[etat["i"]]
        etat["i"] += 1
        return [Track(track_id=1, center=x, bbox=(0, 0, 1, 1), age=5, confirmed=True)]

    c._tracker.mettre_a_jour = mu
    c._derniers_centers[1] = (50.0, 50.0)
    c.traiter_frame(image(), 0, 0.0)
    c.traiter_frame(image(), 1, 0.04)
    assert c.total == 0


def test_presents_refletent_les_tracks_confirmes():
    c = compteur()
    c._tracker.mettre_a_jour = lambda d, f: [
        Track(track_id=1, center=(10.0, 10.0), bbox=(0, 0, 1, 1), age=5, confirmed=True),
        Track(track_id=2, center=(180.0, 90.0), bbox=(0, 0, 1, 1), age=5, confirmed=True),
    ]
    r = c.traiter_frame(image(), 0, 0.0)
    assert r.presents == 2


def test_evenement_horodate_le_comptage():
    c = compteur()
    c._derniers_centers[7] = (50.0, 50.0)
    c._tracker.mettre_a_jour = lambda d, f: [
        Track(track_id=7, center=(150.0, 50.0), bbox=(0, 0, 1, 1), age=6, confirmed=True)
    ]
    c.traiter_frame(image(), 42, 1.68)
    ev = c.evenements[0]
    assert ev.frame == 42
    assert ev.timestamp_s == pytest.approx(1.68)
    assert ev.track_id == 7


def test_hysteresis_exige_deux_frames_avant_de_comptter():
    c = Compteur(Config(frames_confirmation=1, frames_hysteresis=2), FauxDetecteur(),
                 FauxTracker())
    c.ajuster_ligne(LIGNE)
    c._derniers_centers[7] = (50.0, 50.0)
    c._tracker.mettre_a_jour = lambda d, f: [
        Track(track_id=7, center=(160.0, 50.0), bbox=(0, 0, 1, 1), age=6, confirmed=True)
    ]
    c.traiter_frame(image(), 0, 0.0)
    assert c.total == 0, "un simple bruit ne doit pas déclencher de comptage"


def test_sans_ligne_rien_n_est_comptte():
    c = Compteur(Config(frames_confirmation=1), FauxDetecteur(), FauxTracker())
    c._tracker.mettre_a_jour = lambda d, f: [
        Track(track_id=1, center=(150.0, 50.0), bbox=(0, 0, 1, 1), age=6, confirmed=True)
    ]
    r = c.traiter_frame(image(), 0, 0.0)
    assert c.total == 0
    assert r.presents == 1
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_compteur.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'compteur.compteur'`.

- [ ] **Étape 3 : Implémenter `compteur/compteur.py`**

```python
"""Orchestrateur : video + config -> résultat.

Ce module ne connaît pas l'interface. Il est appelé par l'IHM, par les tests,
et demain par une éventuelle API web.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from .config import Config
from .detecteur import Detecteur, charger_modele
from .ligne import Ligne
from .tracker import Tracker
from .types import Evenement, FrameResult, Resultat, Track

log = logging.getLogger(__name__)


class Compteur:
    """Machine à états du comptage : une frame à la fois."""

    def __init__(self, config: Config, detecteur, tracker) -> None:
        self.config = config
        self.detecteur = detecteur
        self.tracker = tracker
        self.ligne: Ligne | None = None
        self.total = 0
        self.presents = 0
        self.presents_max = 0
        self.evenements: list[Evenement] = []
        self._derniers_centers: dict[int, tuple[float, float]] = {}
        self._coupes_side: dict[int, tuple[int, int]] = {}
        self._deja_comptes: set[int] = set()
        self._somme_presents = 0
        self._nb_frames_vues = 0

    def ajuster_ligne(self, ligne: Ligne) -> None:
        self.ligne = ligne

    def reinitialiser(self) -> None:
        self.total = 0
        self.presents = 0
        self.presents_max = 0
        self.evenements = []
        self._derniers_centers = {}
        self._coupes_side = {}
        self._deja_comptes = set()
        self._somme_presents = 0
        self._nb_frames_vues = 0
        self.tracker.reinitialiser()

    def traiter_frame(
        self, img: np.ndarray, frame_index: int, timestamp_s: float
    ) -> FrameResult:
        detections = self.detecteur.detecter(img)
        tracks = self.tracker.mettre_a_jour(detections, frame_index)
        nouveaux: list[Evenement] = []

        for t in tracks:
            precedent = self._derniers_centers.get(t.track_id)
            self._derniers_centers[t.track_id] = t.center

            if self.ligne is None or t.track_id in self._deja_comptes or precedent is None:
                continue

            if self.ligne.a_traverse(precedent, t.center):
                # Hystérésis : on exige que le point soit resté du côté de
                # départ assez longtemps pour écarter les micro-rebonds.
                cote_avant, stable = self._coupes_side.get(t.track_id, (0, 0))
                cote = self.ligne.point_du_cote(precedent)
                if cote != 0 and cote == cote_avant:
                    stable += 1
                else:
                    stable = 0
                self._coupes_side[t.track_id] = (cote, stable)
                if stable >= self.ligne.hysteresis:
                    self._deja_comptes.add(t.track_id)
                    self.total += 1
                    ev = Evenement(
                        frame=frame_index,
                        timestamp_s=timestamp_s,
                        x=t.center[0],
                        y=t.center[1],
                        track_id=t.track_id,
                    )
                    self.evenements.append(ev)
                    nouveaux.append(ev)
            else:
                self._coupes_side[t.track_id] = (self.ligne.point_du_cote(t.center), 0)

        # Purge des tracks disparus pour éviter les fuites mémoire.
        vus = {t.track_id for t in tracks}
        for registre in (self._derniers_centers, self._coupes_side, self._deja_comptes):
            for identifiant in list(registre) - vus:
                if identifiant not in vus:
                    registre.discard(identifiant)

        self.presents = len(tracks)
        self.presents_max = max(self.presents_max, self.presents)
        self._somme_presents += self.presents
        self._nb_frames_vues += 1

        return FrameResult(
            image=img,
            detections=detections,
            tracks=tracks,
            total=self.total,
            presents=self.presents,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            evenements=nouveaux,
        )

    def resultat(self, modele: str, nb_frames: int, secondes: float) -> Resultat:
        presents_moyen = (
            self._somme_presents / self._nb_frames_vues if self._nb_frames_vues else 0.0
        )
        return Resultat(
            total=self.total,
            evenements=list(self.evenements),
            config=self.config,
            modele=modele,
            nb_frames=nb_frames,
            presents_max=self.presents_max,
            presents_moyen=presents_moyen,
            secondes=secondes,
        )


def analyser_video(
    chemin_video: str,
    config: Config,
    callback_frame: Callable[[FrameResult], None] | None = None,
    detecteur=None,
    tracker=None,
) -> Resultat:
    """Analyse un fichier vidéo de bout en bout.

    Si ``callback_frame`` est fourni, il est appelé à chaque frame — c'est ce
    mode qu'utilise l'interface pour l'affichage en direct.
    """
    import cv2

    if detecteur is None:
        detecteur = Detecteur(charger_modele(config.modele), config)
    if tracker is None:
        tracker = Tracker(config)

    compteur = Compteur(config, detecteur, tracker)
    if config.ligne is not None:
        compteur.ajuster_ligne(
            Ligne(
                (config.ligne[0], config.ligne[1]),
                (config.ligne[2], config.ligne[3]),
                epaisseur=config.epaisseur_bande,
                sens=config.sens,
                hysteresis=config.frames_hysteresis,
            )
        )

    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"vidéo illisible : {chemin_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    debut = time.time()
    index = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            resultat_frame = compteur.traiter_frame(img, index, index / fps)
            if callback_frame is not None:
                callback_frame(resultat_frame)
            index += 1
            if total_frames and index % 100 == 0:
                log.info("%d/%d frames", index, total_frames)
    finally:
        cap.release()

    return compteur.resultat(config.modele, index, time.time() - debut)
```

> **Note d'implémentation** : `presents_moyen` est déjà correctement calculé
> dans le code ci-dessus (moyenne du nombre de présents sur toutes les frames,
> via `_somme_presents / _nb_frames_vues`). Le test correspondant est :
>
> ```python
> def test_presents_moyen_est_une_moyenne():
>     c = compteur()
>     for i, n in enumerate([2, 4, 6]):
>         c._tracker.mettre_a_jour = lambda d, f, n=n: [
>             Track(track_id=k + 1, center=(10.0 * k, 10.0), bbox=(0, 0, 1, 1),
>                   age=5, confirmed=True)
>             for k in range(n)
>         ]
>         c.traiter_frame(image(), i, i * 0.04)
>     r = c.resultat("x.pt", 3, 0.12)
>     assert r.presents_moyen == pytest.approx(4.0)
>     assert r.presents_max == 6
> ```
>
> Ajouter ce test dans `tests/test_compteur.py`.

- [ ] **Étape 4 : Lancer les tests, vérifier le passage**

```bash
python -m pytest tests/test_compteur.py -v
```

Attendu : TOUT PASSE (9 tests : 8 de la liste ci-dessus plus
`test_presents_moyen_est_une_moyenne`).

```bash
python -m pytest tests/ -v
```

Attendu : TOUT PASSE.

- [ ] **Étape 7 : Commit**

```bash
git add compteur/compteur.py tests/test_compteur.py
git commit -m "feat: orchestrateur de comptage — franchissement, hystérésis, horodatage"
```

---

## Tâche 6 : Rapports et statistiques

**Fichiers :**
- Créer : `compteur/rapport.py`
- Créer : `tests/test_rapport.py`

**Interfaces :**
- Consomme : `Resultat`, `Evenement` (T1), `Config` (T1).
- Produit :
  - `compteur.rapport.ecrire_csv(resultat: Resultat, chemin: str | Path) -> Path`
  - `compteur.rapport.ecrire_json(resultat: Resultat, chemin: str | Path) -> Path`
  - `compteur.rapport.statistiques(resultat: Resultat) -> dict` — clés :
    `total`, `duree_s`, `personnes_par_minute`, `presents_max`,
    `presents_moyen`, `debit_max_par_minute`, `modele`, `nb_frames`.
  - `compteur.rapport.chemins_par_defaut(chemin_video: str, dossier: str | Path) -> dict`
    — renvoie `{"csv": ..., "json": ...}`.

- [ ] **Étape 1 : Écrire les tests (doivent échouer)**

Créer `tests/test_rapport.py` :

```python
import json

from compteur.config import Config
from compteur.rapport import (
    chemins_par_defaut,
    ecrire_csv,
    ecrire_json,
    statistiques,
)
from compteur.types import Evenement, Resultat


def resultat_fictif():
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


def test_csv_ecrit_une_ligne_par_evenement(tmp_path):
    p = ecrire_csv(resultat_fictif(), tmp_path / "r.csv")
    lignes = p.read_text(encoding="utf-8").strip().splitlines()
    assert lignes[0] == "frame,timestamp_s,x,y,track_id"
    assert len(lignes) == 4


def test_json_contient_total_et_config(tmp_path):
    p = ecrire_json(resultat_fictif(), tmp_path / "r.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["total"] == 3
    assert d["config"]["modele"] == "yolov8n-head.pt"
    assert d["config"]["seuil_confiance"] == 0.3
    assert len(d["evenements"]) == 3


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


def test_chemins_par_defaut():
    d = chemins_par_defaut("D:/videos/manifestation.mp4", "sortie")
    assert d["csv"].endswith("manifestation_head_results.csv")
    assert d["json"].endswith("manifestation_head_results.json")
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_rapport.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'compteur.rapport'`.

- [ ] **Étape 3 : Implémenter `compteur/rapport.py`**

```python
"""Exports et statistiques d'une analyse terminée."""

from __future__ import annotations

import json
import pathlib

from .types import Resultat

ENTETE_CSV = "frame,timestamp_s,x,y,track_id"


def chemin_video_nom(chemin_video: str) -> str:
    return pathlib.Path(chemin_video).stem


def chemins_par_defaut(chemin_video: str, dossier: str | pathlib.Path) -> dict:
    base = chemin_video_nom(chemin_video)
    d = pathlib.Path(dossier)
    return {
        "csv": d / f"{base}_head_results.csv",
        "json": d / f"{base}_head_results.json",
    }


def ecrire_csv(resultat: Resultat, chemin: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(chemin)
    p.parent.mkdir(parents=True, exist_ok=True)
    lignes = [ENTETE_CSV]
    lignes.extend(ev.vers_ligne_csv() for ev in resultat.evenements)
    p.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return p


def ecrire_json(resultat: Resultat, chemin: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(chemin)
    p.parent.mkdir(parents=True, exist_ok=True)
    donnees = {
        "total": resultat.total,
        "modele": resultat.modele,
        "nb_frames": resultat.nb_frames,
        "presents_max": resultat.presents_max,
        "presents_moyen": resultat.presents_moyen,
        "secondes": resultat.secondes,
        "config": resultat.config.vers_dict() if resultat.config else None,
        "evenements": [
            {
                "frame": ev.frame,
                "timestamp_s": ev.timestamp_s,
                "x": ev.x,
                "y": ev.y,
                "track_id": ev.track_id,
            }
            for ev in resultat.evenements
        ],
        "statistiques": statistiques(resultat),
    }
    p.write_text(
        json.dumps(donnees, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return p


def statistiques(resultat: Resultat) -> dict:
    """Indicateurs de synthèse. Aucun chiffre d'erreur : on n'a pas la vérité."""
    duree = float(resultat.secondes or 0.0)
    if duree > 0 and resultat.total:
        debit = resultat.total / duree * 60.0
    else:
        debit = 0.0
    if not resultat.evenements:
        debit_max = 0.0
    else:
        debut = resultat.evenements[0].timestamp_s
        fenetre = 60.0
        brut = [
            ev.timestamp_s
            for ev in resultat.evenements
            if debut <= ev.timestamp_s <= debut + fenetre
        ]
        debit_max = len(brut) if duree else float(len(brut))
    return {
        "total": resultat.total,
        "duree_s": round(duree, 2),
        "personnes_par_minute": round(debit, 1),
        "debit_max_par_minute": round(debit_max, 1),
        "presents_max": resultat.presents_max,
        "presents_moyen": round(resultat.presents_moyen, 2),
        "nb_frames": resultat.nb_frames,
        "modele": resultat.modele,
    }
```

- [ ] **Étape 4 : Lancer les tests, vérifier le passage**

```bash
python -m pytest tests/test_rapport.py -v
```

Attendu : TOUT PASSE (6 tests).

- [ ] **Étape 5 : Commit**

```bash
git add compteur/rapport.py tests/test_rapport.py
git commit -m "feat: exports CSV/JSON et statistiques de débit"
```

---

## Tâche 7 : Comparaison de modèles (tête vs personne) — outillage de décision

**Fichiers :**
- Créer : `tools/comparer_modeles.py`
- Créer : `tests/test_export_video_synthetique.py`

**Interfaces :**
- Consomme : `analyser_video` (T5), `Config` (T1), `rapport` (T6).
- Produit : script en ligne de commande
  `python tools/comparer_modeles.py --video <chemin> --modeles a.pt b.pt --ligne x1,y1,x2,y2 --sens 1`
  qui affiche un tableau comparatif (total, débit, personnes présentes max) et
  écrit `sortie/comparaison.json`.

- [ ] **Étape 1 : Écrire le test de bout en bout sur vidéo synthétique**

Créer `tests/test_export_video_synthetique.py` :

```python
import numpy as np

from compteur.compteur import Compteur
from compteur.config import Config
from compteur.ligne import Ligne
from compteur.types import Detection, Track


def test_video_synthetique_compte_bien_3_passages():
    """Trois disques qui traversent une ligne verticale : total attendu = 3."""
    largeur, hauteur, fps = 320, 240, 10
    nb_frames = 60
    departs = [(40.0, 1), (80.0, 8), (120.0, 15)]  # (x, frame_depart)

    config = Config(frames_confirmation=1, epaisseur_bande=15, frames_hysteresis=0)
    compteur = Compteur(config, detecteur=None, tracker=None)

    # Détecteur de test : un disque blanc = une personne.
    class DetecteurDisques:
        def detecter(self, img):
            seuil = 200
            masque = (img[:, :, 0] > seuil).astype(np.uint8)
            colonnes = np.where(masque.any(axis=0))[0]
            sorties = []
            for cx in colonnes:
                ys = np.where(masque[:, cx] > 0)[0]
                if len(ys) < 4:
                    continue
                sorties.append(
                    Detection(x1=float(cx), y1=float(ys.min()), x2=float(cx + 4),
                              y2=float(ys.max()), score=0.9, class_id=0)
                )
            return sorties

    class TrackerCompteur:
        """Associe chaque colonne du frame à l'ID le plus proche connu."""
        def __init__(self):
            self.memoire: dict[int, tuple[int, float]] = {}
            self._prochain = 1

        def mettre_a_jour(self, detections, frame_index):
            tracks = []
            for det in detections:
                cx = (det.x1 + det.x2) / 2
                if det.x1 in self.memoire:
                    tid, _ = self.memoire[det.x1]
                else:
                    tid = self._prochain
                    self._prochain += 1
                self.memoire[det.x1] = (tid, cx)
                tracks.append(
                    Track(track_id=tid, center=(cx, (det.y1 + det.y2) / 2),
                          bbox=(det.x1, det.y1, det.x2, det.y2), age=1,
                          confirmed=True)
                )
            return tracks

        def reinitialiser(self):
            self.memoire = {}
            self._prochain = 1

        def nb_tracks_vus(self):
            return self._prochain

    compteur.detecteur = DetecteurDisques()
    compteur.tracker = TrackerCompteur()
    compteur.ajuster_ligne(Ligne(p1=(160.0, 0.0), p2=(160.0, 240.0), epaisseur=15,
                                 sens=1, hysteresis=0))

    total = 0
    for i in range(nb_frames):
        img = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
        for x0, f0 in departs:
            x = x0 + (i - f0) * 4
            if 0 <= x < largeur - 5:
                img[100:140, int(x) : int(x) + 5] = 255
        if i == 0:
            img[100:140, int(departs[0][0]) : int(departs[0][0]) + 5] = 255
        r = compteur.traiter_frame(img, i, i / fps)
        total = r.total
    assert total == 3, f"attendu 3 passages, obtenu {total}"
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_export_video_synthetique.py -v
```

Attendu : ÉCHEC sur le compte (la vidéo synthétique sera traitée par le code
avant d'être correcte) — c'est le but.

- [ ] **Étape 3 : Créer `tools/comparer_modeles.py`**

```python
"""Compare plusieurs modèles YOLO sur la même vidéo.

Usage :
    python tools/comparer_modeles.py --video "D:/videos/manif.mp4" \
        --modeles yolov8n-head.pt yolov8n.pt \
        --ligne 556,90,588,654 --sens 1 --dossier sortie
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from compteur.compteur import analyser_video  # noqa: E402
from compteur.config import Config  # noqa: E402
from compteur.rapport import statistiques  # noqa: E402

log = logging.getLogger("comparer")


def analyser(modele: str, args: argparse.Namespace) -> dict:
    config = Config(
        modele=modele,
        seuil_confiance=args.seuil,
        taille_min_px=args.taille_min,
        taille_entree=args.taille_entree,
        frames_confirmation=args.confirmation,
        survie_max=args.survie,
        epaisseur_bande=args.epaisseur,
        sens=args.sens,
        frames_hysteresis=args.hysteresis,
        ligne=tuple(args.ligne),
    )
    resultat = analyser_video(args.video, config)
    return {
        "modele": modele,
        **statistiques(resultat),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--modeles", nargs="+", required=True)
    p.add_argument("--ligne", nargs=4, type=float, required=True,
                   metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--sens", type=int, default=1, choices=(1, -1))
    p.add_argument("--seuil", type=float, default=0.25)
    p.add_argument("--taille-min", type=int, default=20)
    p.add_argument("--taille-entree", type=int, default=640)
    p.add_argument("--confirmation", type=int, default=3)
    p.add_argument("--survie", type=int, default=30)
    p.add_argument("--epaisseur", type=int, default=30)
    p.add_argument("--hysteresis", type=int, default=2)
    p.add_argument("--dossier", default="sortie")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    resultats = []
    for modele in args.modeles:
        log.info("analyse avec %s", modele)
        resultats.append(analyser(modele, args))

    largeur_modele = max(len(r["modele"]) for r in resultats) + 2
    print()
    print(f"{'modèle'.ljust(largeur_modele)} total  débit/min  présents max")
    print("-" * (largeur_modele + 32))
    for r in sorted(resultats, key=lambda x: -x["total"]):
        print(
            f"{r['modele'].ljust(largeur_modele)} "
            f"{str(r['total']).rjust(5)}  "
            f"{str(r['personnes_par_minute']).rjust(9)}  "
            f"{str(r['presents_max']).rjust(13)}"
        )
    print()

    dossier = pathlib.Path(args.dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    sortie = dossier / "comparaison.json"
    sortie.write_text(
        json.dumps(resultats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("résultats écrits dans %s", sortie)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Étape 4 : Lancer le test synthétique, vérifier le passage**

```bash
python -m pytest tests/test_export_video_synthetique.py -v
```

Attendu : PASSE, 3 passages comptés.

- [ ] **Étape 5 : Lancer la suite complète**

```bash
python -m pytest tests/ -v
```

Attendu : TOUT PASSE.

- [ ] **Étape 6 : Commit**

```bash
git add tools/ tests/
git commit -m "feat: outil de comparaison de modèles + test bout-en-bout sur vidéo synthétique"
```

---

## Tâche 8 : Interface — style, overlay, widget vidéo

**Fichiers :**
- Créer : `interface/__init__.py`
- Créer : `interface/style.py`
- Créer : `interface/overlay.py`
- Créer : `interface/widgets_video.py`
- Créer : `tests/test_overlay.py`

**Interfaces :**
- Consomme : `FrameResult`, `Ligne` (T1, T2).
- Produit :
  - `interface.style.appliquer_style(app) -> None` — thème sombre.
  - `interface.overlay.dessiner(img, resultat: FrameResult, ligne: Ligne | None,
    afficher_ids: bool = True, flash: bool = False) -> np.ndarray`
  - `interface.widgets_video.WidgetVideo(parent=None)` —
    signaux `clic(x: int, y: int)`, `image_changee(np.ndarray)`.

- [ ] **Étape 1 : Écrire les tests d'overlay (doivent échouer)**

Créer `tests/test_overlay.py` :

```python
import numpy as np

from compteur.types import Detection, FrameResult, Track
from interface.overlay import dessiner


def image():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_overlay_ne_mute_pas_l_source():
    img = image()
    avant = img.copy()
    dessiner(img, FrameResult(image=img))
    assert np.array_equal(img, avant)


def test_overlay_meme_taille_entree_sortie():
    img = image()
    assert dessiner(img, FrameResult(image=img)).shape == img.shape


def test_boite_dessinee_change_l_image():
    img = image()
    r = FrameResult(
        image=img,
        detections=[Detection(50, 50, 90, 120, 0.9, 0)],
        tracks=[Track(1, (70, 85), (50, 50, 90, 120), 3, True)],
    )
    assert dessiner(img, r).any(), "les boîtes doivent être visibles"


def test_sans_detection_image_inchangee():
    img = image()
    assert not dessiner(img, FrameResult(image=img)).any()


def test_flash_change_l_image():
    img = image()
    r = FrameResult(image=img, total=5)
    assert dessiner(img, r, flash=True).any(), "le flash doit être visible"
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_overlay.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'interface'`.

- [ ] **Étape 3 : Créer `interface/__init__.py`, `interface/style.py`, `interface/overlay.py`**

`interface/__init__.py` — vide (juste un docstring).

`interface/style.py` :

```python
"""Thème sombre et lisible (usage en extérieur)."""

from __future__ import annotations

STYLE = """
QWidget { background-color: #14161a; color: #e6e6e6; font-size: 13px; }
QPushButton {
    background-color: #262a31; border: 1px solid #3a4048; border-radius: 5px;
    padding: 7px 14px;
}
QPushButton:hover { background-color: #31363f; }
QPushButton:disabled { color: #6b7280; border-color: #2a2e35; }
QPushButton#primaire { background-color: #2563eb; border-color: #2563eb; color: #fff; }
QPushButton#primaire:hover { background-color: #1d4ed8; }
QGroupBox {
    border: 1px solid #2b2f36; border-radius: 6px; margin-top: 14px; padding-top: 8px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa3af; }
QSlider::groove:horizontal { height: 5px; background: #2b2f36; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #2563eb; width: 15px; margin: -6px 0; border-radius: 7px;
}
QLabel#compteur { font-size: 76px; font-weight: 700; color: #4ade80; }
QLabel#compteur_titre { font-size: 14px; color: #9aa3af; letter-spacing: 2px; }
QLabel#sous_titre { color: #9ca3af; }
QProgressBar {
    border: 1px solid #2b2f36; border-radius: 4px; text-align: center; height: 18px;
}
QProgressBar::chunk { background-color: #2563eb; }
"""


def appliquer_style(app) -> None:
    app.setStyleSheet(STYLE)
```

`interface/overlay.py` :

```python
"""Dessin des annotations sur l'image. Ne décide rien, ne lit aucune config."""

from __future__ import annotations

import cv2
import numpy as np

COULEUR_BOITE = (240, 160, 60)
COULEUR_TRACK = (200, 200, 80)
COULEUR_TEXTE = (255, 255, 255)
COULEUR_FLASH = (80, 80, 240)


def dessiner(
    img: np.ndarray,
    resultat,
    ligne=None,
    afficher_ids: bool = True,
    flash: bool = False,
) -> np.ndarray:
    """Retourne une copie annotée. ``img`` n'est jamais modifiée."""
    sortie = img.copy()

    for d in getattr(resultat, "detections", []) or []:
        cv2.rectangle(
            sortie,
            (int(d.x1), int(d.y1)),
            (int(d.x2), int(d.y2)),
            COULEUR_BOITE,
            2,
        )

    for t in getattr(resultat, "tracks", []) or []:
        cx, cy = int(t.center[0]), int(t.center[1])
        cv2.circle(sortie, (cx, cy), 4, COULEUR_TRACK, -1)
        if afficher_ids:
            cv2.putText(
                sortie, str(t.track_id), (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COULEUR_TEXTE, 2,
            )

    if ligne is not None:
        sortie = ligne.dessiner(sortie)

    if flash:
        hauteur, largeur = sortie.shape[:2]
        cv2.rectangle(sortie, (0, 0), (largeur - 1, hauteur - 1), COULEUR_FLASH, 8)

    return sortie
```

- [ ] **Étape 4 : Écrire et exécuter `interface/widgets_video.py`**

```python
"""Affichage de l'image annotée, avec émission des clics pour tracer la ligne."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class WidgetVideo(QLabel):
    """Affiche une image numpy et remonte les clics en coordonnées pixel."""

    clic = Signal(int, int)
    image_changee = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #0b0c0e; border: 1px solid #2b2f36;")
        self._image: np.ndarray | None = None
        self._facteur = 1.0
        self._offset = (0, 0)

    def definir_image(self, img: np.ndarray | None) -> None:
        self._image = img
        if img is None:
            self.clear()
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._recalculer_echelle()
        self._peindre()
        self.image_changee.emit(img)

    def _taille_widget(self) -> tuple[int, int]:
        return max(1, self.width()), max(1, self.height())

    def _recalculer_echelle(self) -> None:
        if self._image is None:
            return
        ih, iw = self._image.shape[:2]
        lw, lh = self._taille_widget()
        self._facteur = min(lw / iw, lh / ih)
        dw, dh = int(iw * self._facteur), int(ih * self._facteur)
        self._offset = ((lw - dw) // 2, (lh - dh) // 2)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._recalculer_echelle()
        self._peindre()

    def _peindre(self) -> None:
        if self._image is None:
            return
        pixmap = self._pixmap.scaledToWidth(
            max(1, int(self._image.shape[1] * self._facteur)),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def _vers_pixels(self, x: int, y: int) -> tuple[int, int]:
        dx = x - self._offset[0]
        dy = y - self._offset[1]
        if self._image is None or self._facteur <= 0:
            return 0, 0
        return int(dx / self._facteur), int(dy / self._facteur)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            px, py = self._vers_pixels(event.position().x(), event.position().y())
            self.clic.emit(px, py)
```

- [ ] **Étape 5 : Lancer les tests d'overlay, vérifier le passage**

```bash
python -m pytest tests/test_overlay.py -v
```

Attendu : TOUT PASSE (5 tests).

- [ ] **Étape 6 : Vérifier que l'import du paquet interface ne casse rien**

```bash
python -c "from interface.overlay import dessiner; from interface.widgets_video import WidgetVideo; print('ok')"
```

Attendu : `ok`.

- [ ] **Étape 7 : Commit**

```bash
git add interface/ tests/test_overlay.py
git commit -m "feat: thème sombre, overlay d'annotations, widget vidéo cliquable"
```

---

## Tâche 9 : Panneau de réglages

**Fichiers :**
- Créer : `interface/panneau_reglages.py`
- Créer : `tests/test_reglages.py`

**Interfaces :**
- Consomme : `Config` (T1).
- Produit :
  - `interface.panneau_reglages.PanneauReglages(config: Config, parent=None)`
    - signal `config_modifiee(Config)`
    - `.lire() -> Config` — lit l'état courant des widgets.
    - `.appliquer(config: Config) -> None` — pousse une config dans l'IHM.
    - `.enregistrer_profil(nom: str) -> pathlib.Path` — écrit
      `config/profils/<nom>.json`.
    - `.charger_profil(nom: str) -> bool`

- [ ] **Étape 1 : Écrire les tests (doivent échouer)**

Créer `tests/test_reglages.py` :

```python
import pathlib

import pytest

PySide6 = pytest.importorskip("PySide6")

from compteur.config import Config  # noqa: E402
from interface.panneau_reglages import PanneauReglages  # noqa: E402


@pytest.fixture
def application():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_lire_renvoie_la_config(application):
    p = PanneauReglages(Config(seuil_confiance=0.4))
    assert p.lire().seuil_confiance == 0.4


def test_modifier_un_slider_met_a_jour_la_config(application):
    p = PanneauReglages(Config())
    p.definir_seuil(0.65)
    assert p.lire().seuil_confiance == pytest.approx(0.65)


def test_appliquer_une_config_externe(application):
    p = PanneauReglages(Config())
    p.appliquer(Config(seuil_confiance=0.8, epaisseur_bande=70))
    assert p.lire().seuil_confiance == 0.8
    assert p.lire().epaisseur_bande == 70


def test_profil_enregistre_et_recharge(application, tmp_path):
    p = PanneauReglages(Config())
    p.definir_seuil(0.33)
    p.enregistrer_profil("essai", dossier=tmp_path)
    assert (tmp_path / "essai.json").exists()
    p2 = PanneauReglages(Config(seuil_confiance=0.9))
    assert p2.charger_profil("essai", dossier=tmp_path) is True
    assert p2.lire().seuil_confiance == pytest.approx(0.33)


def test_charger_profil_inexistant_returns_false(application, tmp_path):
    p = PanneauReglages(Config())
    assert p.charger_profil("inconnu", dossier=tmp_path) is False


def test_signal_emis_quand_la_config_change(application):
    p = PanneauReglages(Config())
    captures = []
    p.config_modifiee.connect(captures.append)
    p.definir_seuil(0.5)
    assert len(captures) == 1
    assert captures[0].seuil_confiance == pytest.approx(0.5)
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_reglages.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'interface.panneau_reglages'`.

- [ ] **Étape 3 : Implémenter `interface/panneau_reglages.py`**

```python
"""Panneau de réglages : produit une Config à partir de widgets Qt."""

from __future__ import annotations

import logging
import pathlib

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from compteur.config import Config

log = logging.getLogger(__name__)

Dossier_PROFILS = pathlib.Path("config") / "profils"


class PanneauReglages(QWidget):
    """Trois groupes de réglages + gestion des profils nommés."""

    config_modifiee = Signal(object)

    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._widgets: dict[str, QWidget] = {}
        racine = QVBoxLayout(self)
        racine.setContentsMargins(8, 8, 8, 8)
        racine.addWidget(self._groupe_detection(config))
        racine.addWidget(self._groupe_tracker(config))
        racine.addWidget(self._groupe_ligne(config))
        racine.addWidget(self._groupe_profils())
        racine.addStretch(1)
        self._bloquer = False

    # -- Construction ----------------------------------------------------

    def _groupe_detection(self, c: Config) -> QGroupBox:
        g = QGroupBox("Détection")
        v = QVBoxLayout(g)
        v.addWidget(QLabel("Modèle (fichier .pt)"))
        self._modele = QComboBox()
        self._modele.setEditable(True)
        self._modele.addItems(self._modeles_disponibles())
        self._modele.setCurrentText(c.modele)
        self._modele.currentTextChanged.connect(self._emettre)
        v.addWidget(self._modele)
        self._widgets["modele"] = self._modele

        v.addWidget(QLabel(f"Seuil de confiance — {c.seuil_confiance:.2f}"))
        self._seuil = QDoubleSpinBox()
        self._seuil.setRange(0.05, 0.95)
        self._seuil.setSingleStep(0.05)
        self._seuil.setValue(c.seuil_confiance)
        self._seuil.valueChanged.connect(self._emettre)
        v.addWidget(self._seuil)
        self._widgets["seuil_confiance"] = self._seuil

        v.addWidget(QLabel("Taille minimale (px)"))
        self._taille_min = QSpinBox()
        self._taille_min.setRange(0, 300)
        self._taille_min.setValue(c.taille_min_px)
        self._taille_min.valueChanged.connect(self._emettre)
        v.addWidget(self._taille_min)
        self._widgets["taille_min_px"] = self._taille_min

        v.addWidget(QLabel("Taille d'entrée"))
        self._taille_entree = QComboBox()
        self._taille_entree.addItems(["320", "640", "1280"])
        self._taille_entree.setCurrentText(str(c.taille_entree))
        self._taille_entree.currentTextChanged.connect(self._emettre)
        v.addWidget(self._taille_entree)
        self._widgets["taille_entree"] = self._taille_entree
        return g

    def _groupe_tracker(self, c: Config) -> QGroupBox:
        g = QGroupBox("Tracker")
        v = QVBoxLayout(g)

        v.addWidget(QLabel(f"Frames de confirmation — {c.frames_confirmation}"))
        self._confirmation = QSpinBox()
        self._confirmation.setRange(1, 20)
        self._confirmation.setValue(c.frames_confirmation)
        self._confirmation.valueChanged.connect(self._emettre)
        v.addWidget(self._confirmation)
        self._widgets["frames_confirmation"] = self._confirmation

        v.addWidget(QLabel("Survie max sans détection (frames)"))
        self._survie = QSpinBox()
        self._survie.setRange(1, 120)
        self._survie.setValue(c.survie_max)
        self._survie.valueChanged.connect(self._emettre)
        v.addWidget(self._survie)
        self._widgets["survie_max"] = self._survie

        v.addWidget(QLabel("Seuil de matching"))
        self._matching = QDoubleSpinBox()
        self._matching.setRange(0.1, 0.9)
        self._matching.setSingleStep(0.05)
        self._matching.setValue(c.seuil_matching)
        self._matching.valueChanged.connect(self._emettre)
        v.addWidget(self._matching)
        self._widgets["seuil_matching"] = self._matching
        return g

    def _groupe_ligne(self, c: Config) -> QGroupBox:
        g = QGroupBox("Ligne de franchissement")
        v = QVBoxLayout(g)

        v.addWidget(QLabel("Épaisseur de la bande (px)"))
        self._epaisseur = QSpinBox()
        self._epaisseur.setRange(5, 100)
        self._epaisseur.setValue(c.epaisseur_bande)
        self._epaisseur.valueChanged.connect(self._emettre)
        v.addWidget(self._epaisseur)
        self._widgets["epaisseur_bande"] = self._epaisseur

        v.addWidget(QLabel("Sens de traversée"))
        self._sens = QComboBox()
        self._sens.addItems(["Avant → après", "Après → avant"])
        self._sens.setCurrentIndex(0 if c.sens == 1 else 1)
        self._sens.currentIndexChanged.connect(self._emettre)
        v.addWidget(self._sens)
        self._widgets["sens"] = self._sens

        v.addWidget(QLabel("Frames d'hystérésis"))
        self._hysteresis = QSpinBox()
        self._hysteresis.setRange(0, 10)
        self._hysteresis.setValue(c.frames_hysteresis)
        self._hysteresis.valueChanged.connect(self._emettre)
        v.addWidget(self._hysteresis)
        self._widgets["frames_hysteresis"] = self._hysteresis

        self._inverse = QCheckBox("Inverser le sens")
        self._inverse.toggled.connect(self._emettre)
        v.addWidget(self._inverse)
        return g

    def _groupe_profils(self) -> QGroupBox:
        g = QGroupBox("Profils")
        h = QVBoxLayout(g)
        self._nom_profil = QComboBox()
        self._nom_profil.setEditable(True)
        self._nom_profil.addItems(self._profils_disponibles())
        h.addWidget(self._nom_profil)
        self._btn_enregistrer = QPushButton("Enregistrer ce profil")
        self._btn_enregistrer.clicked.connect(self._on_enregistrer)
        h.addWidget(self._btn_enregistrer)
        self._btn_charger = QPushButton("Charger ce profil")
        self._btn_charger.clicked.connect(self._on_charger)
        h.addWidget(self._btn_charger)
        self._btn_defauts = QPushButton("Valeurs par défaut")
        self._btn_defauts.clicked.connect(self._on_defauts)
        h.addWidget(self._btn_defauts)
        return g

    # -- API publique ----------------------------------------------------

    def _modeles_disponibles(self) -> list[str]:
        racines = [pathlib.Path("."), pathlib.Path("modeles")]
        vus: list[str] = []
        for r in racines:
            if r.is_dir():
                vus.extend(sorted(p.name for p in r.glob("*.pt")))
        return vus or ["yolov8n-head.pt"]

    def _profils_disponibles(self) -> list[str]:
        if not Dossier_PROFILS.is_dir():
            return []
        return sorted(p.stem for p in Dossier_PROFILS.glob("*.json"))

    def lire(self) -> Config:
        c = Config()
        c.modele = self._modele.currentText().strip()
        c.seuil_confiance = round(self._seuil.value(), 3)
        c.taille_min_px = self._taille_min.value()
        c.taille_entree = int(self._taille_entree.currentText())
        c.frames_confirmation = self._confirmation.value()
        c.survie_max = self._survie.value()
        c.seuil_matching = round(self._matching.value(), 3)
        c.epaisseur_bande = self._epaisseur.value()
        c.sens = 1 if self._sens.currentIndex() == 0 else -1
        c.frames_hysteresis = self._hysteresis.value()
        return c

    def definir_seuil(self, valeur: float) -> None:
        self._seuil.setValue(valeur)

    def appliquer(self, config: Config) -> None:
        self._bloquer = True
        try:
            self._modele.setCurrentText(config.modele)
            self._seuil.setValue(config.seuil_confiance)
            self._taille_min.setValue(config.taille_min_px)
            self._taille_entree.setCurrentText(str(config.taille_entree))
            self._confirmation.setValue(config.frames_confirmation)
            self._survie.setValue(config.survie_max)
            self._matching.setValue(config.seuil_matching)
            self._epaisseur.setValue(config.epaisseur_bande)
            self._sens.setCurrentIndex(0 if config.sens == 1 else 1)
            self._hysteresis.setValue(config.frames_hysteresis)
        finally:
            self._bloquer = False
        self._emettre()

    def enregistrer_profil(
        self, nom: str, dossier: pathlib.Path | None = None
    ) -> pathlib.Path:
        d = dossier or Dossier_PROFILS
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{nom.strip()}.json"
        self.lire().vers_fichier(p)
        if p.stem not in [self._nom_profil.itemText(i)
                          for i in range(self._nom_profil.count())]:
            self._nom_profil.addItem(p.stem)
        return p

    def charger_profil(
        self, nom: str, dossier: pathlib.Path | None = None
    ) -> bool:
        d = dossier or Dossier_PROFILS
        p = d / f"{nom.strip()}.json"
        if not p.exists():
            return False
        try:
            self.appliquer(Config.depuis_fichier(p))
        except Exception as exc:  # noqa: BLE001
            log.warning("profil %s illisible : %s", nom, exc)
            return False
        return True

    # -- Slots -----------------------------------------------------------

    def _emettre(self, *_args) -> None:
        if getattr(self, "_bloquer", False):
            return
        self.config_modifiee.emit(self.lire())

    def _on_enregistrer(self) -> None:
        nom = self._nom_profil.currentText().strip()
        if nom:
            self.enregistrer_profil(nom)

    def _on_charger(self) -> None:
        self.charger_profil(self._nom_profil.currentText().strip())

    def _on_defauts(self) -> None:
        self.appliquer(Config.defauts())
```

- [ ] **Étape 4 : Lancer les tests, vérifier le passage**

```bash
python -m pytest tests/test_reglages.py -v
```

Attendu : TOUT PASSE (6 tests). Si PySide6 est absent : SKIP — installer
d'abord (`python -m pip install PySide6`).

- [ ] **Étape 5 : Commit**

```bash
git add interface/panneau_reglages.py tests/test_reglages.py
git commit -m "feat: panneau de réglages avec profils nommés"
```

---

## Tâche 10 : Fenêtre principale et boucle de lecture

**Fichiers :**
- Créer : `interface/app.py`
- Créer : `main.py`
- Créer : `tests/test_app.py`

**Interfaces :**
- Consomme : tout ce qui précède.
- Produit :
  - `interface.app.FenetrePrincipale()` — la fenêtre complète.
  - `main.py` — `python main.py` lance l'application.
- Fonctionnement : la boucle de lecture est pilotée par un `QTimer` (30 Hz
  max), qui lit une frame, la traite, puis applique la vitesse de présentation
  (0.25×, 0.5×, 1×, 2×, 4×, « max »). Le timer d'affichage est découplé du
  timer de traitement.

- [ ] **Étape 1 : Écrire les tests (doivent échouer)**

Créer `tests/test_app.py` :

```python

import pathlib

import pytest

pytest.importorskip("PySide6")

from interface.app import FenetrePrincipale  # noqa: E402


@pytest.fixture
def application():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_fenetre_se_construit(application):
    f = FenetrePrincipale()
    assert f.compteur_total == 0


def test_video_de_financement_present(application):
    f = FenetrePrincipale()
    assert f.video_finiment is not None


def test_compteur_se_met_a_jour(application):
    f = FenetrePrincipale()
    f.maj_compteurs(total=42, presents=7, frames=100)
    assert f.compteur_total == 42
    assert f.presents == 7


def test_charger_video_absente_signale_une_erreur(application, tmp_path):
    f = FenetrePrincipale()
    assert f.charger_video(str(tmp_path / "inexistant.mp4")) is False


def test_ligne_invalide_refusee(application):
    f = FenetrePrincipale()
    assert f.definir_ligne((10.0, 10.0), (10.0, 10.0)) is False


def test_export_sans_video_refuse(application, tmp_path):
    f = FenetrePrincipale()
    assert f.exporter(str(tmp_path)) is False
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_app.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'interface.app'`.

- [ ] **Étape 3 : Implémenter `interface/app.py`**

```python
"""Fenêtre principale : assemble le moteur et l'interface."""

from __future__ import annotations

import logging
import pathlib

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from compteur.compteur import Compteur
from compteur.config import Config
from compteur.detecteur import Detecteur, charger_modele
from compteur.ligne import Ligne
from compteur.rapport import chemins_par_defaut, ecrire_csv, ecrire_json, statistiques
from compteur.tracker import Tracker
from interface.overlay import dessiner
from interface.panneau_reglages import PanneauReglages
from interface.style import appliquer_style
from interface.widgets_video import WidgetVideo

log = logging.getLogger(__name__)

VITESSE_PAR_DEFAUT = 30  # images par seconde au plus


class FenetrePrincipale(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Compteur de manifestation")
        self.resize(1400, 850)

        self.config = Config.defauts()
        self.cap = None
        self.video_finiment: pathlib.Path | None = None
        self.detecteur = None
        self.tracker = None
        self.compteur = None
        self.ligne: Ligne | None = None
        self._points_ligne: list[tuple[float, float]] = []
        self._en_analyse = False
        self._flash = 0

        self._construire()
        self._connecter()
        self.compteur_total = 0
        self.presents = 0
        self.frames = 0

    # -- Construction ----------------------------------------------------

    def _construire(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        racine = QHBoxLayout(central)

        self.video = WidgetVideo()
        self.video.setMinimumWidth(860)
        racine.addWidget(self.video, stretch=3)

        colonne = QVBoxLayout()
        racine.addLayout(colonne, stretch=1)

        self.titre_compteur = QLabel("PERSONNES COMPTÉES")
        self.titre_compteur.setObjectName("compteur_titre")
        self.titre_compteur.setAlignment(Qt.AlignCenter)
        colonne.addWidget(self.titre_compteur)

        self.label_compteur = QLabel("0")
        self.label_compteur.setObjectName("compteur")
        self.label_compteur.setAlignment(Qt.AlignCenter)
        colonne.addWidget(self.label_compteur)

        self.label_details = QLabel("Présents : 0   •   Frames : 0")
        self.label_details.setObjectName("sous_titre")
        self.label_details.setAlignment(Qt.AlignCenter)
        colonne.addWidget(self.label_details)

        self.label_statut = QLabel("Chargez une vidéo pour commencer.")
        self.label_statut.setObjectName("sous_titre")
        self.label_statut.setWordWrap(True)
        colonne.addWidget(self.label_statut)

        self.panneau = PanneauReglages(self.config)
        colonne.addWidget(self.panneau, stretch=1)

        barre = QHBoxLayout()
        self.btn_video = QPushButton("Charger la vidéo")
        self.btn_ligne = QPushButton("Tracer la ligne")
        self.btn_lancer = QPushButton("Lancer")
        self.btn_lancer.setObjectName("primaire")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_export = QPushButton("Exporter")
        for b in (self.btn_video, self.btn_ligne, self.btn_lancer, self.btn_pause,
                  self.btn_stop, self.btn_export):
            barre.addWidget(b)
        colonne.addLayout(barre)

        self.choix_vitesse = QComboBox()
        self.choix_vitesse.addItems(["0.25×", "0.5×", "1×", "2×", "4×", "max"])
        self.choix_vitesse.setCurrentText("max")
        colonne.addWidget(self.choix_vitesse)

        self._timer = QTimer(self)
        self._timer.setInterval(33)

    def _connecter(self) -> None:
        self.btn_video.clicked.connect(self._on_charger_video)
        self.btn_ligne.clicked.connect(self._on_tracer_ligne)
        self.btn_lancer.clicked.connect(self.lancer)
        self.btn_pause.clicked.connect(self._basculer_pause)
        self.btn_stop.clicked.connect(self.arreter)
        self.btn_export.clicked.connect(self._on_exporter)
        self.video.clic.connect(self._on_clic_video)
        self.panneau.config_modifiee.connect(self._sur_config)
        self._timer.timeout.connect(self._tick)

    # -- Actions ---------------------------------------------------------

    def _on_charger_video(self) -> None:
        chemin, _ = QFileDialog.getOpenFileName(
            self, "Choisir une vidéo", "",
            "Vidéos (*.mp4 *.avi *.mkv *.mov);;Tous les fichiers (*)",
        )
        if chemin:
            self.charger_video(chemin)

    def charger_video(self, chemin: str) -> bool:
        if not pathlib.Path(chemin).exists():
            QMessageBox.warning(self, "Vidéo introuvable", f"Fichier absent :\n{chemin}")
            return False
        cap = cv2.VideoCapture(chemin)
        if not cap.isOpened():
            QMessageBox.warning(self, "Vidéo illisible", f"Impossible d'ouvrir :\n{chemin}")
            return False
        self.arreter()
        self.cap = cap
        self.video_finiment = pathlib.Path(chemin)
        self._reset_analyse()
        self._afficher_premiere_frame()
        self.btn_lancer.setEnabled(self.ligne is not None)
        self.label_statut.setText(
            f"{self.video_finiment.name} — "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, "
            f"{cap.get(cv2.CAP_PROP_FPS):.0f} i/s. Trace la ligne, puis lance."
        )
        return True

    def _reset_analyse(self) -> None:
        self.config = self.panneau.lire()
        self.detecteur = None
        self.tracker = None
        self.compteur = None
        self._points_ligne = []
        self.compteur_total = 0
        self.presents = 0
        self.frames = 0
        self.maj_compteurs(0, 0, 0)

    def _afficher_premiere_frame(self) -> None:
        if self.cap is None:
            return
        ok, img = self.cap.read()
        if ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video.definir_image(img)

    def _on_tracer_ligne(self) -> None:
        self._points_ligne = []
        self.ligne = None
        self.btn_lancer.setEnabled(False)
        self.label_statut.setText(
            "Cliquez deux points sur l'image pour tracer la ligne de franchissement."
        )

    def _on_clic_video(self, x: int, y: int) -> None:
        if self.cap is None or self._en_analyse:
            return
        if not self._points_ligne:
            self._points_ligne.append((float(x), float(y)))
            self.label_statut.setText("Premier point posé. Cliquez le second.")
            return
        self._points_ligne.append((float(x), float(y)))
        self.definir_ligne(*self._points_ligne)

    def definir_ligne(self, p1, p2) -> bool:
        try:
            self.ligne = Ligne(
                p1, p2,
                epaisseur=self.config.epaisseur_bande,
                sens=self.config.sens,
                hysteresis=self.config.frames_hysteresis,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Ligne invalide", str(exc))
            return False
        self.btn_lancer.setEnabled(True)
        self.label_statut.setText(
            "Ligne définie. Tu peux encore changer le sens et l'épaisseur dans les réglages."
        )
        return True

    def _sur_config(self, config: Config) -> None:
        self.config = config
        if self.ligne is not None:
            try:
                self.ligne = Ligne(
                    self.ligne.p1, self.ligne.p2,
                    epaisseur=config.epaisseur_bande,
                    sens=config.sens,
                    hysteresis=config.frames_hysteresis,
                )
            except ValueError:
                pass
        if self.detecteur is not None:
            self.detecteur.config = config
            if self._modele_change(config):
                self.detecteur = None
                self.tracker = None
        if self.tracker is not None:
            self.tracker.config = config

    @staticmethod
    def _modele_change(config: Config) -> bool:
        return bool(config.modele) and config.modele != getattr(
            FonetrePrincipale, "_modele_charge", ""
        )

    def lancer(self) -> None:
        if self.cap is None or self.ligne is None:
            return
        try:
            if self.detecteur is None:
                self.detecteur = Detecteur(charger_modele(self.config.modele), self.config)
                FonetrePrincipale._modele_charge = self.config.modele
            if self.tracker is None:
                self.tracker = Tracker(self.config)
            self.compteur = Compteur(self.config, self.detecteur, self.tracker)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Démarrage impossible", str(exc))
            return
        self.compteur.ajuster_ligne(self.ligne)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.compteur_total = 0
        self.presents = 0
        self.frames = 0
        self._en_analyse = True
        self._timer.start()
        self.btn_lancer.setEnabled(False)
        self.btn_ligne.setEnabled(False)
        self.label_statut.setText("Analyse en cours…")

    def _basculer_pause(self) -> None:
        if self._en_analyse:
            self._timer.stop()
            self._en_analyse = False
            self.btn_pause.setText("Reprendre")
        elif self.cap is not None:
            self._timer.start()
            self._en_analyse = True
            self.btn_pause.setText("Pause")

    def arreter(self) -> None:
        self._timer.stop()
        self._en_analyse = False
        self.btn_lancer.setEnabled(self.ligne is not None and self.cap is not None)
        self.btn_ligne.setEnabled(True)
        self.btn_pause.setText("Pause")
        if self.cap is not None:
            self.label_statut.setText("Arrêté.")

    def _tick(self) -> None:
        if self.cap is None or self.compteur is None:
            self._timer.stop()
            return
        ok, img = self.cap.read()
        if not ok:
            self.arreter()
            self._afficher_bilan()
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        ts = self.frames / fps
        resultat_frame = self.compteur.traiter_frame(img, self.frames, ts)
        self.frames += 1

        if resultat_frame.evenements:
            self._flash = 3
        if self._flash > 0:
            self._flash -= 1

        self.maj_compteurs(resultat_frame.total, resultat_frame.presents, self.frames)
        self.video.definir_image(
            dessiner(resultat_frame.image, resultat_frame, self.ligne, flash=self._flash > 0)
        )
        self._appliquer_vitesse(fps)

    def _appliquer_vitesse(self, fps: float) -> None:
        """Ralentit l'affichage — n'affecte jamais le traitement."""
        texte = self.choix_vitesse.currentText()
        if texte == "max":
            self._timer.setInterval(0)
            return
        multiplicateur = float(texte.replace("×", ""))
        intervalle = int((1.0 / (fps * multiplicateur)) * 1000)
        self._timer.setInterval(max(0, intervalle))

    def maj_compteurs(self, total: int, presents: int, frames: int) -> None:
        self.compteur_total = total
        self.presents = presents
        self.frames = frames
        self.label_compteur.setText(str(total))
        self.label_details.setText(f"Présents : {presents}   •   Frames : {frames}")

    def _afficher_bilan(self) -> None:
        if self.compteur is None:
            return
        resultat = self.compteur.resultat(self.config.modele, self.frames, self.frames / 30)
        s = statistiques(resultat)
        self.label_statut.setText(
            f"Terminé — {s['total']} personnes, "
            f"{s['personnes_par_minute']}/min, max {s['presents_max']} présents. "
            "Pense à exporter."
        )

    def _on_exporter(self) -> None:
        dossier = QFileDialog.getExistingDirectory(self, "Dossier de sortie", "sortie")
        if dossier:
            self.exporter(dossier)

    def exporter(self, dossier: str) -> bool:
        if self.compteur is None or self.video_finiment is None:
            QMessageBox.information(self, "Rien à exporter", "Lance d'abord une analyse.")
            return False
        resultat = self.compteur.resultat(
            self.config.modele, self.frames, self.frames / 30
        )
        chemins = chemins_par_defaut(str(self.video_finiment), dossier)
        ecrire_csv(resultat, chemins["csv"])
        ecrire_json(resultat, chemins["json"])
        self.label_statut.setText(
            f"Exporté :\n{chemins['csv'].name}\n{chemins['json'].name}\n(dossier {dossier})"
        )
        return True
```

- [ ] **Étape 4 : Créer `main.py`**

```python
"""Point d'entrée : python main.py"""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from interface.app import FenetrePrincipale
from interface.style import appliquer_style


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    appliquer_style(app)
    fenetre = FenetrePrincipale()
    fenetre.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Étape 5 : Lancer les tests, vérifier le passage**

```bash
python -m pytest tests/test_app.py -v
```

Attendu : TOUT PASSE (6 tests).

- [ ] **Étape 6 : Vérifier que la fenêtre s'ouvre vraiment**

```bash
python -c "from PySide6.QtWidgets import QApplication; from interface.app import FenetrePrincipale; a=QApplication([]); f=FenetrePrincipale(); print('fenetre ok', f.windowTitle())"
```

Attendu : `fenetre ok Compteur de manifestation`.

- [ ] **Étape 7 : Lancer l'application et vérifier visuellement**

```bash
python main.py
```

Contrôler à l'œil : la fenêtre s'ouvre sombre, le compteur est lisible, le
panneau de réglages est à droite, les 6 boutons en bas. **Ne pas tenter de
charger une vidéo à ce stade** — c'est vérifié à la tâche suivante.

- [ ] **Étape 8 : Lancer toute la suite**

```bash
python -m pytest tests/ -v
```

Attendu : TOUT PASSE (aucun test ignoré hors PySide6 optionnel).

- [ ] **Étape 9 : Commit**

```bash
git add interface/app.py main.py tests/test_app.py
git commit -m "feat: fenêtre principale, boucle de lecture, contrôle de vitesse"
```

---

## Tâche 11 : Mode audit — correction manuelle et taux d'erreur mesuré

**Fichiers :**
- Créer : `compteur/audit.py`
- Créer : `interface/audit.py`
- Créer : `tests/test_audit.py`

**Interfaces :**
- Consomme : `Resultat`, `Evenement` (T1).
- Produit :
  - `compteur.audit.SaisieAudit(evenements: list[Evenement])`
    - `.corriger_faux_positif(frame: int) -> None`
    - `.corriger_faux_negatif(frame: int, x: float, y: float) -> None`
    - `.total_corrige() -> int`
    - `.ecart() -> int` — `total_corrige - total_initial`
    - `.taux_erreur() -> float` — `abs(ecart) / max(1, total_corrige)`, en [0, 1]
    - `.vers_dict() -> dict` — pour export
  - `interface.audit.PanneauAudit(...)` — boutons de correction branchés sur la
    frame courante.

- [ ] **Étape 1 : Écrire les tests (doivent échouer)**

Créer `tests/test_audit.py` :

```python
from compteur.audit import SaisieAudit
from compteur.types import Evenement


def evenements():
    return [
        Evenement(frame=10, timestamp_s=0.3, x=100.0, y=50.0, track_id=1),
        Evenement(frame=20, timestamp_s=0.6, x=110.0, y=50.0, track_id=2),
    ]


def test_aucune_correction_total_inchange():
    s = SaisieAudit(evenements())
    assert s.total_corrige() == 2
    assert s.ecart() == 0
    assert s.taux_erreur() == 0.0


def test_faux_positif_reduit_le_total():
    s = SaisieAudit(evenements())
    s.corriger_faux_positif(frame=20)
    assert s.total_corrige() == 1
    assert s.ecart() == -1
    assert s.taux_erreur() == 0.5


def test_faux_negatif_ajoute_au_total():
    s = SaisieAudit(evenements())
    s.corriger_faux_negatif(frame=99, x=150.0, y=50.0)
    assert s.total_corrige() == 3
    assert s.ecart() == 1


def test_double_correction_faux_positif_ignoree():
    s = SaisieAudit(evenements())
    s.corriger_faux_positif(frame=20)
    s.corriger_faux_positif(frame=20)
    assert s.total_corrige() == 1


def test_correction_de_frame_inconnue_ignoree():
    s = SaisieAudit(evenements())
    s.corriger_faux_positif(frame=9999)
    assert s.total_corrige() == 2


def test_export_contient_les_corrections():
    s = SaisieAudit(evenements())
    s.corriger_faux_positif(frame=10)
    d = s.vers_dict()
    assert d["total_initial"] == 2
    assert d["total_corrige"] == 1
    assert d["taux_erreur"] == 0.5
    assert len(d["corrections"]) == 1
```

- [ ] **Étape 2 : Lancer, vérifier l'échec**

```bash
python -m pytest tests/test_audit.py -v
```

Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'compteur.audit'`.

- [ ] **Étape 3 : Implémenter `compteur/audit.py`**

```python
"""Audit : corrections manuelles du décompte et taux d'erreur mesuré.

Ce module ne prétend jamais calculer une erreur « automatiquement ». Il ne fait
que mesurer l'écart entre ce que la machine a compté et ce que l'utilisateur a
validé à l'œil. Le taux d'erreur n'est donc valide que sur la portion
effectivement vérifiée.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Evenement


@dataclass
class Correction:
    frame: int
    type: str  # "faux_positif" ou "faux_negatif"
    x: float = 0.0
    y: float = 0.0


@dataclass
class SaisieAudit:
    evenements: list[Evenement]
    corrections: list[Correction] = field(default_factory=list)

    @property
    def total_initial(self) -> int:
        return len(self.evenements)

    def _est_deja_corrige(self, frame: int, type_: str) -> bool:
        return any(c.frame == frame and c.type == type_ for c in self.corrections)

    def corriger_faux_positif(self, frame: int) -> None:
        """Retire du total un comptage que l'utilisateur juge incorrect."""
        if self._est_deja_corrige(frame, "faux_positif"):
            return
        if not any(ev.frame == frame for ev in self.evenements):
            return
        self.corrections.append(Correction(frame=frame, type="faux_positif"))

    def corriger_faux_negatif(self, frame: int, x: float, y: float) -> None:
        """Ajoute au total une personne que la machine a ratée."""
        if self._est_deja_corrige(frame, "faux_negatif"):
            return
        self.corrections.append(
            Correction(frame=frame, type="faux_negatif", x=x, y=y)
        )

    def total_corrige(self) -> int:
        faux_positifs = sum(1 for c in self.corrections if c.type == "faux_positif")
        faux_negatifs = sum(1 for c in self.corrections if c.type == "faux_negatif")
        return self.total_initial - faux_positifs + faux_negatifs

    def ecart(self) -> int:
        return self.total_corrige() - self.total_initial

    def taux_erreur(self) -> float:
        total = self.total_corrige()
        if total <= 0:
            return 0.0
        return abs(self.ecart()) / total

    def vers_dict(self) -> dict:
        return {
            "total_initial": self.total_initial,
            "total_corrige": self.total_corrige(),
            "ecart": self.ecart(),
            "taux_erreur": round(self.taux_erreur(), 4),
            "corrections": [
                {"frame": c.frame, "type": c.type, "x": c.x, "y": c.y}
                for c in self.corrections
            ],
        }
```

- [ ] **Étape 4 : Lancer les tests, vérifier le passage**

```bash
python -m pytest tests/test_audit.py -v
```

Attendu : TOUT PASSE (6 tests).

- [ ] **Étape 5 : Créer `interface/audit.py`**

```python
"""Panneau de correction manuelle, branché sur la frame courante."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from compteur.audit import SaisieAudit


class PanneauAudit(QWidget):
    """Deux boutons et un relevé d'écart. N'invente aucun chiffre."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.saisie: SaisieAudit | None = None
        self._frame_courante = 0
        v = QVBoxLayout(self)

        g = QGroupBox("Audit")
        h = QVBoxLayout(g)

        self.bouton_rater = QPushButton("Cette personne a été ratée")
        self.bouton_rater.clicked.connect(self._corriger_faux_negatif)
        h.addWidget(self.bouton_rater)

        self.bouton_bruit = QPushButton("Ce comptage était un faux positif")
        self.bouton_bruit.clicked.connect(self._corriger_faux_positif)
        h.addWidget(self.bouton_bruit)

        self.bilan = QLabel("Vérifie la vidéo, corrige ce qui est faux.")
        self.bilan.setObjectName("sous_titre")
        self.bilan.setWordWrap(True)
        h.addWidget(self.bilan)
        v.addWidget(g)

    def definir_saisie(self, saisie: SaisieAudit) -> None:
        self.saisie = saisie
        self._maj_bilan()

    def definir_frame(self, frame: int) -> None:
        self._frame_courante = frame

    def _corriger_faux_negatif(self) -> None:
        if self.saisie is None:
            return
        self.saisie.corriger_faux_negatif(self._frame_courante, 0.0, 0.0)
        self._maj_bilan()

    def _corriger_faux_positif(self) -> None:
        if self.saisie is None:
            return
        self.saisie.corriger_faux_positif(self._frame_courante)
        self._maj_bilan()

    def _maj_bilan(self) -> None:
        if self.saisie is None:
            self.bilan.setText("Vérifie la vidéo, corrige ce qui est faux.")
            return
        d = self.saisie.vers_dict()
        self.bilan.setText(
            f"Compté : {d['total_initial']}   •   "
            f"Corrigé : {d['total_corrige']}   •   "
            f"Écart : {d['ecart']:+d}   •   "
            f"Erreur mesurée : {d['taux_erreur'] * 100:.1f} %\n"
            "Valide uniquement sur la portion vérifiée."
        )
```

- [ ] **Étape 6 : Brancher l'audit dans la fenêtre principale**

Dans `interface/app.py`, ajouter l'import et l'insertion :

```python
from interface.audit import PanneauAudit
```

Après `colonne.addWidget(self.panneau, stretch=1)`, ajouter :

```python
self.audit = PanneauAudit()
colonne.addWidget(self.audit)
```

Dans `lancer()`, après la création du `Compteur`, initialiser l'audit :

```python
from compteur.audit import SaisieAudit
self.audit.definir_saisie(SaisieAudit(self.compteur.evenements))
```

Dans `_tick()`, mettre à jour la frame courante :

```python
self.audit.definir_frame(self.frames)
```

Dans `_afficher_bilan()`, afficher l'audit si actif :

```python
if self.audit.saisie is not None:
    d = self.audit.saisie.vers_dict()
    self.label_statut.setText(
        f"Terminé — {d['total_corrige']} validés "
        f"({d['total_initial']} comptés, écart {d['ecart']:+d}, "
        f"erreur mesurée {d['taux_erreur'] * 100:.1f} % sur la portion vérifiée)."
    )
```

- [ ] **Étape 7 : Lancer toute la suite**

```bash
python -m pytest tests/ -v
```

Attendu : TOUT PASSE.

- [ ] **Étape 8 : Vérifier visuellement**

```bash
python main.py
```

Contrôler que le panneau « Audit » apparaît sous les réglages, avec ses deux
boutons et la mention « Valide uniquement sur la portion vérifiée ».

- [ ] **Étape 9 : Commit**

```bash
git add compteur/audit.py interface/audit.py interface/app.py tests/test_audit.py
git commit -m "feat: mode audit — correction manuelle et taux d'erreur mesuré"
```

---

## Tâche 12 : Packaging `.exe` et documentation

**Fichiers :**
- Créer : `crowd-counter.spec`
- Créer : `README.md`
- Créer : `requirements-dev.txt`

**Interfaces :**
- Consomme : tout le projet.
- Produit : `dist/CompteurManifestation.exe` lançable sur une machine Windows
  sans Python installé.

- [ ] **Étape 1 : Créer `requirements-dev.txt`**

```
-r requirements.txt
pyinstaller>=6.10
```

- [ ] **Étape 2 : Créer `crowd-counter.spec`**

```python
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller : construit un .exe autonome."""

import os

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[("config/default.json", "config")],
    hiddenimports=[
        "ultralytics",
        "ultralytics.yolo",
        "supervision",
        "cv2",
        "scipy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pandas", "notebook"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="CompteurManifestation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

- [ ] **Étape 3 : Construire l'exécutable**

```bash
python -m PyInstaller --clean crowd-counter.spec
```

Attendu : `dist/CompteurManifestation.exe` existe. La première construction
prend plusieurs minutes (torch est volumineux).

- [ ] **Étape 4 : Vérifier l'exécutable**

Lancer `dist\CompteurManifestation.exe` depuis l'explorateur Windows.
Contrôler : la fenêtre s'ouvre, le thème sombre s'applique, aucune erreur
console. Placer un `yolov8n-head.pt` dans le dossier de l'exécutable et vérifier
qu'il apparaît dans la liste des modèles du panneau de réglages.

- [ ] **Étape 5 : Écrire le `README.md`**

````markdown
# Compteur de manifestation

Compte les personnes franchissant une ligne virtuelle dans une vidéo de
caméra fixe. Moteur de comptage sans interface, fenêtre PySide6 par-dessus.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

## Utilisation

1. **Charger la vidéo** — bouton *Charger la vidéo*.
2. **Tracer la ligne** — bouton *Tracer la ligne*, puis deux clics sur l'image.
3. **Régler le sens** — dans *Ligne de franchissement*, choisissez la direction
   qui vous intéresse.
4. **Ajuster la sensibilité** — le paramètre qui compte le plus est
   *Frames de confirmation* dans le groupe *Tracker* : 1 compte tout de suite
   (sensible aux faux positifs), 5 ne valide qu'une personne vue plusieurs
   frames de suite.
5. **Lancer** — le compteur s'actualise, la ligne clignote à chaque passage.
6. **Exporter** — génère un CSV (un événement par ligne) et un JSON (total,
   statistiques, config complète).

## Mode audit

Pendant la relecture, le panneau *Audit* permet de signaler les erreurs à
l'œil. Le taux d'erreur affiché ne vaut que pour la portion que vous avez
effectivement vérifiée — c'est une mesure, pas une estimation automatique.

## Comparer deux modèles

```bash
python tools/comparer_modeles.py --video "D:/videos/manif.mp4" \
    --modeles yolov8n-head.pt yolov8n.pt \
    --ligne 556 90 588 654 --sens 1
```

Affiche un tableau comparatif et écrit `sortie/comparaison.json`.

## Construction de l'exécutable

```bash
pip install -r requirements-dev.txt
python -m PyInstaller --clean crowd-counter.spec
```

Résultat : `dist/CompteurManifestation.exe`. Placer les modèles `.pt` à côté
de l'exécutable.

## Architecture

```
compteur/    moteur pur, sans dépendance graphique — testable sans écran
interface/   fenêtre PySide6
tools/      Scripts en ligne de commande
tests/       pytest
```

Le point clé : `compteur/compteur.py::analyser_video` est une fonction
`vidéo + config -> résultat`. L'interface l'appelle, les tests l'appellent, et
une éventuelle API web pourra l'appeler plus tard sans rien réécrire.

## Limites connues

- Le taux d'erreur n'est jamais calculé automatiquement : il n'existe pas de
  vérité terrain. Le mode audit mesure l'écart sur ce que vous vérifiez.
- Compteur conçu pour un flux peu occlusé (rue, passage espacé). Les foules
  compactes où les corps se chevauchent ne sont pas le cas d'usage visé.
- Le modèle de détection est un paramètre : n'importe quel `.pt` Ultralytics
  est accepté.

## Licence

À définir par l'auteur.
````

- [ ] **Étape 6 : Vérifier que le README est cohérent avec le code**

```bash
python tools/comparer_modeles.py --help
```

Attendu : l'aide s'affiche sans erreur, les options du README existent.

- [ ] **Étape 7 : Commit**

```bash
git add README.md crowd-counter.spec requirements-dev.txt .gitignore
git commit -m "docs: README, packaging PyInstaller, dépendances de dev"
```

- [ ] **Étape 8 : Pousser**

```bash
git push
```

---

## Vérification finale

- [ ] **Étape 1 : Suite de tests complète au vert**

```bash
python -m pytest tests/ -v
```

- [ ] **Étape 2 : Aucun import graphique dans le moteur**

```bash
python -m pytest tests/test_isolation_ui.py -v
```

- [ ] **Étape 3 : Comparaison des modèles sur une vraie vidéo**

```bash
python tools/comparer_modeles.py --video "<ta vidéo>" \
    --modeles <modèle_tête.pt> <modèle_corps.pt> \
    --ligne x1 y1 x2 y2 --sens 1
```

Décision à consigner dans le README : quel modèle a donné le compte le plus
plausible à l'œil.

- [ ] **Étape 4 : Exécutable lancé une fois**

```bash
dist\CompteurManifestation.exe
```

- [ ] **Étape 5 : Pousser la branche**

```bash
git push
```

---

## Ce que ce plan ne fait pas

- Pas d'interface web ni d'API (volontairement : le moteur est prêt, la
  décision est reportée).
- Pas d'entraînement de modèle.
- Pas de reconnaissance faciale ni d'identification.
- Pas de densimétrie pour les foules compactes.
- Pas de support multi-caméras.
- Pas de calcul automatique du taux d'erreur (impossible sans vérité terrain).
