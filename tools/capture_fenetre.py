"""Rendu de la fenêtre réelle en PNG, pour contrôle visuel.

Ne fait PAS partie de la suite de tests : c'est un outil de vérification
lancé à la main, qui produit une image de la fenêtre telle que Qt la peint
(thème sombre compris) pour qu'on puisse la regarder.

Deux détails qui ont coûté une itération, et qu'il ne faut pas défaire :

- **Pas de plateforme `offscreen`.** Elle n'a aucune police : tout le texte
  ressort en carrés vides, et une capture illisible ne prouve rien. On rend
  donc avec le moteur de rendu Windows, sans jamais afficher la fenêtre à
  l'écran — `grab()` peint le widget sans le montrer.
- **Une vraie vidéo.** `_rafraichir_affichage` relit la frame courante dans
  la capture ; sans `VideoCapture` ouverte, l'aperçu de la ligne n'a aucune
  image sur laquelle se dessiner et la capture montre un rectangle vide.

Usage :
    python tools/capture_fenetre.py [chemin.png]
"""

from __future__ import annotations

import pathlib
import sys

# Importer le dépôt comme racine, comme le fait `tests/conftest.py`.
RACINE = pathlib.Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

# Volontairement NI `QT_QPA_PLATFORM=offscreen` : voir la note de module.
import os  # noqa: E402

os.environ.pop("QT_QPA_PLATFORM", None)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from interface.app import FenetrePrincipale  # noqa: E402
from interface.style import appliquer_style  # noqa: E402


def _video_de_demonstration(chemin: pathlib.Path) -> pathlib.Path:
    """Une vidéo 1280x720 avec un fond de scène et trois silhouettes.

    Le fond n'est pas uniforme : un aplat gris ne permettrait pas de juger du
    contraste du texte ni de la lisibilité de la bande translucide.
    """
    auteur = cv2.VideoWriter(
        str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1280, 720)
    )
    for i in range(10):
        img = np.full((720, 1280, 3), 90, dtype=np.uint8)
        # Bande d'horizon, comme un fond de rue.
        img[380:400, :] = (70, 70, 70)
        for x in (500, 760, 900):  # trois silhouettes qui marchent vers la droite
            img[300 + i : 470 + i, x : x + 26] = (40, 40, 40)
        auteur.write(img)
    auteur.release()
    return chemin


def capture(chemin: pathlib.Path) -> pathlib.Path:
    app = QApplication.instance() or QApplication([])
    appliquer_style(app)
    f = FenetrePrincipale()
    f.resize(1500, 900)

    f.charger_video(str(_video_de_demonstration(chemin.with_suffix(".mp4"))))
    f.maj_compteurs(total=1234, presents=57, frames=812)

    # Tracé de la ligne en deux clics, sur la verticale du milieu.
    f._on_tracer_ligne()
    f._on_clic_video(640, 40)
    f._on_clic_video(640, 680)

    # `grab()` peint le widget sans l'afficher : la fenêtre n'apparaît jamais
    # à l'écran, mais le rendu est celui du vrai moteur de dessin.
    f.grab().save(str(chemin))
    return chemin


if __name__ == "__main__":
    sortie = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else RACINE / "capture_fenetre.png"
    )
    print(capture(sortie))
