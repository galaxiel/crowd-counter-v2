"""Point d'entrée : ``python main.py``.

Rien de lourd ici. Le thème est posé avant la construction de la fenêtre, et
la fenêtre ne charge AUCUN modèle à l'ouverture (voir `interface.app`) : le
premier écran doit s'afficher immédiatement, même sans poids sur le disque.
"""

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
