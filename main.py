"""Point d'entrée : ``python main.py``.

Rien de lourd ici. Le thème est posé avant la construction de la fenêtre, et
la fenêtre ne charge AUCUN modèle à l'ouverture (voir `interface.app`) : le
premier écran doit s'afficher immédiatement, même sans poids sur le disque.

**Un bloc avant tout le reste, uniquement pour la version 2.**
`compteur.telechargement.ajouter_au_sys_path()` doit passer AVANT le premier
`import torch` du processus. torch n'est pas dans l'exécutable v2 : il est
téléchargé au premier lancement et posé dans `%LOCALAPPDATA%\
CompteurManifestation\torch\site`, qu'il faut ajouter à `sys.path`. Cet import
est donc ici, à la toute première ligne utile, et pas dans `interface.app` —
où PyInstaller ne peut plus voir le chemin d'import au moment de son analyse.

**Depuis les sources, ce bloc ne fait rien de visible.** `dossier_cache()`
pointe vers un dossier qui n'existe probablement pas, et `ajouter_au_sys_path`
renvoie `False`. Le comportement est donc identique à celui d'avant pour
quiconque lance `python main.py` avec un torch déjà installé dans son
environnement.

La fenêtre principale est construite AVEC l'état de torch. Si le
téléchargement échoue, `interface.demarrage` renvoie un état « absent » et la
fenêtre s'ouvre quand même : c'est la règle absolue de la v2.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

#: Doit précéder l'import de `interface.app`, qui touche torch via
#: `compteur.detecteur`. Sans torch sur le `sys.path`, l'analyse statique de
#: PyInstaller échoue à suivre jusqu'ici et l'exécutable démarre sans pouvoir
#: jamais compter.
from compteur import telechargement

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Reconstruction du chemin d'import du torch téléchargé. Doit précéder
    # QApplication : c'est le tout premier `sys.path` du processus gelé, et le
    # seul endroit où torch peut encore y être posé.
    telechargement.ajouter_au_sys_path()

    app = QApplication(sys.argv)

    # ⚠ `interface.app` N'EST PAS IMPORTÉ ICI, et c'est délibéré.
    #
    # `interface/app.py` fait `import cv2` au chargement du module, et `cv2`
    # fait `import numpy` au sien. Or `numpy` n'est pas dans NOTRE paquet :
    # c'est le venv qui l'installe. Importer `interface.app` avant `preparer`
    # revenait donc à exiger le venv avant d'avoir eu l'occasion de le
    # construire — un paquet incapable de démarrer pour pouvoir démarrer.
    # C'est exactement le traceback du binaire gelé (`main.py` ligne 55).
    #
    # L'ordre ci-dessous est donc STRICT, et il n'est pas décoratif :
    #   1. `ajouter_au_sys_path`  — pose le venv s'il existe déjà ;
    #   2. QApplication + style   — PySide6 seul, toujours disponible ;
    #   3. `preparer`              — l'écran d'installation, toujours
    #                                atteignable, et qui installe le venv ;
    #   4. `interface.app`         — seulement maintenant, quand le venv est
    #                                interrogeable.
    #
    # L'import reste dans cette fonction, et non au niveau du module, pour
    # que PyInstaller continue de voir le chemin d'import à analyser.
    from interface.demarrage import preparer
    from interface.style import appliquer_style

    appliquer_style(app)

    # Téléchargement au premier lancement, avec fenêtre de progression.
    # `preparer` ne lève jamais : au pire il renvoie un état « absent », et
    # l'application démarre sans moteur de calcul.
    etat_torch = preparer(None)

    # Le venv est maintenant interrogeable : le moteur peut être importé.
    from interface.app import FenetrePrincipale

    fenetre = FenetrePrincipale()
    fenetre.definir_etat_torch(etat_torch)
    fenetre.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())