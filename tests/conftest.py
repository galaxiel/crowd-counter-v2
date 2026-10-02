"""Configuration commune à la suite de tests.

La suite doit tourner depuis n'importe quel répertoire de travail : les scripts
lancés depuis ailleurs (tâche 7) et PyInstaller (tâche 12, depuis `dist/`)
n'ont pas le dépôt comme dossier courant. On insère donc la racine du dépôt
dans `sys.path` et on ancre les chemins sur l'emplacement de ce fichier plutôt
que sur le CWD.
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))
