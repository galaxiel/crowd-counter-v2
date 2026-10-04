"""Interface graphique (PySide6).

Ce paquet est le SEUL endroit du projet autorisé à importer un toolkit
graphique : `tests/test_isolation_ui.py` vérifie que `compteur/` n'en importe
jamais, ce qui garde le moteur utilisable sans écran (tests, scripts de la
tâche 7, future API web).

Rien n'est importé ici à l'import du paquet : `import interface` reste sans
coût et sans dépendance, et PySide6 n'est chargé que par le sous-module
réellement utilisé.
"""

__all__ = ["overlay", "style", "widgets_video"]