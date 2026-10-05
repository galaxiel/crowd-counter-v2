"""Moteur de comptage — aucune dépendance graphique.

Ce paquet est volontairement utilisable sans écran : les tests, la comparaison
de modèles et une future API web l'importent tous directement.
"""

#: Numéro de version du logiciel (semantic versioning).
#:
#: Règle de numérotation, appliquée strictement :
#:
#: - **majeur** — la façon dont le logiciel s'installe, ou son API, change.
#:   La v2 est un 2.0.0 et pas un 1.1.0 parce que torch n'est plus embarqué :
#:   il se télécharge au premier lancement. Ce n'est pas du contenu ajouté,
#:   c'est un changement du contrat d'installation — un dossier `dist/` de v1
#:   et un de v2 ne sont pas interchangeables.
#: - **mineur** — une fonctionnalité, rétrocompatible.
#: - **correctif** — une correction de bug, sans changement de comportement.
#:
#: Source de vérité UNIQUE : `interface/app.py` importe cette constante et ne
#: redéfinit jamais la chaîne. Deux numéros de version dans le dépôt, c'est
#: deux numéros de version qui divergent au premier tag.
VERSION = "2.0.0"

__all__ = ["VERSION", "types", "config"]
