"""Sonde de diagnostic — s'exécute DANS l'exécutable gelé.

Ce fichier est un `runtime_hooks` de `crowd-counter.spec` : le bootloader
l'exécute avant `main.py`, dans le processus gelé authentique. C'est la seule
façon d'observer `sys.frozen` et `sys._MEIPASS` **réellement** posés, et non
simulés par un monkeypatch.

**Sans effet en usage normal.** Tout le corps est sous
`if not os.environ.get("COMPTEUR_PROBE"): return`. Sans la variable
d'environnement, ce fichier ne fait rien : ni écriture, ni journal, ni
import de `compteur`. Un runtime hook s'exécute à chaque démarrage, et un
logiciel de comptage ne doit pas payer une sonde à chaque lancement.

Ce qu'il prouve, et pourquoi c'est nécessaire : un repli silencieux de
`Config.defauts()` ne lève rien. Il journalise un `WARNING` que
l'utilisateur ne voit pas (fenêtre console fermée, ou non ouverte), et
l'application démarre **quand même**, avec les mêmes valeurs — puisque
`config/default.json` et le dataclass `Config` coincident par construction,
ce qu'un test vérifie par ailleurs. Sans cette sonde, « la fenêtre
s'ouvre » ne prouve donc pas que le fichier embarqué a été lu.

Sortie : JSON, chemin indiqué par `COMPTEUR_PROBE_RAPPORT`. Voir
`tools/verifier_config_gelee.py`.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys


def _ecrire_rapport() -> None:
    rapport = os.environ.get("COMPTEUR_PROBE_RAPPORT")
    if not rapport:
        return

    meipass = getattr(sys, "_MEIPASS", None)
    donnees = {
        "frozen": bool(getattr(sys, "frozen", False)),
        "meipass": meipass,
        "executable": sys.executable,
        "python": sys.version.split()[0],
    }
    try:
        # Import réel : c'est le même module que celui de l'application.
        from compteur.config import (
            Config,
            _racine,
            chemin_defaut_config,
        )

        donnees["racine"] = str(_racine())
        donnees["config_resolu"] = str(chemin_defaut_config())
        donnees["config_attendu"] = str(
            pathlib.Path(meipass or ".").resolve() / "config" / "default.json"
        )
        donnees["config_lisible"] = chemin_defaut_config().is_file()
        donnees["repli_silencieux"] = not donnees["config_lisible"]

        c = Config.defauts()
        donnees["modele"] = c.modele
        donnees["taille_entree"] = c.taille_entree
        donnees["fenetre_lissage"] = c.fenetre_lissage
        donnees["seuil_matching"] = c.seuil_matching
    except Exception as exc:  # noqa: BLE001 — la sonde ne doit rien casser
        donnees["erreur"] = f"{type(exc).__name__}: {exc}"

    # Écriture atomique : l'outil qui attend le fichier ne doit jamais lire
    # un rapport à moitié écrit.
    cible = pathlib.Path(rapport)
    cible.parent.mkdir(parents=True, exist_ok=True)
    temporaire = cible.with_suffix(".tmp")
    temporaire.write_text(
        json.dumps(donnees, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporaire.replace(cible)


# Garde principale : rien à faire si l'utilisateur n'a rien demandé.
if os.environ.get("COMPTEUR_PROBE"):
    _ecrire_rapport()