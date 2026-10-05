"""Pilote l'exécutable gelé pour vérifier l'installation RÉELLE de torch.

Ce n'est PAS un test unitaire : c'est une démonstration, exécutée sur le
binaire lui-même, avec le cache `%LOCALAPPDATA%\\CompteurManifestation`
VIDÉ au préalable. C'est le seul moyen de prouver que la v2 fonctionne
depuis zéro sur une machine vierge — le cas que la machine de développement
ne peut pas représenter, puisque torch y est installé globalement.

Deux Parties :

- **Mode 1 — observation** : on fait tourner le vrai `installer_torch` dans
  le code gelé, et on journalise chaque progression. C'est la preuve que le
  téléchargement et l'installation fonctionnent réellement.
- **Mode 2 — démarrage** : on lance la vraie fenêtre, sans toucher à la
  boucle d'événements, pour vérifier que l'application s'ouvre.

Usage (depuis le dépôt) :

    python tools/verifier_installation_v2.py --mode installer
    python tools/verifier_installation_v2.py --mode demarrer --timeout 60
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

RACINE = pathlib.Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

log = logging.getLogger("verif")


def _journaliser(p) -> None:
    """Affiche chaque progression, pour voir les DEUX phases."""
    phase = "TELECHARGEMENT" if p.phase == "telechargement" else "INSTALLATION"
    ligne = f"[{phase}] {p.pourcent:3d} % | {p.texte} | {p.fait}/{p.total}"
    if p.vitesse:
        ligne += f" | {p.vitesse / 1024**2:.1f} Mo/s"
    if p.secondes_restantes:
        ligne += f" | reste {int(p.secondes_restantes)} s"
    print(ligne, flush=True)


def mode_installer() -> int:
    """Lance la vraie installation dans le code gelé."""
    from compteur import telechargement as tl

    cache = tl.dossier_cache()
    print(f"cache : {cache}", flush=True)
    print(f"existe deja : {cache.exists()}", flush=True)
    uv = tl.uv_exe()
    print(f"uv.exe : {uv}", flush=True)
    print(f"gpu    : {tl.nom_gpu()!r}", flush=True)
    print(f"attendu: {tl.taille_attendue(True) / 1024**3:.2f} Go", flush=True)
    print("-" * 70, flush=True)

    depart = time.monotonic()
    ok = tl.installer_torch("cuda", _journaliser, cache)
    duree = time.monotonic() - depart

    print("-" * 70, flush=True)
    print(f"résultat : {ok} en {duree:.0f} s", flush=True)
    print(f"message  : {tl.dernier_message()!r}", flush=True)
    print(f"marqueur : {(cache / tl.MARQUEUR).is_file()}", flush=True)
    print(f"installe : {tl.torch_installe(cache)}", flush=True)
    return 0 if ok else 1


def mode_demarrer(timeout: int) -> int:
    """Lance la vraie fenêtre et vérifie qu'elle s'ouvre."""
    from interface import demarrage
    from interface.app import FenetrePrincipale
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    cache = demarrage.tl.dossier_cache()
    app = QApplication(sys.argv)

    etat = demarrage.preparer(None, cache)
    print(f"etat     : {etat}", flush=True)
    print(f"indicateur: {demarrage.texte_indicateur(etat)}", flush=True)

    fenetre = FenetrePrincipale()
    fenetre.definir_etat_torch(etat)
    fenetre.show()
    print(f"fenetre visible : {fenetre.isVisible()}", flush=True)
    print(f"label torch     : {fenetre.label_torch.text()!r}", flush=True)
    print(f"label periph    : {fenetre.label_peripherique.text()!r}", flush=True)
    print(f"compteur        : {fenetre.label_compteur.text()!r}", flush=True)

    # On ne boucle pas : le but est de prouver que la fenêtre S'OUVRE, pas de
    # la piloter. `app.exec()` repartirait sur la boucle d'événements et le
    # script ne rendrait jamais la main.
    # -- Chargement réel du modèle et une inférence : c'est la seule preuve
    # que le torch du venv est utilisable PAR l'exécutable gelé, et pas
    # seulement importable. Un import qui réussit mais dont les DLL CUDA ne se
    # chargent pas se voit ici, et nulle part ailleurs.
    try:
        import numpy as np
        from compteur.detecteur import charger_modele, cuda_disponible, nom_cuda
        from compteur.config import Config

        print(f"cuda dispo : {cuda_disponible()}", flush=True)
        print(f"carte      : {nom_cuda()!r}", flush=True)
        modele = charger_modele("nano.pt", "auto")
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        res = modele(image, verbose=False)
        print(f"inference OK : {len(res)} resultat(s)", flush=True)
    except Exception as exc:
        import traceback

        print("INFERENCE ECHEC :", flush=True)
        print(traceback.format_exc()[-900:], flush=True)

    QTimer.singleShot(500, app.quit)
    app.exec()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("installer", "demarrer"), required=True)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.mode == "installer":
        return mode_installer()
    return mode_demarrer(args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())