"""Preuve que le chemin de configuration est résolu DANS l'exécutable gelé.

`tools/verifier_lancement_exe.py` vérifie que la fenêtre s'ouvre. Il ne
prouve pas que `config/default.json` a été lu : un repli silencieux ne lève
rien, ne journalise qu'un `WARNING` que l'utilisateur ne voit pas, et
l'application démarrerait quand même, avec les mêmes valeurs puisque le
fichier et le dataclass coïncident.

Ce script lève la limite en instrumentant le gel pour de vrai :

1. il lance l'exécutable avec une variable d'environnement qui lui fait
   écrire, AVANT toute construction de fenêtre, un rapport de diagnostic
   (`injection_probe.py`) : `sys.frozen`, `sys._MEIPASS`, le chemin résolu
   par `compteur.config._racine()`, et les valeurs lues ;
2. il compare ce chemin au fichier réellement présent dans `_internal/` ;
3. il échoue si le chemin ne désigne pas le fichier embarqué, ou si les
   valeurs lues ne correspondent pas au JSON.

`injection_probe.py` est exécuté par le bootloader, donc il s'exécute dans
le vrai contexte gelé : `sys.frozen` et `sys._MEIPASS` ne sont pas
simulés. C'est la différence entre « le test unitaire passe » et « le
logiciel livré trouve sa configuration ».

Usage :

    python tools/verifier_config_gelee.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

RACINE = pathlib.Path(__file__).resolve().parent.parent
EXE = RACINE / "dist" / "CompteurManifestation" / "CompteurManifestation.exe"
INTERNE = EXE.parent / "_internal"
CONFIG_EMBARQUE = INTERNE / "config" / "default.json"
PROBE = RACINE / "tools" / "injection_probe.py"
RAPPORT = RACINE / "build" / "probe_config.json"


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--attente", type=int, default=180)
    args = p.parse_args()

    echecs: list[str] = []
    if not EXE.is_file():
        print(f"ERREUR : {EXE} absent.", file=sys.stderr)
        return 1
    if not CONFIG_EMBARQUE.is_file():
        echecs.append(f"config/default.json absent de {INTERNE}")
    if not PROBE.is_file():
        echecs.append(f"sonde absente : {PROBE}")
    if echecs:
        for e in echecs:
            print(f"ERREUR : {e}", file=sys.stderr)
        return 1

    RAPPORT.parent.mkdir(parents=True, exist_ok=True)
    RAPPORT.unlink(missing_ok=True)

    env = dict(os.environ)
    env["COMPTEUR_PROBE"] = str(PROBE)
    env["COMPTEUR_PROBE_RAPPORT"] = str(RAPPORT)
    # Le bootloader exécute le script de runtime hook AVANT main.py : c'est le
    # seul point où l'on peut observer le gel authenticité.
    proc = subprocess.Popen(
        [str(EXE)],
        cwd=str(EXE.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.wait(timeout=args.attente)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    # Le rapport est écrit dès le chargement de la sonde ; l'application a
    # pu rester ouverte (c'est le comportement attendu d'un double-clic).
    for _ in range(40):
        if RAPPORT.is_file():
            break
        import time

        time.sleep(0.25)

    if not RAPPORT.is_file():
        print("ECHEC : la sonde n'a rien écrit dans l'exécutable.", file=sys.stderr)
        print(f"        rapport attendu : {RAPPORT}", file=sys.stderr)
        return 1

    r = json.loads(RAPPORT.read_text(encoding="utf-8"))
    print("Rapport écrit par le processus gelé :")
    for cle in (
        "frozen",
        "meipass",
        "executable",
        "racine",
        "config_resolu",
        "config_attendu",
        "config_lisible",
        "modele",
        "taille_entree",
        "fenetre_lissage",
        "seuil_matching",
    ):
        if cle in r:
            print(f"    {cle:16} = {r[cle]}")

    print()
    if r.get("frozen") is not True:
        echecs.append("sys.frozen n'est pas True dans l'exécutable")
    if r.get("config_resolu") != str(CONFIG_EMBARQUE):
        echecs.append(
            f"chemin gelé résolu {r.get('config_resolu')!r} "
            f"au lieu de {str(CONFIG_EMBARQUE)!r}"
        )
    if r.get("config_lisible") is not True:
        echecs.append("Config.defauts() n'a pas pu lire le fichier embarqué")
    if r.get("repli_silencieux"):
        echecs.append("Config.defauts() a déclenché le repli silencieux")

    # Les valeurs doivent correspondre au JSON embarqué, pas au dataclass :
    # c'est la seule façon de distinguer une lecture d'un repli, puisque
    # `Config()` et le fichier coïncident par construction (un test le vérifie).
    attendu = json.loads(CONFIG_EMBARQUE.read_text(encoding="utf-8"))
    for cle in ("modele", "taille_entree", "fenetre_lissage", "seuil_matching"):
        if r.get(cle) != attendu[cle]:
            echecs.append(
                f"{cle} lu = {r.get(cle)!r}, JSON embarqué = {attendu[cle]!r}"
            )

    if echecs:
        print("ECHEC :")
        for e in echecs:
            print(f"  - {e}")
        return 1
    print("OK : le binaire gelé résout et LIT config/default.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())