"""Copier les poids `.pt` à côté de `dist/CompteurManifestationV2/CompteurManifestationV2.exe`.

Pourquoi une copie et pas un embarquement dans l'exécutable :

- les poids font 50 Mo chacun (`medium.pt` : 50 Mo, `yolov8n-head.pt` :
  50 Mo, `nano.pt` : 6 Mo) et sont **volontairement hors du dépôt**
  (`.gitignore` : `*.pt`) ;
- le modèle est un **paramètre**, pas une dépendance : changer de détecteur,
  en ajouter un, ou essayer un modèle entraîné localement doit se faire en
  déposant un fichier, sans reconstruire l'exécutable ;
- l'archive onefile est ré-écrite en entier à chaque construction : y mettre
  les poids allonge le build sans rien apporter.

Le panneau de réglages liste les `.pt` du dossier courant et de
`modeles/` (`interface/panneau_reglages.py::DOSSIERS_MODELES`). Un double-clic
sur l'exécutable donne pour dossier courant le dossier de l'exécutable : les
poids copiés ici apparaissent donc dans la liste, sans code supplémentaire.

Usage :

    python tools/copier_modeles.py            # medium.pt, nano.pt, yolov8n-head.pt
    python tools/copier_modeles.py --dest /tmp/ailleurs
    python tools/copier_modeles.py --propre   # ne recopie que ce qui manque
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

RACINE = pathlib.Path(__file__).resolve().parent.parent

#: Dossiers de sortie par défaut. Le `.spec` produit un dossier
#: (`dist/CompteurManifestationV2/`) et non un `.exe` isolé — voir le README,
#: section « Running without installing anything ». On tolère les deux : si un
#: build onefile a été produit, les poids vont à côté du `.exe`.
dossiers_cibles = (
    RACINE / "dist" / "CompteurManifestationV2",
    RACINE / "dist",
)

#: Poids livrés avec le projet, dans l'ordre de préférence pour l'utilisateur.
MODELES_DEFAUT = ("medium.pt", "nano.pt", "yolov8n-head.pt")

#: Extension acceptée, alignée sur `compteur.detecteur.FORMATS_MODELES`.
SUFFIXES = (".pt", ".onnx")


def modeles_trouves(dossier: pathlib.Path = RACINE) -> list[pathlib.Path]:
    """Poids présents à la racine du dépôt, triés par taille décroissante.

    Trier par taille n'est pas cosmétique : `nano.pt` (6 Mo) est le seul
    modèle assez léger pour être ouvert sur une machine sans GPU rapide, et
    l'utilisateur doit le voir en premier.
    """
    trouves = [
        p
        for p in dossier.iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES
    ]
    return sorted(trouves, key=lambda p: p.stat().st_size)


def copier(
    sources: list[pathlib.Path],
    dest: pathlib.Path,
    ecraser: bool = True,
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Copie ``sources`` dans ``dest``. Retourne (copiés, ignorés)."""
    dest.mkdir(parents=True, exist_ok=True)
    copies: list[pathlib.Path] = []
    ignores: list[pathlib.Path] = []
    for src in sources:
        cible = dest / src.name
        if cible.exists() and not ecraser:
            ignores.append(cible)
            continue
        shutil.copy2(src, cible)
        copies.append(cible)
    return copies, ignores


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument(
        "--dest",
        default=None,
        help="dossier cible (défaut : le premier dossier de build qui existe)",
    )
    p.add_argument(
        "--propre",
        action="store_true",
        help="ne recopie que les poids manquants",
    )
    args = p.parse_args(argv)

    dest = (
        pathlib.Path(args.dest)
        if args.dest
        # Premier dossier de build présent, dans l'ordre de préférence :
        # `dist/CompteurManifestationV2/` (mode dossier) avant `dist/`
        # (mode onefile, où l'exécutable est à la racine de `dist/`).
        else next((d for d in dossiers_cibles if d.is_dir()), dossiers_cibles[0])
    )
    if not dest.is_dir():
        present = [d for d in dossiers_cibles if d.is_dir()]
        if present:
            print(
                f"ERREUR : {dest} n'existe pas.\n"
                f"Le build a produit {present[0]} — utilise :\n"
                f"    python tools/copier_modeles.py "
                f"--dest \"{present[0]}\"",
                file=sys.stderr,
            )
        else:
            print(
                f"ERREUR : {dest} n'existe pas. Construis d'abord l'exécutable :\n"
                "    python -m PyInstaller --clean --workpath build/v2 crowd-counter-v2.spec",
                file=sys.stderr,
            )
        return 1

    trouves = modeles_trouves()
    if not trouves:
        print(
            "Aucun poids .pt / .onnx à la racine du dépôt.\n"
            "Dépose-les à côté de main.py, puis relance ce script.\n"
            f"Dossier attendu : {RACINE}",
            file=sys.stderr,
        )
        return 1

    copies, ignores = copier(trouves, dest, ecraser=not args.propre)

    for c in copies:
        print(f"copié  {c.name}  ({c.stat().st_size / 1024 / 1024:.1f} Mo)")
    for i in ignores:
        print(f"déjà là {i.name} (inchangé)")
    print(f"\n{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())