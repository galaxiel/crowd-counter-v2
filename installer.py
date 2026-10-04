"""Installation des dépendances, en choisissant la bonne version de torch.

Pourquoi un script plutôt que `pip install -r requirements.txt` : torch se
publie en deux familles de Wheels distinctes — CPU et CUDA — et le paquet nommé
`torch` sur PyPI ne dit pas laquelle on obtient. Le résultat est une
installation qui a l'air bonne et tourne à 7 img/s.

Ce script détecte la carte, annonce ce qu'il va faire, puis l'exécute :

- **carte NVIDIA détectée** -> torch depuis l'index CUDA ;
- **aucune carte** -> torch CPU depuis PyPI ;
- **CUDA demandé à la main sur une machine sans carte** -> ÉCHEC, avant toute
  installation. C'est la seule erreur fatale du script, et elle est
  volontaire : ici, continuer produirait une installation silencieusement
  inutile, alors que le logiciel lui-même préfère basculer sur le CPU avec un
  message (voir `compteur/detecteur.py`).

Les deux contrats divergent, et c'est justifié : l'installateur est exécuté
UNE fois, par quelqu'un qui lit l'écran ; le logiciel tourne sur le terrain,
par quelqu'un qui ne lit rien.

Utilisation :

    python installer.py                     # détecte et installe
    python installer.py --dry-run           # affiche la branche retenue
    python installer.py --peripherique cpu  # force le wheel CPU
    python installer.py --peripherique cuda # exige le GPU, échoue sans carte
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from typing import NamedTuple

log = logging.getLogger("installer")

#: Index officiel des wheels torch CUDA. Sans suffixe de version, pip prend la
#: dernière publication CUDA — celle que l'exécutable embarque.
INDEX_CUDA = "https://download.pytorch.org/whl/cu126"

#: Paquets de premier rang, installés dans tous les cas.
PAQUETS_COMMUNS = ("ultralytics", "opencv-python", "numpy", "PySide6")


class Detection(NamedTuple):
    """Ce que la machine a, et d'où vient la réponse.

    `origine` est exposée parce qu'une détection muette est invérifiable :
    l'utilisateur doit pouvoir distinguer « j'ai cherché et rien n'a répondu »
    de « torch n'est pas installé, donc je ne sais pas ».
    """

    nvidia: bool
    origine: str
    carte: str = ""


def nvidia_smi() -> str:
    """Nom de la première carte NVIDIA, ou chaîne vide si `nvidia-smi` muet.

    Lu AVANT torch parce qu'il répond même quand torch n'est pas installé —
    ce qui est précisément la situation de ce script. Un `nvidia-smi` absent,
    en échec ou bavard est traité comme une absence de réponse : ce n'est
    pas une erreur, c'est une information, et la branche de repli (wheel CPU)
    fonctionne dans tous les cas.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ""
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("nvidia-smi présent mais inexploitable : %s", exc)
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def detecter_gpu() -> Detection:
    """Cherche une carte NVIDIA : `nvidia-smi` d'abord, torch en confirmation.

    L'ordre n'est pas indifférent. `nvidia-smi` parle du MATÉRIEL : s'il
    répond, il y a une carte, que torch sache ou non l'utiliser. torch parle
    de l'ENVIRONNEMENT : il peut répondre `False` sur une machine qui a une
    carte mais un torch CPU-only installé. torch ne sert donc qu'à rattraper
    une carte que `nvidia-smi` ne voit pas (installation partielle, PATH
    incomplet).

    **`CUDA_VISIBLE_DEVICES` prime sur `nvidia-smi`, et c'est mesuré.** Cette
    variable est lue par torch, PAS par le pilote : sur une machine avec une
    carte, `nvidia-smi` continue donc d'annoncer le GPU même quand la variable
    est vide. Interroger le pilote d'abord ignorantait le seul mécanisme qui
   permette de dire « ici, le GPU est désactivé » — typiquement un conteneur,
    un service, ou l'opérateur qui travaille autour d'un GPU instable.

    Quand la variable est positionnée, on ne demande donc qu'à torch, qui la
    respecte : c'est aussi ce qui permet de vérifier la branche CPU sur une
    machine qui a bel et bien une carte.
    """
    import os

    filtre = os.environ.get("CUDA_VISIBLE_DEVICES")
    if filtre is None:
        nom = nvidia_smi()
        if nom:
            return Detection(True, "nvidia-smi", nom)

    try:
        import torch
    except ImportError:
        return Detection(False, "ni nvidia-smi, ni torch")
    try:
        # `device_count()` et non `is_available()` : mesuré sur cette machine,
        # `is_available()` renvoie `True` alors que `device_count()` renvoie 0
        # quand le GPU est désactivé. Compter les cartes ne ment pas.
        if int(torch.cuda.device_count()) > 0:
            return Detection(
                True, "torch", str(torch.cuda.get_device_name(0)).strip()
            )
    except Exception as exc:  # noqa: BLE001 — torch présent mais cassé
        log.info("torch présent mais CUDA inexploitable : %s", exc)
    if filtre is not None:
        return Detection(False, "CUDA_VISIBLE_DEVICES", f"= {filtre!r}")
    return Detection(False, "ni nvidia-smi, ni torch.cuda")


def commande_torch(gpu: bool) -> list[str]:
    """Arguments `pip install` de torch, CUDA ou CPU."""
    if gpu:
        return ["torch", "torchvision", "--index-url", INDEX_CUDA]
    # Sans GPU, le wheel PyPI EST le wheel CPU. Aucun drapeau à ajouter : c'est
    # ce que `pip install torch` donne, et le dire évite qu'un utilisateur
    # croie avoir installé CUDA en lisant requirements.txt.
    return ["torch", "torchvision"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="installer.py",
        description="Installe les dépendances en détectant le GPU.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="affiche la branche retenue sans rien installer",
    )
    p.add_argument(
        "--sans-gpu",
        action="store_true",
        help=(
            "considère la machine comme dépourvue de carte NVIDIA "
            "(équivalent portable de CUDA_VISIBLE_DEVICES=\"\")"
        ),
    )
    p.add_argument(
        "--peripherique",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "auto (défaut) : CUDA si une carte est présente ; "
            "cuda : l'exige, et échoue sans carte ; "
            "cpu : installe le wheel CPU même avec une carte"
        ),
    )
    return p.parse_args(argv)


def annoncer(detection: Detection, gpu: bool) -> None:
    """Affiche la branche retenue, AVANT toute installation.

    Un installateur qui n'annonce rien est un installateur dont on ne peut pas
    vérifier la décision : `--dry-run` sert exactement à cela.
    """
    print(f"Carte NVIDIA détectée : {'oui' if detection.nvidia else 'non'}")
    print(f"  source : {detection.origine}")
    if detection.carte:
        print(f"  carte : {detection.carte}")
    print(f"Version de torch à installer : {'CUDA' if gpu else 'CPU'}")
    print(f"  index : {INDEX_CUDA if gpu else 'PyPI (wheel CPU)'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    detection = Detection(False, "--sans-gpu") if args.sans_gpu else detecter_gpu()

    if args.peripherique == "cuda" and not detection.nvidia:
        print(
            "ÉCHEC : calcul GPU demandé, mais aucune carte NVIDIA n'a été "
            f"détectée sur cette machine ({detection.origine}).\n"
            "Rien n'a été installé. Deux solutions :\n"
            "  - laisser le programme décider : python installer.py\n"
            "  - installer la version CPU   : python installer.py --peripherique cpu",
            file=sys.stderr,
        )
        return 2

    gpu = detection.nvidia and args.peripherique != "cpu"
    annoncer(detection, gpu)

    etapes = [commande_torch(gpu), list(PAQUETS_COMMUNS)]
    if args.dry_run:
        for etape in etapes:
            print(f"  pip install {' '.join(etape)}")
        print("\n--dry-run : rien n'a été installé.")
        return 0

    for etape in etapes:
        commande = [sys.executable, "-m", "pip", "install", *etape]
        print(f"\n$ {' '.join(commande)}")
        code = subprocess.call(commande)
        if code != 0:
            print(
                f"\nÉCHEC de l'installation (code {code}). "
                "Corrige la cause ci-dessus, puis relance.",
                file=sys.stderr,
            )
            return code
    print("\nInstallation terminée. Lance `python main.py`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
