"""Tests du sélecteur de périphérique, de l'indicateur et de l'installateur.

Trois objets distincts, trois raisons d'être distincts :

- `Config.peripherique` est un réglage, comme la résolution d'analyse ;
- `detecteur.peripherique_effectif` est une DÉCISION, qui dépend de ce que la
  machine sait faire — donc testable seulement en simulant la machine ;
- l'installateur choisit un WHEEL, ce qui n'a rien à voir avec le choix d'un
  périphérique à l'exécution.

Les tests de la machine NVIDIA existante ne vérifient pas le chemin GPU par
lui-même — il est déjà exercé sur cette machine — mais le contrat des trois
valeurs, y compris le cas CUDA indisponible, qui n'est pas le cas par défaut.
"""

import pathlib
import subprocess
import sys

import pytest

#: Lancés en sous-processus (vraie détection de GPU, vrai argv) : parmi les
#: plus lourds de la suite. Marqués `slow` — voir pytest.ini.
pytestmark = pytest.mark.slow

from compteur.config import (
    PERIPHERIQUE_AUTO,
    PERIPHERIQUE_CPU,
    PERIPHERIQUE_CUDA,
    PERIPHERIQUES,
    Config,
    mode_peripherique,
)
from compteur.detecteur import (
    DEVICE_CUDA,
    LIBELLE_CPU,
    libelle_peripherique,
    peripherique_effectif,
)

RACINE = pathlib.Path(__file__).resolve().parent.parent


# -- Configuration -------------------------------------------------------


def test_peripherique_vaut_auto_par_defaut():
    """`auto` est le seul mode qui marche sur une machine inconnue."""
    assert Config().peripherique == PERIPHERIQUE_AUTO


def test_le_fichier_versionne_port_le_meme_defaut():
    """Le fichier `config/default.json` ne doit pas diverger du dataclass.

    Sans cette entrée, `Config.defauts()` (qui lit le fichier) et `Config()`
    (qui lit les valeurs codées) décrivraient deux applications différentes.
    """
    defauts = Config.defauts()
    assert defauts.peripherique == Config().peripherique


def test_les_trois_modes_existent():
    assert PERIPHERIQUES == ("auto", "cuda", "cpu")


@pytest.mark.parametrize("mode", PERIPHERIQUES)
def test_un_profil_peut_porter_n_importe_quel_mode(mode, tmp_path):
    """Aller-retour JSON complet, comme un profil enregistré."""
    p = tmp_path / "profil.json"
    Config(peripherique=mode).vers_fichier(p)
    assert Config.depuis_fichier(p).peripherique == mode


def test_un_mode_inconnu_tombe_sur_auto():
    """Un profil à la main ne doit pas rendre le moteur inutilisable."""
    assert mode_peripherique("rocm") == PERIPHERIQUE_AUTO
    assert mode_peripherique(None) == PERIPHERIQUE_AUTO


def test_un_mode_inconnu_ne_casse_pas_le_chargement():
    c = Config.depuis_dict({**Config().vers_dict(), "peripherique": "gpu-amd"})
    assert c.peripherique == PERIPHERIQUE_AUTO


# -- Décision de périphérique --------------------------------------------


@pytest.fixture
def sans_gpu(monkeypatch):
    """Simule une machine sans carte NVIDIA."""
    monkeypatch.setattr("compteur.detecteur.cuda_disponible", lambda: False)
    monkeypatch.setattr("compteur.detecteur.nom_cuda", lambda: "")


@pytest.fixture
def avec_gpu(monkeypatch):
    """Simule une machine avec une carte, sans dépendre du matériel réel."""
    monkeypatch.setattr("compteur.detecteur.cuda_disponible", lambda: True)
    monkeypatch.setattr(
        "compteur.detecteur.nom_cuda", lambda: "NVIDIA GeForce RTX 4070 SUPER"
    )


def test_auto_choisit_cuda_quand_il_y_en_a_un(avec_gpu):
    c = peripherique_effectif(PERIPHERIQUE_AUTO)
    assert c.device == DEVICE_CUDA
    assert c.cuda is True
    assert "RTX 4070" in c.libelle
    assert c.avertissement is None


def test_auto_choisit_le_cpu_sans_carte(sans_gpu):
    c = peripherique_effectif(PERIPHERIQUE_AUTO)
    assert c.device == "cpu"
    assert c.cuda is False
    assert c.libelle == LIBELLE_CPU
    # Un avertissement est normal ici : c'est le cas d'un portable sans GPU.
    assert c.avertissement


def test_cuda_exige_choisit_cuda_meme_sans_gpu_absent(avec_gpu):
    c = peripherique_effectif(PERIPHERIQUE_CUDA)
    assert c.device == DEVICE_CUDA
    assert c.avertissement is None


def test_cuda_demande_sans_carte_bascule_avec_un_message(sans_gpu):
    """Le cas central : `cuda` demandé sur une machine sans NVIDIA.

    Le contrat est une BASCULE avec message, pas une exception : sur le
    terrain, une analyse quatre fois plus lente qui rend un chiffre vaut mieux
    qu'un plantage. Le message doit nommer la cause, sinon l'opérateur croit
    que le programme est cassé.
    """
    c = peripherique_effectif(PERIPHERIQUE_CUDA)
    assert c.device == "cpu"
    assert c.avertissement is not None
    assert "GPU" in c.avertissement
    assert "bascule" in c.avertissement


def test_cpu_force_le_cpu_meme_avec_un_gpu(avec_gpu):
    """Le remède quand le GPU plante sur une scène particulière."""
    c = peripherique_effectif(PERIPHERIQUE_CPU)
    assert c.device == "cpu"
    assert c.cuda is False
    assert c.libelle == LIBELLE_CPU
    # Forcer le CPU en présence d'un GPU est un sacrifice de vitesse : il est
    # dit, sans effacer le choix de l'opérateur.
    assert c.avertissement is not None
    assert "plus lente" in c.avertissement


def test_aucun_mode_ne_signale_une_anomalie_en_mode_auto_avec_gpu(avec_gpu):
    """Le cas par défaut doit être silencieux : rien à signaler à personne."""
    assert peripherique_effectif(PERIPHERIQUE_AUTO).avertissement is None


def test_le_device_cuda_est_bien_le_format_attendu_par_torch(avec_gpu):
    """`cuda:0`, pas `cuda` : ultralytics attend un périphérique indexable."""
    assert peripherique_effectif(PERIPHERIQUE_AUTO).device == "cuda:0"


# -- Libellé affiché -----------------------------------------------------


def test_le_libelle_annonce_le_peripherique_avec_gpu(avec_gpu):
    texte = libelle_peripherique(PERIPHERIQUE_AUTO)
    assert texte.startswith("Calcul : ")
    assert "CUDA" in texte
    assert "RTX 4070" in texte


def test_le_libelle_annonce_le_cpu_sans_gpu(sans_gpu):
    """L'opérateur doit voir « CPU » et non un device comme `cuda:0`."""
    assert libelle_peripherique(PERIPHERIQUE_AUTO) == f"Calcul : {LIBELLE_CPU}"


def test_le_libelle_ne_contient_jamais_de_device_brut(avec_gpu):
    """« cuda:0 » à l'écran serait du jargon : on affiche le nom de la carte."""
    assert "cuda:0" not in libelle_peripherique(PERIPHERIQUE_AUTO)


# -- Installateur --------------------------------------------------------


def _dry_run(*args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RACINE / "installer.py"), "--dry-run", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(RACINE),
        env=env,
    )


def test_le_dry_run_choisit_la_branche_cuda_sur_une_machine_avec_carte():
    """Cette machine a une RTX 4070 : la branche CUDA doit être retenue.

    Le test vérifie l'affichage, pas la présence du GPU : sur une machine sans
    carte la même commande affiche la branche CPU, ce que le test suivant
    vérifie par un chemin indépendant du matériel.
    """
    proc = _dry_run()
    assert proc.returncode == 0
    assert "Version de torch à installer : CUDA" in proc.stdout
    assert "download.pytorch.org" in proc.stdout


def test_le_dry_run_choisit_la_branche_cpu_quand_le_gpu_est_desactive():
    """`--sans-gpu` est le moyen PORTABLE de vérifier la branche CPU.

    `CUDA_VISIBLE_DEVICES=""` ne fonctionne pas sous Windows : la variable est
    lue par le runtime Linux, et `nvidia-smi` — qui interroge le pilote —
    continue d'annoncer la carte. Un test qui s'appuierait dessus serait vert
    sur WSL et rouge sur cette machine, donc muet.
    """
    proc = _dry_run("--sans-gpu")
    assert proc.returncode == 0
    assert "Carte NVIDIA détectée : non" in proc.stdout
    assert "Version de torch à installer : CPU" in proc.stdout
    assert "download.pytorch.org" not in proc.stdout


def test_le_dry_run_n_installe_rien():
    proc = _dry_run()
    assert "rien n'a été installé" in proc.stdout
    assert "Successfully installed" not in proc.stdout


def test_cuda_exige_sans_carte_echoue_avant_d_installer():
    """Le seul échec du script, et il est volontaire.

    Le logiciel bascule avec un message ; l'installateur, non. Ici, continuer
    produirait une installation silencieusement inutile, alors que
    l'utilisateur est devant son écran, une seule fois, et qu'il peut choisir
    autre chose. Le code de sortie doit être distinct de zéro pour qu'un
    script d'installation automatisé s'arrête.
    """
    proc = _dry_run("--sans-gpu", "--peripherique", "cuda")
    assert proc.returncode == 2
    assert "ÉCHEC" in proc.stderr
    assert "Rien n'a été installé" in proc.stderr
    # L'échec doit dire quoi faire ensuite, pas seulement constater.
    assert "--peripherique cpu" in proc.stderr


def test_forcer_le_cpu_meme_avec_une_carte_installe_le_wheel_cpu():
    """`--peripherique cpu` prime sur la carte détectée."""
    proc = _dry_run("--peripherique", "cpu")
    assert proc.returncode == 0
    assert "Carte NVIDIA détectée : oui" in proc.stdout
    assert "Version de torch à installer : CPU" in proc.stdout
