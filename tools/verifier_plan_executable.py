"""Execute les blocs Python du plan contre le depot reel.

Le plan est le cahier des charges des taches futures : si son code ne tourne
pas, les taches suivantes recopient le bug. Ce script extrait chaque bloc
```python de la section demandee, l'ecrit dans un dossier temporaire avec le
paquet `compteur` reel, et lance pytest dessus.

Usage :
    python tools/verifier_plan_executable.py "Tâche 5" "Tâche 7"

Historique : quatre tests du plan se sont reveles etre des faux positifs ou
des faux negatifs structurels, alors que le plan semblait correct a la
relecture. Ce script existe pour que ca ne se reproduise pas.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

RACINE = pathlib.Path(__file__).resolve().parent.parent
PLAN = RACINE / "docs" / "superpowers" / "plans" / "2026-10-02-comptage-manifestation.md"

# Quel nom de fichier de compteur/ chaque tache doit produire.
SORTIE_PAR_TACHE = {
    "Tâche 5": "compteur.py",
    "Tâche 6": "rapport.py",
    "Tâche 7": "rapport.py",
}


def blocs(titre):
    texte = PLAN.read_text(encoding="utf-8")
    start = re.search(rf"^## {re.escape(titre)}", texte, re.M)
    if not start:
        sys.exit(f"section introuvable : {titre}")
    fin = re.search(r"^## Tâche \d+ :", texte[start.end():], re.M)
    if fin:
        bornee = texte[start.start() : start.end() + fin.start()]
    else:
        bornee = texte[start.start() :]
    return re.findall(r"```python\n(.*?)```", bornee, re.S)


def main():
    # Les sections ne sont pas independantes : la tache 7 importe
    # compteur.compteur, cree par la tache 5. Verifier "Tache 7" seule
    # echoue donc sur cet import, ce qui est attendu.
    titres = sys.argv[1:] or ["Tâche 5", "Tâche 7"]
    avec_boucle = tempfile.TemporaryDirectory()
    with avec_boucle as tmp:
        base = pathlib.Path(tmp)
        base.mkdir(exist_ok=True)

        tests, modules = [], []
        for titre in titres:
            for i, code in enumerate(blocs(titre)):
                nom = f"tache{titre.split()[1]}_{i}.py"
                (base / nom).write_text(code, encoding="utf-8")
                (tests if "def test_" in code else modules).append((titre, nom))

        (base / "conftest.py").write_text(
            "import sys, pathlib\n"
            f'sys.path.insert(0, r"{RACINE}")\n'
            f'sys.path.insert(0, r"{base}")\n',
            encoding="utf-8",
        )
        # Les modules du plan doivent s'importer par leur vrai nom. Le plan annonce
        # en entete de tache les fichiers a creer : compteur/compteur.py pour la
        # tache 5. On copie le paquet reel et on ecrase avec la version du
        # plan, d'apres le nom de fichier annonce dans la section.
        shutil.copytree(RACINE / "compteur", base / "compteur")
        shutil.copytree(RACINE / "config", base / "config")
        # La tache 5 depend des taches 3 et 4 (detecteur, tracker), qui
        # n'existent pas encore dans le depot. On fournit des bouchons
        # minimaux : seule la geometrie de la ligne est verifiee ici.
        (base / "compteur" / "detecteur.py").write_text(
            "class Detecteur:\n"
            "    def __init__(self, modele, config):\n"
            "        self.modele = modele\n"
            "        self.config = config\n"
            "    def detecter(self, img):\n"
            "        return []\n\n\n"
            "def charger_modele(chemin):\n"
            "    raise RuntimeError('bouchon de verification')\n",
            encoding="utf-8",
        )
        (base / "compteur" / "tracker.py").write_text(
            "class Tracker:\n"
            "    def __init__(self, config, backend=None):\n"
            "        self.config = config\n"
            "    def mettre_a_jour(self, detections, frame_index):\n"
            "        return []\n"
            "    def reinitialiser(self):\n"
            "        pass\n"
            "    def nb_tracks_vus(self):\n"
            "        return 0\n",
            encoding="utf-8",
        )
        # Chaque module du plan doit s'importer par son vrai nom : on copie
        # le paquet reel puis on ecrase le fichier avec la version du plan.
        for titre, nom_source in modules:
            nom = SORTIE_PAR_TACHE.get(titre)
            if nom is None:
                continue
            shutil.copy(base / nom_source, base / "compteur" / nom)

        cmd = [sys.executable, "-m", "pytest", "-q", "--no-header",
               "-p", "no:cacheprovider"] + [str(base / n) for _, n in tests]
        # Herite de l'environnement reel (numpy, cv2, pytest) et y ajoute
        # seulement le dossier temporaire en tete de PYTHONPATH.
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = str(base) + os.pathsep + str(RACINE)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(base), env=env)
        print(proc.stdout[-4000:])
        if proc.stderr.strip():
            print("STDERR:", proc.stderr[-1500:])
        return proc.returncode


if __name__ == "__main__":
    sys.exit(main())