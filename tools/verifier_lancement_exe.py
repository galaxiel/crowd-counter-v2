"""Vérifier que l'exécutable `dist/CompteurManifestation*` se LANCE vraiment.

Un `.exe` qui se construit n'est pas un `.exe` qui marche. Cet outil lance
l'exécutable **une seule fois** — un démarrage onefile décompresse 3 Go,
et deux lancements coûteraient deux fois ce délai — puis interroge en boucle
le processus et ses fenêtres.

Trois choses qu'aucun build ne dit :

1. **le processus survit au démarrage** — un `.exe` qui plante se ferme
   avant, et le double-clic ne montre rien ;
2. **une fenêtre titrée apparaît** — énumération Win32 des fenêtres de premier
   niveau du PID (`interface/app.py` fixe le titre). Pas une capture
   d'écran : celle-ci peut capturer le bureau vide si l'application n'a pas
   encore peint, et « rien de visible » ne distingue pas « pas encore
   peint » de « planté » ;
3. **le repli silencieux de `Config` n'a pas eu lieu** — c'est journalisé
   dans la sortie console, et c'est le seul endroit où le défaut se voit :
   un utilisateur ne lit pas le journal.

Le dossier courant est forcé sur `dist/`, ce que fait l'explorateur Windows au
double-clic — c'est ce qui doit permettre de trouver les `.pt`.

Usage :

    python tools/verifier_lancement_exe.py
    python tools/verifier_lancement_exe.py --attente 240
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import pathlib
import subprocess
import sys
import time

RACINE = pathlib.Path(__file__).resolve().parent.parent

#: Emplacements selon le mode de build. Le `.spec` produit par défaut un
#: DOSSIER (`dist/CompteurManifestation/`) — le onefile est plafonné à 2 Go,
#: ce que les DLL CUDA dépassent ; voir `crowd-counter.spec`. Les deux sont
#: essayés pour que l'outil serve aussi à un build `COMPTEUR_ONEFILE=1`.
EXES_CANDATS = (
    RACINE / "dist" / "CompteurManifestation" / "CompteurManifestation.exe",
    RACINE / "dist" / "CompteurManifestation.exe",
)


def trouver_exe() -> pathlib.Path | None:
    """Premier exécutable construit, ou None."""
    return next((p for p in EXES_CANDATS if p.is_file()), None)

#: Modèle attendu à côté de l'exécutable (valeur de config/default.json).
MODELE_ATTENDU = "medium.pt"

#: Titre de la fenêtre principale (interface/app.py).
TITRE_ATTENDU = "Compteur de manifestation"

#: Extrait de la sortie console qui trahit le repli silencieux de Config.
SIGNE_REPLI = "config/default.json introuvable"

#: Après apparition de la fenêtre, on laisse ce délai avant de conclure.
PAUSE_APRES_FENETRE_S = 5.0


def lister_fenetres(pid: int) -> list[tuple[int, str]]:
    """Fenêtres visibles de premier niveau du processus `pid` : (handle, titre)."""
    trouvees: list[tuple[int, str]] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )

    @WNDENUMPROC
    def _visiter(hwnd, _lparam):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True
        owner = ctypes.wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid:
            return True
        longueur = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        tampon = ctypes.create_unicode_buffer(longueur + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, tampon, longueur + 1)
        trouvees.append((int(hwnd), tampon.value))
        return True

    ctypes.windll.user32.EnumWindows(_visiter, 0)
    return trouvees


def verifier(attente: int) -> int:
    exe = trouver_exe()
    if exe is None:
        attendu = "\n".join(f"    {p}" for p in EXES_CANDATS)
        print("ERREUR : aucun exécutable construit.", file=sys.stderr)
        print(f"Emplacements attendus :\n{attendu}", file=sys.stderr)
        print(
            "Construisez-le : python -m PyInstaller --clean crowd-counter.spec",
            file=sys.stderr,
        )
        return 1
    EXE = exe  # noqa: N806 — chemin de la build vérifiée, figé pour le rapport

    taille_mo = EXE.stat().st_size / 1024 / 1024
    mode = "one-dossier" if EXE.parent.name == "CompteurManifestation" else "onefile"
    print(f"exécutable  : {EXE}  ({taille_mo:.1f} Mo)")
    print(f"mode        : {mode}")
    dossier = EXE.parent
    interne = dossier / "_internal"
    if interne.is_dir():
        vol = sum(f.stat().st_size for f in interne.rglob("*") if f.is_file())
        print(f"_internal/  : {vol / 1024 ** 3:.2f} Go de DLL et modules")
    total = sum(f.stat().st_size for f in dossier.rglob("*") if f.is_file())
    print(f"poids total : {total / 1024 ** 3:.2f} Go (dossier complet)")
    print(f"dossier courant : {dossier}  (comme au double-clic)")
    print(f"attente max : {attente} s\n")

    journal = RACINE / "build" / "lancement_exe.log"
    journal.parent.mkdir(parents=True, exist_ok=True)

    echecs: list[str] = []
    debut = time.time()
    with journal.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(
            [str(EXE)], cwd=str(EXE.parent), stdout=f, stderr=subprocess.STDOUT
        )

        fenetres: list[tuple[int, str]] = []
        instant_fenetre: float | None = None
        mort_avant_fenetre = False

        while time.time() - debut < attente:
            if proc.poll() is not None:
                mort_avant_fenetre = True
                break
            fenetres = lister_fenetres(proc.pid)
            if fenetres:
                if instant_fenetre is None:
                    instant_fenetre = time.time() - debut
                    print(
                        f"[1] fenêtre apparue à {instant_fenetre:.1f} s "
                        f"(PID {proc.pid})"
                    )
                if time.time() - debut > instant_fenetre + PAUSE_APRES_FENETRE_S:
                    break
            time.sleep(0.5)

        if mort_avant_fenetre:
            code = proc.returncode
            print(
                f"[1] ECHEC : le processus a quitté après "
                f"{time.time() - debut:.1f} s, code de sortie {code}"
            )
            echecs.append(f"l'exécutable a quitté avec le code {code}")
        elif instant_fenetre is None:
            print(
                f"[1] ECHEC : aucune fenêtre après {attente} s "
                f"(processus {'vivant' if proc.poll() is None else 'mort'})"
            )
            echecs.append(f"aucune fenêtre après {attente} s")

        print("\n[2] fenêtres visibles du processus :")
        if fenetres:
            for hwnd, titre in fenetres:
                marque = "  <-- attendue" if TITRE_ATTENDU in titre else ""
                print(f"    hwnd={hwnd}  titre={titre!r}{marque}")
            if not any(TITRE_ATTENDU in t for _, t in fenetres):
                echecs.append(f"aucune fenêtre titrée {TITRE_ATTENDU!r}")
        else:
            print("    (aucune)")

        if proc.poll() is None:
            print("\n[3] processus encore vivant — arrêt propre.")
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print("    terminate() ignoré — kill()")
                proc.kill()
                proc.wait(timeout=20)
        else:
            echecs.append("le processus s'est arrêté de lui-même pendant l'analyse")

    # -- 4. Les poids. ---------------------------------------------------
    print("\n[4] poids à côté de l'exécutable :")
    poids = sorted(EXE.parent.glob("*.pt"))
    if poids:
        for p in poids:
            print(f"    {p.name}  ({p.stat().st_size / 1024 / 1024:.1f} Mo)")
    else:
        print("    (aucun)")
    if not (EXE.parent / MODELE_ATTENDU).is_file():
        echecs.append(
            f"{MODELE_ATTENDU} absent de {EXE.parent} — "
            "lancer `python tools/copier_modeles.py`"
        )

    # -- 5. Le repli silencieux. ----------------------------------------
    console = journal.read_text(encoding="utf-8", errors="replace")
    print(f"\n[5] sortie console ({journal}) :")
    lignes = console.splitlines()
    if not lignes:
        print("    (vide — l'application n'a rien journalisé)")
    else:
        for ligne in lignes[-25:]:
            print(f"    {ligne}")
    if SIGNE_REPLI in console:
        echecs.append(
            f"« {SIGNE_REPLI} » : config/default.json n'a pas été embarqué "
            "ou n'a pas été trouvé"
        )

    print("\n" + "=" * 62)
    if echecs:
        print("ECHEC :")
        for e in echecs:
            print(f"  - {e}")
        return 1
    print("OK : démarre, ouvre sa fenêtre, trouve ses poids, config lue.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument(
        "--attente",
        type=int,
        default=240,
        help="secondes maximum d'attente (défaut : 240)",
    )
    return verifier(p.parse_args().attente)


if __name__ == "__main__":
    raise SystemExit(main())