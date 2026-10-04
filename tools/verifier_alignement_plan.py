"""Compare les blocs de code Python du plan avec les fichiers réellement livrés.

Ne teste pas la correction en elle-même : vérifie que le plan et le dépôt ne
divergent pas, parce que c'est le plan qui sert de cahier des charges aux
tâches suivantes.
"""
import pathlib
import re
import sys

RACINE = pathlib.Path(__file__).resolve().parent.parent
PLAN = RACINE / "docs" / "superpowers" / "plans" / "2026-10-02-comptage-manifestation.md"

MAPPING = [
    # (cible, fichier, ancre, taille_minimale) — la taille minimale ecarte les
    # blocs qui citent juste le nom (par exemple le fichier de test cite
    # chemin_defaut_config sans definir la classe Config).
    ("tests/test_isolation_ui.py", "tests/test_isolation_ui.py", "MODULES_INTERDITS", 1500),
    ("tests/test_config.py", "tests/test_config.py", "test_ligne_none_est_acceptee", 1500),
    ("compteur/config.py", "compteur/config.py", "@dataclass", 1500),
    ("compteur/types.py", "compteur/types.py", "class Detection", 1500),
]


def blocs_python(texte: str):
    return re.findall(r"```python\n(.*?)```", texte, re.S)


def principal(plan_texte: str, ancre: str, taille_min: int):
    """Le plus gros bloc python du plan contenant l'ancre du fichier cible."""
    meilleur, taille = None, 0
    for bloc in blocs_python(plan_texte):
        if ancre in bloc and len(bloc) > taille:
            meilleur, taille = bloc, len(bloc)
    if meilleur is None or taille < taille_min:
        return None
    return meilleur


def main() -> int:
    plan = PLAN.read_text(encoding="utf-8")
    ecarts = []
    for cible, fichier, ancre, taille_min in MAPPING:
        bloc = principal(plan, ancre, taille_min)
        reel = (RACINE / fichier).read_text(encoding="utf-8")
        if bloc is None:
            ecarts.append(
                f"{cible}: aucun bloc de >= {taille_min} car contenant {ancre!r}"
            )
            continue
        # Compare la presence des definitions, pas le texte exact : les
        # commentaires du plan et du fichier peuvent differer legitimement.
        # On compare les NOMS de definitions, pas la ligne entiere : le plan
        # peut indenter differemment.
        motif = r"^\s*(?:def|class)\s+(\w+)"
        defs_plan = set(re.findall(motif, bloc, re.M))
        defs_reel = set(re.findall(motif, reel, re.M))
        manquantes = defs_reel - defs_plan
        if manquantes:
            ecarts.append(
                f"{cible}: {len(manquantes)} definition(s) dans le code mais "
                f"absentes du plan : {sorted(manquantes)}"
            )
        else:
            print(f"OK  {cible}: {len(defs_reel)} definitions alignees")
    if ecarts:
        print()
        for e in ecarts:
            print("ECART " + e)
        return 1
    print()
    print("aucun ecart : le plan et le depot sont alignes")
    return 0


if __name__ == "__main__":
    sys.exit(main())