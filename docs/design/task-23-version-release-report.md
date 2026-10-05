# Task 23 — Version 2.0.0 et release GitHub

Date : 2026-10-05 · branche `master` · release publiée.

## Où est le numéro de version

**Une seule source de vérité : `compteur/__init__.py`.**

```python
VERSION = "2.0.0"
```

Accompagnée de la règle de numérotation en commentaire `#:` :

| Palier | Déclencheur | Exemple |
|---|---|---|
| **majeur** | la façon dont le logiciel s'installe, ou son API, change | v2 : torch n'est plus embarqué, il se télécharge au premier lancement |
| **mineur** | une fonctionnalité, rétrocompatible | |
| **correctif** | une correction de bug, sans changement de comportement | |

`VERSION` est exporté dans `__all__`.

## Où l'utilisateur le voit

Titre de la fenêtre, `interface/app.py:149` :

```python
self.setWindowTitle(f"Compteur de manifestation — {VERSION}")
```

Rendu : `Compteur de manifestation — 2.0.0`.

`interface/app.py:51` fait `from compteur import VERSION`. **Aucun littéral de
version n'existe ailleurs** dans `interface/` ou `compteur/` — vérifié au grep.
`tools/verifier_lancement_exe.py:60` garde `TITRE_ATTENDU` comme préfixe court :
la vérification porte sur le logiciel, pas sur le format du titre, et ne casse
donc pas au prochain changement de format.

## Test

Un seul test : `tests/test_app.py::test_fenetre_a_le_titre_avec_le_numero_de_version`
(remplace l'ancien `test_fenetre_a_le_titre_attendu`, qui codait en dur le
titre sans version). Il lit `VERSION` depuis le moteur et compare le titre à
`f"Compteur de manifestation — {VERSION}"` — donc une redéfinition divergente
échoue au lieu de passer en silence.

**521 tests passent, 3 xfailed.** Rien de cassé.

## Release GitHub

- **URL** : https://github.com/galaxiel/crowd-counter-v2/releases/tag/v2.0.0
- **Asset** : `CompteurManifestationV2-v2.0.0-win64.zip`
- **Taille exacte** : **167 562 786 octets** (~159,8 Mio / 167,6 Mo)
- **sha256** : `d5f4609b11587be3091690a07a4f9b82c3d5a03749aeba085d179dfef39ffc35`
- **Tag** : `v2.0.0` → `b24d8cc` (vérifié via l'API git/ref : SHA exacte
  `b24d8cc4eb4ed637c0a496bb477f13ed5fcd352a`)

### Contenu du zip

615 entrées, 0 séparateur backslash (archive conforme, extractible sous Linux
et macOS), `testzip()` = None.

```
CompteurManifestationV2.exe   10 289 059
medium.pt                     52 020 374
nano.pt                        6 250 350
pyi_mode.txt                          8
_internal/                     611 fichiers
```

Les modèles sont **à côté** de l'exécutable, comme dans `dist/CompteurManifestation/`.
`yolov8n-head.pt` est **absent** — choix de l'utilisateur, tranché.

### Message de release (anglais)

Explique le changement v1 → v2 (installation), donne les trois conséquences
pratiques (premier lancement long, ~2,5 Go à télécharger, dossiers v1/v2 non
interchangeables), les instructions de décompression, puis une section
**Known limitations** honnête : Windows only, NVIDIA GPU requis, ligne tracée
à la main, occlusion non résolue, **aucun benchmark annoté** — le compteur est
une estimation, pas une mesure.

## Notes de méthode

### `Compress-Archive` (PowerShell 5.1) produit un zip non conforme

Première tentative : `Compress-Archive` a écrit 164 225 011 octets avec des
séparateurs **backslash** (`_internal\ultralytics\...`). C'est le comportement
de `System.IO.Compression` sous .NET Framework : l'archive s'extrait mal sous
Linux et macOS, où le backslash n'est pas un séparateur de chemin. Un asset de
release downloads par des gens qui ne sont pas tous sous Windows : c'était un
vrai défaut, pas une question de principe.

Remplacé par `pwsh` 7 (`[ZipFile]::CreateFromDirectory`), qui écrit des
séparateurs POSIX. D'où la différence de taille (167 Mo contre 164 Mo) :
meilleure compression, pas contenu différent.

`tar` (GNU tar 1.35, celui de Git Bash) **ne sait pas écrire de zip** —
`tar -a -c -f x.zip` sort un fichier que `zipfile` refuse. Écarté.

### Le tag pointait au mauvais commit

`gh release create --target b24d8cc` a créé le tag sur `12a6188` (HEAD de
`master`), avec `targetCommitish: master` dans les métadonnées — `gh` résout
la branche courante quand le nom de tag n'existe pas encore, et le SHA given
n'a pas servi.

`gh release edit --target b24d8cc` renvoie **HTTP 422 Validation Failed** :
l'API GitHub refuse un `target_commitish` qui n'est ni branche ni nom de tag.

Corrigé en déplaçant le tag à la main (`git tag -f` + `git push -f --tags`),
puis revérifié sur l'API. **Conséquence assumée** : le tag `v2.0.0` pointe sur
`b24d8cc`, qui ne contient **pas encore** la constante `VERSION` (commit
`12a6188`, postérieur). C'est ce que la consigne demandait — « tags sur
`b24d8cc` » — et c'est cohérent : `b24d8cc` est le commit qui a *produit*
l'exécutable publié. Mais le source à ce tag affiche encore l'ancien titre sans
version. Noté comme limite connue dans le message de release, à corriger en
2.0.1 ou par un tag réécrit si l'utilisateur préfère.

## Fichiers modifiés / créés

| Fichier | Changement |
|---|---|
| `compteur/__init__.py` | `VERSION = "2.0.0"` + règle de numérotation, `__all__` |
| `interface/app.py` | `from compteur import VERSION` ; titre avec version |
| `tests/test_app.py` | `test_fenetre_a_le_titre_avec_le_numero_de_version` |
| `tools/verifier_lancement_exe.py` | commentaire : `TITRE_ATTENDU` est un préfixe |
| `docs/design/task-23-version-release-report.md` | ce rapport |

Commit `12a6188` poussé sur `master`. Aucun exécutable reconstruit ; `b24d8cc`
intact.

## Préoccupations

1. **Le tag `v2.0.0` ne contient pas la version.** Demandé explicitement, donc
   fait. Mais un utilisateur qui clone le tag et lit `interface/app.py` ne voit
   pas `VERSION`. Signalé dans la release ; à trancher (retag ou 2.0.1).
2. **L'exécutable publié affiche l'ancien titre.** Il a été compilé avant le
   commit `12a6188` : dans la vraie fenêtre, un utilisateur de la v2 release
   voit `Compteur de manifestation` sans « — 2.0.0 ». Il faut un **rebuild**
   de la v2 pour que le titre à l'écran corresponde. Non fait ici : la
   consigne disait de ne pas reconstruire d'exécutable. C'est le point à
   traiter en priorité.
3. **Pas de checksum dans les notes de release** — le sha256 est dans les
   métadonnées de l'asset GitHub, mais pas recopié dans le texte.
4. **`--target` de `gh release` n'est pas fiable** pour un SHA (cf. § précédent).
   Passer par `git tag -f` à l'avenir.
5. **`Compress-Archive` est à éviter** pour tout asset public.