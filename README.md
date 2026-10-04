# Crowd Counter

Counts the people crossing a virtual line in a fixed-camera video. Counting
engine with no GUI (`compteur/`), PySide6 window on top (`interface/`).

One scene, one line, one number: how many people crossed, in which direction,
at what rate. Nothing more — no facial recognition, no identification, no
density estimation.

---

## Table of contents

- [What the software does](#what-the-software-does)
- [Running without installing anything](#running-without-installing-anything)
- [Installing and running from source](#installing-and-running-from-source)
- [Where to put the models](#where-to-put-the-models)
- [Performance and GPU](#performance-and-gpu)
  - [Choosing where the compute happens](#choosing-where-the-compute-happens)
- [Building the executable](#building-the-executable)
- [Using the software](#using-the-software)
- [Exporting](#exporting)
- [Known limitations](#known-limitations)
- [Architecture](#architecture)
- [Tests](#tests)

---

## What the software does

1. **Detection** — YOLO (Ultralytics) spots heads in each frame. The model is
   a parameter, never hardcoded.
2. **Tracking** — IoU association: each head keeps a stable identity from one
   frame to the next.
3. **Counting** — when an identity moves from one side of the line to the
   other, in the chosen direction, that's a count.
4. **Reporting** — one big number on screen, a CSV and a JSON on export.

The default model (`medium.pt`) is a head detector trained on SCUT-HEAD. On
the reference video it finds 29 px heads where `yolov8n-head.pt` finds 21 px
ones — 38 % more real crossings.

## Running without installing anything

`dist/CompteurManifestation/` is self-contained: Python does not need to be
installed on the machine.

1. Copy the whole `dist/CompteurManifestation/` folder **whole** to wherever
   you want (a USB stick, a desktop, `C:\Program Files\`) — the `.exe` alone
   is not enough, it needs its neighbouring DLLs.
2. Double-click `CompteurManifestation.exe`.

A black console window opens next to the application: that is intentional. It
shows the log and, if something goes wrong, the traceback. Closing the
software closes the console too.

Startup is **immediate**: nothing is decompressed, everything is already on
disk.

### Why a folder and not a single `.exe`

PyInstaller can bundle everything into one file ("onefile"). With torch CUDA
this mode **does not work**: the archive format uses signed 32-bit offsets,
and the CUDA DLLs alone weigh 3.8 GB. The binary builds without error, then
**crashes at launch**, exiting without a message. The details are in
`crowd-counter.spec`; the workaround (for a CPU install that fits under 2 GB)
is `COMPTEUR_ONEFILE=1`.

The folder mode has a direct advantage for this software: no decompressing
3 GB on every launch.

### What must sit next to the executable

```
dist/CompteurManifestation/
├── CompteurManifestation.exe
├── _internal/            <- DLLs and Python modules, don't touch
├── medium.pt             <- default model
├── nano.pt               <- light model (CPU or older machine)
└── yolov8n-head.pt      <- standard Ultralytics head detector
```

The weights are **not** inside the executable: they are parameters, not
dependencies. See [Where to put the models](#where-to-put-the-models).

## Installing and running from source

Python 3.11 or newer (tested on 3.14).

```bash
python -m venv venv
venv\Scripts\activate
python installer.py
python main.py
```

`installer.py` detects the graphics card and installs the matching version of
torch. **Do not use `pip install -r requirements.txt`** for this step: torch
is published in two wheel families, CPU and CUDA, and the package named `torch`
on PyPI does not say which one you get — it is the install that looks correct
and runs at 7 img/s.

The script checks itself without installing anything:

```bash
python installer.py --dry-run                    # shows the chosen branch
python installer.py --dry-run --sans-gpu         # verifies the CPU branch
python installer.py --dry-run --peripherique cuda # fails if there is no GPU
```

`--dry-run` installs nothing: it is the way to check what the script
understood about your machine before letting it touch `pip`.

To work on the software itself (tests, build):

```bash
python -m pip install -r requirements-dev.txt
```

## Where to put the models

The settings panel lists the `.pt` files it finds in two folders, in this
order:

1. the current folder;
2. `modeles/`.

Practical consequence:

| Where I launch it | Where I put the `.pt` files |
|---|---|
| `python main.py` from the root | at the repository root |
| `dist\CompteurManifestation\CompteurManifestation.exe` | in `dist\CompteurManifestation\` |

Double-clicking the executable makes the executable's own folder the current
folder: weights dropped into `dist\CompteurManifestation\` therefore appear in
the dropdown without any configuration.

The field stays **editable**: you can type an absolute path to a `.pt` that is
in neither folder.

After a rebuild:

```bash
python tools/copier_modeles.py
```

copies the weights from the root to `dist/`. `--propre` only copies what is
missing.

## Performance and GPU

**torch must be installed as the CUDA version.** On this machine:

```
torch 2.14.1+cu126
torchvision 0.29.1+cu126
```

That is what `installer.py` does when it detects an NVIDIA card.

### How much it changes

Measured on this machine, on the reference video:

| Device | Throughput | 4-minute video |
|---|---|---|
| CUDA (RTX 4070 SUPER) | ~27 img/s | ~9 min |
| CPU | ~7 img/s | ~35 min |

A CPU torch works, but the counter becomes unusable live, and long scenes
will take hours. The executable embeds CUDA — hence its 3 GB. `torch_cuda.dll`
alone weighs 1 GB, `cublasLt` 500 MB, cuDNN over 1 GB.

### Choosing where the compute happens

The *Calcul sur* setting, in the *Périphérique de calcul* panel:

| Choice | Effect |
|---|---|
| **Automatic** (default) | CUDA if the machine has one, CPU otherwise |
| **NVIDIA GPU (CUDA)** | The GPU is required. Without a card, the analysis **falls back to CPU with a message** — crashing in the field costs more than a slow run you can at least watch |
| **CPU only** | The processor is required even with a GPU present. This is the remedy when the GPU crashes on a particular scene |

This choice does **not** change the count: the same scene is analysed the same
way in both cases.

The *Calcul : CUDA — NVIDIA GeForce RTX 4070 SUPER* or *Calcul : CPU
uniquement* indicator is displayed **permanently** in the status bar, under
the counter. It updates as soon as a setting changes, and it reports what is
actually being used — not what was requested.

Under PyInstaller, `nvidia-smi` only queries the driver: it ignores
`CUDA_VISIBLE_DEVICES`, which is a Linux convention. To verify the CPU branch
on a machine that does have a card, the installer exposes `--sans-gpu`.

### AMD is not supported

A deliberate choice: **a single executable, built with CUDA**. A CPU-only
torch wheel weighs ~200 MB against ~3 GB for CUDA; shipping both would double
the size of the `dist/` folder for a use case that is not ours (an AMD card in
2026 is rare on the demonstration-camera market this targets). The program
therefore runs on AMD in CPU mode, at ~7 img/s — usable for a short video, not
for a whole demonstration.

### If the GPU is not being used

The application reports it in the status bar and in the console, and analyses
anyway on the CPU, very slowly.

Under PyInstaller, two traps were handled explicitly (see
`crowd-counter.spec` and `hooks/hook-torchvision.py`):

- torchvision's NMS extension is called `_C.stable` since 0.29 and is not a
  Python module: without a dedicated hook, inference stops on `Couldn't load
  custom C++ ops`;
- CUDA DLLs are loaded by `torch.ops.load_library()`, never by an `import`:
  PyInstaller does not find them on its own.

## Building the executable

```bash
python -m pip install -r requirements-dev.txt
python -m PyInstaller --clean crowd-counter.spec
python tools/copier_modeles.py
```

Result: `dist/CompteurManifestation/` — the executable, its `_internal/`
folder and the weights next to it.

The build takes **6 to 10 minutes**: PyInstaller analyses torch, which alone
contains a few thousand modules, then copies 4 GB of DLLs to disk. `upx=True`
would make compression interminable on signed NVIDIA DLLs; the `.spec`
disables it.

`--clean` empties the cache between two builds. Without it, the second build is
noticeably faster but may reuse an outdated dependency graph — after a torch
version change, use `--clean`.

### Verify that it launches

```bash
python tools/verifier_lancement_exe.py
```

Launches the executable the way a user would, checks that a window titled
*Compteur de manifestation* appears, that `medium.pt` is next to it, and that
the console output does not contain the silent `Config` fallback. A build that
succeeds does not prove that a double-click works — that is exactly what this
tool covers.

## Using the software

1. **Load the video** — *Charger la vidéo* button.
2. **Draw the line** — *Tracer la ligne* button, then two clicks on the image,
   at the top and bottom of the line you care about. Drawing is "armed": a
   stray click outside that gesture is ignored.
3. **Set the direction** — in *Ligne de franchissement*, choose the counted
   direction. The green arrow shows the active direction.
4. **Adjust the sensitivity** — the parameter that matters most is *Frames de
   confirmation*: 1 counts immediately (sensitive to false positives), 5 only
   validates a person seen over several consecutive frames.
5. **Run** — the counter updates continuously, the line flashes on every
   crossing.
6. **Export** — see below.

### Audit mode

During a replay, the *Audit* panel lets you flag errors by eye. The error rate
shown is only valid for the portion you actually checked: it is a
**measurement**, not an automatic estimate.

## Exporting

*Export* writes into the chosen folder:

- a **CSV**: one event per row (frame, timestamp, tracked identity, position,
  score);
- a **JSON**: total, rate statistics, complete configuration.

The default proposed folder is `sortie/`, relative to the current folder —
that is, `dist\sortie\` when launching the executable.

## Known limitations

**Counting returns 0 on some scenes.** This is the most important limitation,
and it does not show on screen: the software opens normally, the video plays,
the detection boxes appear, and the number stays at zero. It is not a crash,
it is a setting — or a geometric limitation of the algorithm.

What makes a crossing detectable is measured on the reference video: a head
there is 20 to 29 px and moves **0.69 px per frame**. The actual displacement
from one frame to the next is therefore the same order of magnitude as the
detector's noise — the *median* displacement measured between two consecutive
frames is 0.00 px. At that scale, two distinct people are as close to each
other as each of them is to their own position in the previous frame.

Three settings make the counter drop to zero, and the symptom is the same for
all three:

- **Movement too small + confirmation threshold too high.** *Frames de
  confirmation* is 3 by default: a track only appears in the output after
  three frames. On a flow where per-frame movement is below the detector's
  noise, the identity can be lost before reaching that threshold, or the
  three frames can all pass far from the line. **Remedy: set *Frames de
  confirmation* to 1**, and/or reduce *Épaisseur de la bande* so the crossing
  fits into fewer frames. Both increase sensitivity — at the cost of false
  positives.
- **Smoothing delays the crossing until it cancels it.** *Fenêtre de
  lissage* is 1 (raw mode) by default, and this is deliberate: the
  `a_traverse` test then reads the instantaneous position, which is the only
  position that "moves" at this scale. Beyond K=1, smoothing delays detection
  of the crossing by half a window, while the anti-rebound lock still reads
  the side of the **raw** position — already past to the other side. The lock
  then rejects, and the reference walk goes from **1 count to 0**. **Remedy:
  leave it at 1.** This default is known and documented in
  `compteur/config.py`; fixing it means making `_cotes`/`_stabilite` read the
  smoothed position, not only the crossing test.
- **Line badly placed.** `a_traverse` interpolates the exact crossing point
  and checks that it falls **along the drawn segment**. A line drawn across
  the street counts the people passing under the middle of the frame; a line
  that stops too high or too low ignores crossings outside its segment.
  Lengthen the line over the full useful height of the image.

Other limitations:

- **The error rate is never computed automatically.** There is no ground truth
  without manual annotation. Audit mode measures the gap on what you check,
  and nothing else.
- **Dense crowds outside the use case.** The counter targets a poorly
  occluded flow (street, spaced passage). When bodies overlap, heads merge
  into a single box and one person can hide two.
- **Overhead view only.** A high camera, wide framing, no backlight or rain.
  The model is a head detector: it counts heads, not whole people.
- **One direction per run.** Counting "both directions" requires two passes or
  two windows.
- **No multi-camera, no web API.** The engine is ready for an API
  (`analyser_video` is a function `video + config -> result`); the decision is
  deferred.

## Architecture

```
compteur/    pure engine, no GUI dependency — testable without a screen
interface/   PySide6 window
config/      default.json: the default values
installer.py GPU detection + torch installation (CUDA or CPU)
tools/       command-line scripts
tests/       pytest
hooks/       PyInstaller hooks (torchvision)
```

The key point: `compteur/compteur.py::Compteur` is the engine. The interface
calls it, the tests call it, and a possible web API will be able to call it
later without rewriting anything. The constraint is enforced by a test
(`tests/test_isolation_ui.py`): no module in `compteur/` imports PySide6,
Tkinter or anything else graphical.

`config/default.json` is the single source of truth for default values — the
`Config` dataclass merely reflects it. A test compares the two so they never
diverge. Under PyInstaller, this file is embedded and found via `sys._MEIPASS`
(`tests/test_config_gelee.py`).

## Tests

```bash
python -m pytest tests/ -v
```

## Licence

To be defined by the author.