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
- [Performance](#performance)
- [Building the executable](#building-the-executable)
- [Using the software](#using-the-software)
- [Design decisions and scope](#design-decisions-and-scope)
- [Known limitations](#known-limitations)
- [Architecture](#architecture)
- [Tests](#tests)
- [Further reading](#further-reading)

---

## What the software does

1. **Detection** — YOLO (Ultralytics) spots heads in each frame. The model is
   a parameter, never hardcoded.
2. **Tracking** — IoU association: each head keeps a stable identity from one
   frame to the next.
3. **Counting** — when an identity moves from one side of the line to the
   other, in the chosen direction, that's a count. Once counted, the identity
   is released: it can never be counted again.
4. **Reporting** — a summary at the end with the total, the rate, and a curve.

The model only looks inside the detection band (see below), and the default
model (`medium.pt`) is a head detector trained on SCUT-HEAD. On the reference
video it finds 29 px heads where `yolov8n-head.pt` finds 21 px ones — 38 %
more real crossings.

## Running without installing anything

`dist/CompteurManifestationV2/` is self-contained: Python does not need to be
installed on the machine.

1. Copy the whole `dist/CompteurManifestationV2/` folder **whole** to wherever
   you want (a USB stick, a desktop, `C:\Program Files\`) — the `.exe` alone
   is not enough, it needs its neighbouring DLLs.
2. Double-click `CompteurManifestationV2.exe`.

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
`crowd-counter-v2.spec`; the workaround (for a CPU install that fits under 2 GB)
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
| `dist\CompteurManifestationV2\CompteurManifestationV2.exe` | in `dist\CompteurManifestationV2\` |

Double-clicking the executable makes the executable's own folder the current
folder: weights dropped into `dist\CompteurManifestationV2\` therefore appear in
the dropdown without any configuration.

The field stays **editable**: you can type an absolute path to a `.pt` that is
in neither folder.

After a rebuild:

```bash
python tools/copier_modeles.py
```

copies the weights from the root to `dist/`. `--propre` only copies what is
missing.

## Performance

Measured on the reference video, 4-minute scene:

| Device | Throughput | 4-minute video |
|---|---|---|
| CUDA (RTX 4070 SUPER) | ~27 img/s | ~9 min |
| CPU | ~7 img/s | ~35 min |

AMD cards run in CPU mode — a deliberate trade-off. The full story (device
choice in the app, why the first launch downloads ~2.4 GB of CUDA, the
PyInstaller packaging traps) is in
[`docs/BUILDING.md`](docs/BUILDING.md#performance-and-gpu).

## What the download looks like

The executable does not contain PyTorch; it fetches the right version on first
launch and caches it in `%LOCALAPPDATA%\CompteurManifestation\`. Later launches
start normally.

| | |
|---|---|
| Download | **160 MB** |
| Needs internet | yes, on first launch |
| Administrator rights | no |
| First launch | downloads PyTorch (~2.4 GB CUDA, ~200 MB CPU) |
| Windows version | 10 or later |

Download it from the [releases page](../../releases).

> The first launch needs internet. A machine that has neither an NVIDIA card nor
> a working download path falls back to CPU (~200 MB), which works but is slow.

## Building the executable

The whole procedure lives in [`docs/BUILDING.md`](docs/BUILDING.md) — clean
environment rule, exact pip commands (the torch wheel index matters), post-
build verification. The shape of it:

```bash
python -m venv build-env
build-env\Scripts\python -m PyInstaller --clean --workpath build/v2 crowd-counter-v2.spec
build-env\Scripts\python tools/copier_modeles.py
```

Read the doc before building: the clean-environment rule is not optional, and
a polluted build environment is the difference between a lean executable and
a bloated one.

## Using the software

Load a video, draw the line (two clicks), pick the counted direction, run.
The full walkthrough — the direction reminder when the count stays at zero,
the sensitivity settings, the end-of-analysis summary, audit mode, and how to
read the green boxes that make a mistake visible — is in
[`docs/USAGE.md`](docs/USAGE.md).

## Design decisions and scope

### What this is for

Counting the people crossing a virtual line, from a video feed coming from a
**fixed camera in an elevated position** filming a street. The camera never
moves. Expected order of magnitude: **200 to 300 people** in total at a
demonstration, walking past spaced out rather than packed.

### What is deliberately out of scope

These are not "not yet implemented" — they were considered and ruled out:

- **No person identification, no faces, no recognition.** The counter counts
  heads. It has no idea who they belong to, and adding that would change the
  legal and privacy profile entirely.
- **No multi-camera mode.** One camera, one line.
- **No web application, server, or API.** The engine is ready for one — see
  below — but shipping it is a separate decision with different costs.
- **No densitometry.** This targets a marching flow, not a standing crowd of
  5,000 people. The two problems look similar and are not: density is
  estimated from area coverage, which says nothing about how many people walk
  through a given line.
- **No model training.** Pre-trained local models only.

### The detection band

The model only looks inside a band **250 px wide on each side of the line** —
500 px in total, symmetrical. Everything outside it is dimmed on screen and
never reaches the detector.

This is the core design decision, and it cuts both ways:

- **It improves the count.** People far from the line are mostly noise for a
  line counter — they crowd together, their boxes merge, and their identities
  get confused. Confining the tracker to the people actually approaching the
  line gives it fewer targets to get right, and it gets them right.
- **It is what makes the video display legible.** The dimmed region tells the
  operator immediately where the counting happens, instead of boxes appearing
  and vanishing with no visible reason.

The band used to be lopsided (200 px before the line, 100 px after); the
operator chose to make it symmetrical at 250 px so the counting zone is easier
to see. The tracker still needs room *before* the line — an identity is built
over several frames, and that is what makes the crossing reliable — but after
the crossing the person keeps being followed for display: the green box
follows them until they leave the band, then disappears.

Measured on the reference video, band widths from 200 px to 600 px all produce
a higher count than analysing the whole frame; wider does not help further.

The band moves with the line: click anywhere in it and drag, and the line
comes along. Both are locked once the analysis starts — a line that moved
mid-analysis would silently produce a meaningless total.

> **Limitation:** on a diagonal line the band becomes a bounding rectangle
> that covers most of the frame, so the benefit disappears. It applies to
> vertical and horizontal lines, which is the intended use.

### Why a head detector, and why not ByteTrack

The original attempt with a general-purpose "person" detector performed badly
on demonstration footage: banners and placards broke it, and adjacent people
merged into a single detection. A **head** detector is far more robust there,
because a head stays visible above a placard.

That choice is why the default model is `medium.pt`, a SCUT-HEAD trained
detector.

The tracker went the other way. ByteTrack was tried first and rejected: at
29 px per head moving 0.69 px per frame, its association was unreliable. A
plain **IoU tracker** was measured at 0.85 inter-frame overlap and works
better. That is the opposite of the usual "use the newer, better algorithm"
instinct — it was chosen because the measurement said so.

### The engine's public contract

```python
# compteur/compteur.py
def analyser_video(
    chemin_video: str,
    config: Config,
    callback_frame: Callable[[FrameResult], None] | None = None,
) -> Resultat
```

The interface calls this live. The tests call it without a screen. A future
web API would call it on a video file. Nothing would need rewriting — which
is the entire reason the engine has no GUI dependency.

## Known limitations

**The counted direction must match the direction people walk.** With the
default settings this is the only reliable way to get a zero: on the reference
video, counting left-to-right gives 1 person while right-to-left gives 315.

The failure is quiet — the software opens, the video plays, boxes appear, and
the number stays at zero. The interface shows a reminder after a few seconds
rather than leaving you to guess, but it cannot know which way your crowd
walks. Pick the direction that matches.

Two other settings can drop the count to zero, both of which deviate from the
defaults:

- **Confirmation threshold too high for a slow-moving crowd.** *Frames de
  confirmation* is 3 by default: a track only appears after three frames. If
  people move less between frames than the detector's own noise, the identity
  can be lost before reaching that threshold. **Remedy: set it to 1.** That
  increases sensitivity, at the cost of false positives.
- **Smoothing delays the crossing until it cancels it.** *Fenêtre de
  lissage* is 1 (raw mode) by default, deliberately: the `a_traverse` test then
  reads the instantaneous position, which is the only position that "moves" at
  this scale. Beyond K=1, smoothing delays detection of the crossing by half a
  window while the anti-rebound lock still reads the **raw** position — already
  past. **Remedy: leave it at 1.** Known and documented in
  `compteur/config.py`.

  On the reference video, a head is 20 to 29 px and moves **0.69 px per
  frame** — the same order of magnitude as the detector's noise. This is why
  the defaults are conservative: the geometry of the problem, not caution for
  its own sake.

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
hooks-v2/    PyInstaller hooks (torch, torchvision, Qt)
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

The suite never downloads torch and never runs real inference — fake backends
everywhere, lazy imports in `compteur/detecteur.py` — so it is fast by design.
Iterate on one module, widen only when needed:

```bash
python -m pytest tests/test_ligne.py   # one module: seconds
python -m pytest -m "not slow"         # everything but the heavy few
python -m pytest -n auto               # full suite, parallel (pytest-xdist)
python -m pytest                       # full suite, sequential
```

The `slow` marker covers the few files that spawn real subprocesses
(`test_peripherique.py`, `test_demarrage_sans_venv.py`,
`test_telechargement.py`). Skip them while iterating; run the full suite
before pushing.

## Further reading

- [`docs/BUILDING.md`](docs/BUILDING.md) — rebuilding the Windows executable,
  and everything about GPU / CUDA performance.
- [`docs/USAGE.md`](docs/USAGE.md) — day-to-day use: line, direction,
  sensitivity, summary, audit mode, green boxes.
- [`docs/design/DESIGN.md`](docs/design/DESIGN.md) — the original design
  specification: use case, algorithm, engine contract, settings rationale.
- [`docs/design/IMPLEMENTATION-PLAN.md`](docs/design/IMPLEMENTATION-PLAN.md) —
  the task-by-task build log (in French). Long, and written as a work journal
  rather than as documentation; read it only if you want to know *why* a
  decision was taken, including the ones that were later reversed.

## Support

This project is free and open source. If it saves you time, a coffee is
welcome — the button in the app (or [this
link](https://buymeacoffee.com/galaxiel)) opens the page directly.

## Licence

The code in this repository (the `compteur/`, `interface/` and `tools/`
packages, `main.py`, tests) is released under the [MIT
License](LICENSE) — open to everyone, all modifications, commercial use
included. The only requirement is keeping the copyright notice.

The **pre-trained models** (`medium.pt`, `yolov8n-head.pt`, …) are **not**
part of the repository and are **not** covered by the MIT license: they come
from their own sources (Ultralytics YOLO and SCUT-HEAD) and keep their own
licenses. They are downloaded or copied separately by the user.