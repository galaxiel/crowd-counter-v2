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

## Two builds: which one do you want?

| | v1 | v2 |
|---|---|---|
| Download | **4.3 GB** | **160 MB** |
| Needs internet | no | yes, on first launch |
| Administrator rights | no | no |
| First launch | immediate | downloads PyTorch (~2.4 GB CUDA, ~200 MB CPU) |
| Windows version | any | 10 or later |

**v2 is the one to try.** The executable does not contain PyTorch; it fetches
the right version on first launch and caches it in
`%LOCALAPPDATA%\CompteurManifestation\`. Later launches start normally.

v1 still exists and still works — it needs no network at all, which matters if
you are analysing videos on a machine with no connection.

Download v2 from the [releases page](../../releases).

> The first launch needs internet. A machine that has neither an NVIDIA card nor
> a working download path falls back to CPU (~200 MB), which works but is slow.

## Building the executable

**Build from a clean virtual environment.** This is not advice — it is the
difference between a 4.3 GB executable and a 9.7 GB one.

Ultralytics imports its optional inference backends conditionally. If they are
installed, PyInstaller bundles them: TensorRT (1.5 GB), TensorFlow (1.1 GB),
ONNX Runtime (741 MB), xformers (413 MB) and bitsandbytes (213 MB). **None of
them is used** — this project only ever runs PyTorch. That is 5.5 GB of dead
weight in the output folder.

Excluding them from `crowd-counter.spec` does not work: PyInstaller crashes
while reading one of their hooks. The reliable approach is to make them
unavailable in the first place — a virtual environment that never had them.

### Procedure

```bash
# 1. clean environment, nothing else in it
python -m venv build-env

# 2. torch CUDA + matching torchvision — see the warning below
build-env\Scripts\python -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 ^
    --index-url https://download.pytorch.org/whl/cu126

# 3. the rest
build-env\Scripts\python -m pip install ultralytics opencv-python PySide6 pyinstaller pytest

# 4. build
build-env\Scripts\python -m PyInstaller --clean crowd-counter.spec
build-env\Scripts\python tools/copier_modeles.py
```

**torch and torchvision must come from the same index.** A CPU-only
torchvision next to a CUDA torch crashes `torchvision::nms` on CUDA. This is
the single most common packaging mistake here.

Result: `dist/CompteurManifestation/` — the executable, its `_internal/` folder
and the weights next to it. **4.3 GB**, launching in ~1.3 s.

### Verify it before shipping it

```bash
build-env\Scripts\python tools/verifier_lancement_exe.py
```

Launches the executable the way a user would, checks that a window titled
*Compteur de manifestation* appears, that the weights are next to it, and that
the console output does not contain the silent `Config` fallback. A build that
succeeds does not prove that a double-click works — that is exactly what this
tool covers.

> If the folder is locked during a rebuild (`Device or resource busy`), the
> executable from the previous build is still running. Close it first.

### Why it cannot go below ~4 GB

3.8 GB of the 4.3 GB are torch's CUDA DLLs — `torch_cuda.dll` alone is 1 GB,
cuBLAS 500 MB, cuDNN over 1 GB. As long as CUDA ships inside the executable,
that is the floor. Going lower means not shipping torch at all and downloading
it on first launch; see `HISTOIRE.md`.

## Using the software

1. **Load the video** — *Charger la vidéo* button.
2. **Draw the line** — *Tracer la ligne* button, then two clicks on the image,
   at the top and bottom of the line you care about. Drawing is "armed": a
   stray click outside that gesture is ignored. The detection band follows the
   line wherever you put it.
3. **Set the direction** — in *Ligne de franchissement*, choose the counted
   direction. The green arrow shows the active direction. **If the count stays
   at zero after a few seconds, this is almost always why** — the software
   shows a reminder rather than failing silently.
4. **Adjust the sensitivity** — the parameter that matters most is *Frames de
   confirmation*: 1 counts immediately (sensitive to false positives), 5 only
   validates a person seen over several consecutive frames. Every parameter has
   a tooltip explaining what it does and which value suits this scene; the
   panel's *Afficher l'aide* button shows them all at once.
5. **Run** — the counter updates continuously, the line flashes on every
   crossing. The band and the line cannot be moved once the run starts.
6. **Read the summary** — when the analysis stops, a section appears between
   the status bar and the settings: total, duration, average rate, peak rate
   and a curve of people per minute.

### The end-of-analysis summary

The summary reports only what was measured:

- **Total** — people counted in the chosen direction.
- **Average rate** — total over the analysed duration.
- **Peak rate** — the busiest 60 s window, with when it happened. On analyses
  shorter than a minute the window shrinks, and the figure is reported as
  `12 pers / 40 s` rather than extrapolated to a rate that was never measured.

There is **no automatic error rate**. There is no ground truth without manual
annotation, and the software does not invent one. The only way to know how far
a total is from the truth is to count a segment by hand and compare.

### Audit mode

During a replay, the *Audit* panel lets you flag errors by eye. The error rate
shown is only valid for the portion you actually checked: it is a
**measurement**, not an automatic estimate.

### Reading the green boxes

When a person is counted, their box turns **green** for a few seconds, then
fades out as they leave the band on the far side of the line.

This is not decoration — **it is the only way to see a mistake without ground
truth.** Nobody can recount a demonstration by hand, and the counter will never
say "I missed one": the total is just a number, and a wrong number looks exactly
like a right one. But a person walking across the line *without* their box
turning green is a visible miss. Twenty seconds of watching tells you whether
the count can be trusted, and nothing else on screen tells you that.

So when you check a result, don't only read the total. Watch the line for a
while and count the green boxes yourself. If a head crosses and stays amber, you
have found a miss — and you know roughly how far off the number is.

## Design decisions and scope

### What this is for

Counting the people crossing a virtual line, from a video feed coming from a
**fixed camera in an elevated position** filming a street. The camera never
moves. Expected order of magnitude: **200 to 300 people** in total at a
demonstration, walking past spaced out rather than packed.

### What is deliberately out of scope (v1)

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

The model only looks inside a band **200 px wide before the line and 100 px
after it** — 300 px in total, deliberately lopsided. Everything outside it is
dimmed on screen and never reaches the detector.

This is the core design decision, and it cuts both ways:

- **It improves the count.** People far from the line are mostly noise for a
  line counter — they crowd together, their boxes merge, and their identities
  get confused. Confining the tracker to the people actually approaching the
  line gives it fewer targets to get right, and it gets them right.
- **It is what makes the video display legible.** The dimmed region tells the
  operator immediately where the counting happens, instead of boxes appearing
  and vanishing with no visible reason.

**Why the band is not centred.** The tracker needs room *before* the line: an
identity is built over several frames, from several images, and that is what
makes the crossing reliable. After the line the count is already made — a
crossed person can never be counted again — so the 100 px that only serve to
show them walking away are enough. A centred 200 px band spent half of its
width on a region that cannot change the result.

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

## Further reading

- [`docs/design/DESIGN.md`](docs/design/DESIGN.md) — the original design
  specification: use case, algorithm, engine contract, settings rationale.
- [`docs/design/IMPLEMENTATION-PLAN.md`](docs/design/IMPLEMENTATION-PLAN.md) —
  the task-by-task build log (in French). Long, and written as a work journal
  rather than as documentation; read it only if you want to know *why* a
  decision was taken, including the ones that were later reversed.

## Licence

To be defined by the author.