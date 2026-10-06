# Building the executable

How to rebuild the Windows executable from source, and everything about GPU /
CUDA performance. To run the application from source instead, see the
[README](../README.md#installing-and-running-from-source).

## Building the executable

**Build from a clean virtual environment.** This is not advice — it is the
difference between a lean executable and a bloated one.

Ultralytics imports its optional inference backends conditionally. If they are
installed, PyInstaller bundles them: TensorRT (1.5 GB), TensorFlow (1.1 GB),
ONNX Runtime (741 MB), xformers (413 MB) and bitsandbytes (213 MB). **None of
them is used** — this project only ever runs PyTorch. Excluding them from the
spec does not work: PyInstaller crashes while reading one of their hooks. The
reliable approach is to make them unavailable in the first place — a virtual
environment that never had them.

```bash
# 1. clean environment, nothing else in it
python -m venv build-env

# 2. torch CUDA + matching torchvision MUST come from the same index
build-env\Scripts\python -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 ^
    --index-url https://download.pytorch.org/whl/cu126

# 3. the rest — from PyPI, never from the torch index
build-env\Scripts\python -m pip install ultralytics opencv-python PySide6 pyinstaller pytest

# 4. build — note the workpath
build-env\Scripts\python -m PyInstaller --clean --workpath build/v2 crowd-counter-v2.spec
build-env\Scripts\python tools/copier_modeles.py
```

**torch and torchvision must come from the same index.** A CPU-only
torchvision next to a CUDA torch crashes `torchvision::nms` on CUDA. This is
the single most common packaging mistake here.

Result: `dist/CompteurManifestationV2/` — ~290 MB without the weights
(~360–400 MB with them). torch is NOT bundled; it is downloaded into
`%LOCALAPPDATA%\CompteurManifestation\` on first launch. The `--workpath
build/v2` keeps PyInstaller's intermediate `.toc` files in a workpath of their
own — sharing it with other builds mixes stale state in.

### Verify the build before shipping it

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
will take hours. That is why the first launch downloads the CUDA wheel
(~2.4 GB): `torch_cuda.dll` alone weighs 1 GB, `cublasLt` 500 MB, cuDNN over
1 GB.

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
`crowd-counter-v2.spec` and `hooks-v2/hook-torchvision.py`):

- torchvision's NMS extension is called `_C.stable` since 0.29 and is not a
  Python module: without a dedicated hook, inference stops on `Couldn't load
  custom C++ ops`;
- CUDA DLLs are loaded by `torch.ops.load_library()`, never by an `import`:
  PyInstaller does not find them on its own.
