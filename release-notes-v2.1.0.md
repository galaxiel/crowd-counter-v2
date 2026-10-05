# Compteur de manifestation v2.1.0

**Windows only · NVIDIA GPU recommended** — downloads PyTorch (~2.4 GB CUDA) on first launch, then works offline.

## What's new in 2.1.0

- **Drag & drop** — move the detection line (and its band) with the mouse; the band and line follow together.
- **Counted boxes fade out** — after a person crosses the line, their green box stays visible for a short fade instead of disappearing right on the border, so you can see where the counted person went.
- **Symmetric detection band** — the band now extends evenly around the line (250 px each side).
- **Removed the presentation speed selector** — analysis always runs at maximum speed.
- **Buy me a coffee button** — native button in the interface.
- **MIT license** on the code; model licenses noted in the README.
- Title bar now shows the version: **Compteur de manifestation — 2.1.0**.

## Installation

1. Download `CompteurManifestationV2-v2.1.0-win64.zip`.
2. Extract the whole folder somewhere (keep the `.exe` next to `_internal/` and the `.pt` files).
3. Run `CompteurManifestationV2.exe`.
4. **First launch needs internet** — it downloads and installs PyTorch (~2.4 GB CUDA, ~200 MB CPU) into `%LOCALAPPDATA%\CompteurManifestation\`. Later launches are offline and fast.

## Verify the download

sha256: `31c693f084dc63bea0a2c68c551335ef1e45b3bad59f64003d24b4e2113c0f00`

## Known limitations

- **Windows only**, version 10 or later. NVIDIA GPU required for fast analysis (CUDA); a machine without a card falls back to CPU, which works but is slow.
- The counting line is **drawn by hand** (two clicks) — on a fixed camera above the scene, as intended.
- Occlusion is not resolved: people fully hidden behind others are not counted.
- The counter is an **estimate, not a measurement** — there is no annotated benchmark, only manual counts on test videos.
