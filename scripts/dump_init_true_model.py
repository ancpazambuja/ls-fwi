#!/usr/bin/env python3
"""Dump the true (highres) model of the training example that seeds the
'trained' latent init of the inversion campaign.

Why: the audience notebook's traverse figure opens with the true model of
the example whose latent initializes the inversion. That image exists in
no MLflow artifact (the TRAIN context figures sample 3 random training
examples per run), so it must be rendered from the dataset once.

Chain of custody (verify before trusting the output):
  - the inversion resolves latent_init='trained' as
        latent = encoder(cropped_x[latent_init_idx])   # latent_init_idx = 0
    (src/processing/task_processing.py, `if grid['latent_init'] == "trained"`)
  - cropped_x follows BaseBuilder._get_filelist semantics:
        filelist = sorted(glob(<data>/model_suave_[0-9]*.npy))
        train y  = [filelist[i] for i in train_indexes]
    (src/data/data_builder.py)
  - for the deep dataset's 900-example train split (seed-79 family),
    train_indexes[0] = 129 (logged as _x_train_indexes in the run params),
    so the initial latent encodes filelist[129].

Run INSIDE the container (the dataset is bind-mounted at GEMINI_DATA_ROOT):

    python scripts/dump_init_true_model.py

Writes notebooks/figures/best_run/init_true.png (tracked). Sanity check:
the rendered structure should resemble the traverse figure's 'initial
decode' column — the initial latent is this model's encoding, so the AE
reconstructs approximately it.
"""
import sys
from glob import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "figures" / "best_run" / "init_true.png"
DATA = Path("/gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/"
            "deeptomograph_data/data_new_and_bigger")
FILE_POS = 129          # train_indexes[0]: file-list position of the init example

files = sorted(glob(str(DATA / "model_suave_[0-9]*.npy")))
print(f"model_suave files found: {len(files)}")
if not files:
    sys.exit(f"no model_suave_*.npy under {DATA} — adjust DATA or run where "
             "the dataset is mounted")
if FILE_POS >= len(files):
    sys.exit(f"file position {FILE_POS} out of range ({len(files)} files) — "
             "the file-list layout differs from the assumed one; use the src "
             "data pipeline instead")

f0 = files[FILE_POS]
print(f"filelist[{FILE_POS}] = {f0}")
y0 = np.load(f0)
print(f"shape: {y0.shape}, range: {y0.min():.3f}..{y0.max():.3f}")
if y0.ndim != 2:
    sys.exit("loaded file is not a single 2D model (batched layout?) — the "
             "simple file indexing does not apply; use the src data pipeline")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402  (after Agg)

fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(y0, cmap="turbo", aspect="auto")   # turbo = DEFAULT_VELOCITY_CMAP
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("true (highres) model\ntrain example, file-list position 129")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
print("saved:", OUT)
