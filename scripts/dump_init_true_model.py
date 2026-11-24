#!/usr/bin/env python3
"""Dump the TRUE model of the training example that seeds the 'trained'
latent init of the inversion campaign.

Why: the audience notebook's traverse figure opens with the true model of
the example whose latent initializes the inversion. That image exists in
no MLflow artifact (the TRAIN context figures sample 3 random training
examples per run), so it must be rendered from the dataset once.

Chain of custody (verify before trusting the output):
  - the inversion resolves latent_init='trained' as
        latent = encoder(cropped_x[latent_init_idx])   # latent_init_idx = 0
    (src/processing/task_processing.py, `if grid['latent_init'] == "trained"`)
  - cropped_x follows BaseBuilder._get_filelist semantics: for each file
    family, filelist = sorted(glob(<prefix>)), then the split indexes pick
    positions IN THAT FAMILY'S OWN FILELIST
    (src/data/data_builder.py, `_get_filelist(prefix, indexes_list)`)
  - the split is the seed-79 one shared by refit 9247757e and all its
    inversion runs (logged as _x_train_indexes in BOTH runs' params —
    cross-checked): 900 train examples, train_indexes[0] = 129, and
    test_indexes[1/10/20] = 476/696/5 (the campaign's inverted targets,
    which validates the parsing).
  - FAMILY: the true (sharp, full-resolution) models are `model_[0-9]*.npy`;
    `model_suave_*` is the SMOOTHED highres family (the earlier render that
    looked like "a low-resolution version" was this wrong family).

Run INSIDE the container (the dataset is bind-mounted at GEMINI_DATA_ROOT):

    python scripts/dump_init_true_model.py

Writes notebooks/figures/best_run/init_true.png (tracked). Sanity checks:
  - the image should be SHARP (thin layers / faults), unlike the smoothed
    model_suave render;
  - file counts of the model_, model_suave_ and input_suave_ families are
    printed — they must agree for the position mapping to be meaningful.
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

families = {
    "true (model_)": f"{DATA}/model_[0-9]*.npy",
    "smoothed highres (model_suave_)": f"{DATA}/model_suave_[0-9]*.npy",
    "seismic input (input_suave_)": f"{DATA}/input_suave_[0-9]*.npy",
}
counts = {}
for name, pattern in families.items():
    n = len(glob(pattern))
    counts[name] = n
    print(f"{name:34s}: {n} files")
if len(set(counts.values())) != 1:
    print("WARNING: family file counts differ — the sorted-position mapping "
          "between families may not be 1:1; verify before use.")

files = sorted(glob(families["true (model_)"]))
if not files:
    sys.exit(f"no model_[0-9]*.npy under {DATA} — adjust DATA or run where "
             "the dataset is mounted")
if FILE_POS >= len(files):
    sys.exit(f"file position {FILE_POS} out of range ({len(files)} files)")

f0 = files[FILE_POS]
print(f"\ntrue-model filelist[{FILE_POS}] = {f0}")
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
ax.set_title("TRUE model — train example,\nfile-list position 129")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
print("saved:", OUT)
