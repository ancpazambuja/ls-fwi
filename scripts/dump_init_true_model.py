#!/usr/bin/env python3
"""Dump the TRUE model of the training example that seeds the 'trained'
latent init of the inversion campaign.

Why: the audience notebook's traverse figure opens with the true model of
the example whose latent initializes the inversion. That image exists in
no MLflow artifact (the TRAIN context figures sample 3 random training
examples per run), so it must be rendered from the dataset once.

Chain of custody — resolved the hard way, now pipeline-exact:
  - scripts/run_inversion.sh launches every campaign inversion with
    LATENT_IDX=10 (its default), overriding the CLI default of 0;
  - task_processing resolves latent_init='trained' as
        latent = encoder(cropped_x[latent_init_idx])
    so the initial latent encodes train-split position 10;
  - the split is the seed-79 one shared by refit 9247757e and all its
    inversion runs: the run params log _x_train_indexes with
    train_indexes[10] = 182 (dataset example 182), and
    test_indexes[1/10/20] = 476/696/5 (the campaign's inverted targets —
    validates the parsing).
  - THIS script does not re-implement any of that: it calls the repo's own
    load_data_and_models() (the exact entry point the apps use) with the
    same seed/test_perc, asserts the split matches the logged one, and
    renders train_models.get_true_models()[LATENT_IDX].

Run INSIDE the container (the dataset is bind-mounted at GEMINI_DATA_ROOT;
the builder loads all model families, so it needs some time and memory):

    python scripts/dump_init_true_model.py

Writes notebooks/figures/best_run/init_true.png (tracked). Sanity check:
the rendered structure should resemble the traverse figure's 'initial
decode' column — the initial latent encodes this model.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from src.apps.global_vars import load_data_and_models  # noqa: E402

LATENT_IDX = 10          # scripts/run_inversion.sh default (LATENT_IDX=10)
SEED = 79                # the best ML model's refit seed (encdec 5x5, seed 79)
EXPECTED_LOGGED = 182    # _x_train_indexes[10] logged in the best run's params

(data_basedir, dirs, train_data, test_data,
 train_models, test_models, train_indexes, test_indexes, newer) = \
    load_data_and_models(data='deep', data_type='deep', data_resolution='high',
                         data_max_samples=None, test_perc=0.1, seed=SEED)

print(f"train split positions: {len(train_indexes)} | "
      f"train_indexes[{LATENT_IDX}] = {train_indexes[LATENT_IDX]}")
if int(train_indexes[LATENT_IDX]) != EXPECTED_LOGGED:
    sys.exit(f"split mismatch: the pipeline reproduced train_indexes["
             f"{LATENT_IDX}] = {train_indexes[LATENT_IDX]} but the best run's "
             f"logged params say {EXPECTED_LOGGED} — wrong seed/test_perc?")

true_models = train_models.get_true_models().get_models()
y0 = true_models[LATENT_IDX]
print(f"true model of train-split position {LATENT_IDX}: "
      f"shape {y0.shape}, range {y0.min():.3f}..{y0.max():.3f}")

fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(y0, cmap="turbo", aspect="auto")   # turbo = DEFAULT_VELOCITY_CMAP
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"TRUE model — train example {int(train_indexes[LATENT_IDX])}\n"
             f"(train-split position {LATENT_IDX}, seed {SEED})")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
OUT = REPO / "notebooks" / "figures" / "best_run" / "init_true.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
print("saved:", OUT)
