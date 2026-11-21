# Experimental protocol

This document describes the *fast-fail* protocol of the reparameterized-FWI
study: validate the latent-space search on the simplest possible task before
coupling it to wave physics.

## Research question

Can optimization recover a velocity model by traversing **only the latent
space** of a frozen decoder trained as a velocity-model autoencoder? If this
direct latent recovery fails or converges too slowly, the reparameterized
approach is unlikely to work inside FWI (where the misfit is indirect), and
the idea should be abandoned early.

## Stage 0 — data

Model→model pairs (X = Y = velocity model; single channel), from the
deep-tomography research datasets; resolution selects the model family (true
/ mediumres / lowres). See [Data availability](data_availability.md).

## Stage 1 — autoencoder training

1. **HPO (grid search).** scikit-learn `GridSearchCV` (5-fold) driven by
   `src/apps/grid_search_parallel.py`, one task per GPU (`--device cuda:all`).
   Search space: learning rate and weight decay (grouped grid, Adam fixed),
   and — for TransUNet — the attention depth (`num_layers` ∈ {2, 12, 24}).
   Fixed: `nn.MSELoss` (model space), dropout 0.2, batch normalization on,
   batch size 16, cosine-annealing warm restarts, gradient clipping 1.0.
   Every configuration is an MLflow run.
2. **Refit.** The best configuration (identified by its MLflow run id) is
   retrained on the full dataset (`src/apps/refit.py`): 90/10 holdout, early
   stopping (patience 300), best weights checkpointed next to the run name.
   The refit run stores all architecture attributes, consumed by the
   downstream stages.

## Stage 2 — latent-space characterization (`src/apps/inference.py`)

The refit checkpoint is loaded and split into `Encoder` (model → latent) and
`Generator` (latent → model); the latent has shape `(1, 512, 21, 25)` for the
deep dataset. The following diagnostics are logged as MLflow figures:

- **Round trips:** autoencoder reconstruction and generator round-trip on the
  test set (MSE / MAE / SSIM).
- **Latent geometry:** cosine-distance matrix over test codes; interpolation
  between the closest, farthest, and a random pair — within the segment
  (α ∈ [0, 1]) and extrapolated (α ∈ [−1, 2]).
- **Direction traversal:** decoder response along z(α) = α·p for encoded,
  test, and random directions.

These probe the *smoothness* and *coverage* of the learned manifold — the
property the inversion relies on.

## Stage 3 — latent recovery (`src/apps/inversion.py`)

- The `Generator` is frozen (`requires_grad = False`) and pinned to **eval
  mode** during optimization (dropout off, BatchNorm using the checkpoint's
  running statistics); the latent code z is the only trainable object
  (`GeneratorInversion`).
- Inversion targets: test-set velocity models (currently indices [1, 10, 20]).
- Initializations: `random` (U[0,1)), `randomn` (N(0,1)), `zeros`, and
  `trained` (encoder applied to a *training* example — in-distribution,
  target-agnostic warm start).
- Optimizer: Adam (default lr 0.1), gradient clipping 1.0, cosine-annealing
  warm restarts; `max_epochs` iterations (default 130).
- Loss: model-space MSE between G(z) and the target model.
- Per-epoch history (latent, generated model, loss) is logged to MLflow as
  figures; final metrics are MSE / MAE / SSIM.

## Interpretation

- **`trained` init converging** → encoded models sit in reachable basins.
- **`random`/`randomn` converging similarly** → wide basins; the manifold is
  forgiving (good sign for FWI).
- **Large gap or plateau** → the difficulty is dominated by reaching the
  manifold, or the manifold is too rigid — a fast-fail signal.

## Determinism

Seeds control data partitioning, initialization, batch ordering and dropout
(`utl.set_seed`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Residual
non-determinism (thread races after the data split) is inherited from the
original codebase and documented in the grid-search app header.

## Planned next steps

- Target-agnostic latent initializations based on train statistics
  (`mean`, Gaussian fit, perturbation around encoded codes).
- Latent-cloud statistics (per-channel scales, effective dimensionality).
- Baselines: pixel-space optimization and deep-image-prior (untrained
  decoder) on the same targets.
- Physics-based data misfit (adjoint-state FWI) replacing the model-space
  MSE.
