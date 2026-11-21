# User guide

## Scope

This guide describes how to run the four stages of the latent-recovery
pipeline: autoencoder hyper-parameter search (grid search), final training
(refit), latent-space characterization (inference), and latent recovery
(inversion).

## Inputs

### Datasets

Data is resolved from **cluster paths** (see
[Installation](installation.md)); the loader selects the layout by
`--dataset_type`:

| `--dataset_type` | Layout | Notes |
|---|---|---|
| `shallow` | legacy `.../deeptomograph_data/data/` | loads only the true + highres model slots |
| `deep` (default) | `.../deeptomograph_data/data_new_and_bigger/` | newer, deeper dataset (z = 680) |

This project trains **model→model autoencoders**: X = Y = velocity model
(single-channel 2D image). The `--dataset_resolution` option selects the
model family used as the pair: `high` → true models, `mid` → mediumres
models, `low` → lowres models. Samples are cropped to 672×800 (multiples of
2⁵) before entering the network.

### Preprocessing

sklearn pipelines applied in place (`src/preprocessing/pipeline.py`):
per-column `BatchAbsMaxScaler` on X and Y (identical factors, so X ≡ Y holds
after normalization), then tensorization; X is converted to channels-first
`(N, 1, nz, nx)`.

### Experiment tracking

All runs are logged to MLflow (CLI `--experiment_tracking_uri`; cluster
server by default, `file://` for smoke tests). The
`--experiment_tracking_name` selects the experiment; the default is
`deep_tomography_2.0_fallback`.

## Outputs

- **MLflow runs**: parameters, per-epoch metrics (train/valid loss), figures
  (data samples, latent diagnostics, inversion histories), and refit
  checkpoints (weights/optimizer/history next to the run name).
- **Console logs**: `<label_id>.out` / `<label_id>.err` next to the working
  directory (convention inherited from the original experiments).

## Entry points and main options

### `src/apps/grid_search_parallel.py`

| Argument | Default | Meaning |
|---|---|---|
| `--label_id` | `undefined` | Run label; also names the `.out/.err` logs |
| `--dataset_type` / `--dataset_resolution` | — / `high` | Dataset layout / model family |
| `--dataset_max_samples` | `-1` | Limit samples (`-1` = all) |
| `--test_perc` | `0.1` | Holdout fraction |
| `--cv` | `5` | Cross-validation folds |
| `--model_arch` | — | `encdec` (U-Net) or `transencdec` (TransUNet) |
| `--conv_filter_size_x/z` | 13, 13 | Convolution kernel size |
| `--conv_dilation_size_x/z` | 1, 1 | Dilation |
| `--unembed_type` | `paper` | Transformer unembedding (`none`, `paper`, `linear`) |
| `--max_epochs` | 130 | Training budget per grid element |
| `--seed` | `-1` | Random seed (`-1` = non-deterministic) |
| `--device` | `cuda:0` | `cpu`, `cuda:N` or `cuda:all` (one task per GPU) |

### `src/apps/refit.py`

Same data/model arguments, plus:

| Argument | Meaning |
|---|---|
| `--refit` | Required flag (double check) |
| `--gs_best_run_id` (+ `--gs_best_run_experiment_tracking_name`) | MLflow run id of the best grid-search configuration; its params configure the refit |
| `--att_num_layers` | Transformer depth override (transencdec only) |
| `--model_path` | Where to save the trained model after refit |
| `--one_step_training` | Single-phase training (default true) |

### `src/apps/inference.py`

| Argument | Meaning |
|---|---|
| `--model_path` | Path to the trained (refit) model |
| `--refit_run_id` (+ `--refit_experiment_tracking_name`) | Refit MLflow run whose attributes reconfigure the architecture (kernel sizes, attention config, seed check) |
| `--model_arch` | `encdec` / `transencdec` |

### `src/apps/inversion.py`

Same as inference, plus:

| Argument | Default | Meaning |
|---|---|---|
| `--max_epochs` | 130 | Latent-optimization iterations |
| `--lr` | 0.1 | Adam learning rate for the latent |
| `--latent_init` | — | `random`, `randomn`, `zeros` or `trained` |
| `--latent_init_idx` | 0 | Train-example index for the `trained` initialization |

## Typical workflows

```bash
REPO=/workspace   # repository root (bind-mounted)

# 1. Grid search (HPO of the autoencoder)
PYTHONPATH=$REPO python $REPO/src/apps/grid_search_parallel.py \
    --label_id=GS_AE_DEEP_HIGH --dataset_type=deep --dataset_resolution=high \
    --model_arch=encdec --conv_filter_size_x=13 --conv_filter_size_z=13 \
    --max_epochs=300 --seed=100 --device=cuda:all

# 2. Refit the best configuration (weights checkpointed; run id needed below)
PYTHONPATH=$REPO python $REPO/src/apps/refit.py \
    --label_id=REFIT_AE_DEEP_HIGH --refit --gs_best_run_id=<best-run-id> \
    --gs_best_run_experiment_tracking_name=deep_tomography_2.0_fallback \
    --dataset_type=deep --dataset_resolution=high --max_epochs=500 \
    --seed=100 --model_path=/workspace/models --device=cuda:all

# 3. Latent-space characterization
PYTHONPATH=$REPO python $REPO/src/apps/inference.py \
    --label_id=INFER_AE_DEEP_HIGH --model_path=/workspace/models/<model> \
    --refit_run_id=<refit-run-id> --refit_experiment_tracking_name=deep_tomography_2.0_fallback \
    --dataset_type=deep --dataset_resolution=high --model_arch=encdec \
    --seed=100 --device=cuda:all

# 4. Latent recovery (reparameterized inversion, model space)
PYTHONPATH=$REPO python $REPO/src/apps/inversion.py \
    --label_id=INV_AE_DEEP_HIGH --model_path=/workspace/models/<model> \
    --refit_run_id=<refit-run-id> --refit_experiment_tracking_name=deep_tomography_2.0_fallback \
    --dataset_type=deep --dataset_resolution=high --model_arch=encdec \
    --latent_init=trained --lr=0.1 --max_epochs=130 --seed=100 --device=cuda:all
```

## Known limitations

- The restart feature (`--execution_id`) only re-attaches the MLflow run;
  task/model resumption is not implemented (`read_checkpoint_dir()` raises
  `NotImplementedError`).
- Some inversion constants are hardcoded for the `deep` configuration: latent
  shape `(1, 512, 21, 25)` and the inverted test-model indices `[1, 10, 20]`.
