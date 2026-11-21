# Tutorials

## Smoke run (software validation)

Runs the real entry point on a few samples to validate the environment,
data loading, preprocessing, training and MLflow logging. Requires the
reference container (see [Installation](installation.md)) and access to the
cluster datasets.

```bash
cd /workspace   # repository root, bind-mounted in the container

PYTHONPATH=$PWD python src/apps/grid_search_parallel.py \
    --label_id=SMOKE \
    --experiment_tracking_uri=file:///workspace/mlruns \
    --experiment_tracking_name=smoke \
    --dataset_type=deep --dataset_resolution=high \
    --model_arch=encdec --conv_filter_size_x=3 --conv_filter_size_z=3 \
    --max_epochs=1 --dataset_max_samples=8 --test_perc=0.25 --cv=2 \
    --seed=0 --device=cuda:0 --verbose_level=1
```

**What you should see:**

- `FINDING OUT DATA FORMAT` with the newer-dataset model shape (z = 680);
- the grid-search fits (1 epoch × 2 folds);
- an MLflow run in the experiment `smoke` with parameters
  (`batch_size`, `lr`, `max_epochs`, …) and `mean_valid_score`.

A local `file://` tracking store keeps the smoke test self-contained; point
`--experiment_tracking_uri` at the cluster server for real runs.

## Full walkthrough

Follow the four stages in order — HPO, refit, inference, inversion — using
the commands in the [User guide](user_guide.md). The run ids chain the
stages: the grid search produces the best-run id consumed by the refit; the
refit run id (and its stored architecture attributes) configures both the
inference and inversion apps.

## Inspecting the results

- **MLflow UI** (cluster server or `mlflow ui --backend-store-uri
  file:///workspace/mlruns`): per-epoch losses, latent diagnostics figures
  (interpolations, traversals), inversion histories (latent, generated model
  and loss per epoch), and refit checkpoints.
- **Console logs**: `<label_id>.out` / `<label_id>.err` next to the working
  directory.
