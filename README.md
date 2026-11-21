# Reparameterized FWI (`ls-fwi`)

**Simple Encoder-Decoder Architecture For Velocity Foundation Models**

> **Status:** research prototype — a deliberate *fast-fail* study. The code
> and ML machinery are inherited from the
> [spatial-support-vmb](https://github.com/ancpazambuja/spatial-support-vmb)
> project; the scientific goal is new.

## Scientific overview

In deep-learning-assisted FWI, most of the community uses neural networks as
*implicit regularizers*: the inductive bias of the network (deep image prior,
spectral/frequency bias) restricts the search space of velocity models during
inversion. Different architectures bring interesting results — notably help
with cycle skipping — but convergence from a random velocity model still
demands an enormous number of iterations.

This project tests a different route: **representation learning instead of
inductive bias**. An autoencoder is trained on velocity models; its decoder
(the *generator*) is then frozen, and the inversion searches **only the
latent space** learned by the network — a well-behaved manifold of plausible
models — rather than the space of velocity models itself.

**Fast-fail rationale.** Before coupling the latent search to wave physics,
the simplest version is validated: recover a *known* target velocity model by
optimizing only the latent code of the frozen generator (model-space MSE).
If this direct latent recovery does not converge, or takes too long, the idea
is unlikely to survive inside full FWI — and the project fails fast, cheaply.

## Pipeline (four stages)

| Stage | Entry point | Purpose |
|---|---|---|
| 1. HPO | `src/apps/grid_search_parallel.py` | Grid search (scikit-learn `GridSearchCV`, one task per GPU) for the autoencoder hyper-parameters; MLflow-logged |
| 2. Refit | `src/apps/refit.py` | Final autoencoder training with the best configuration; weights checkpointed |
| 3. Latent lab | `src/apps/inference.py` | AE/generator evaluation and latent-space characterization (pairwise cosine distances, interpolation/extrapolation, direction traversal) |
| 4. Latent recovery | `src/apps/inversion.py` | Frozen generator + trainable latent; model-space inversion of test velocity models |

## Repository layout

- `src/apps/`: entry points (grid search, refit, inference, inversion) + CLI and global-vars definitions
- `src/processing/`: models (`EncoderDecoder`, `Encoder`, `Generator`, `GeneratorInversion`), skorch builders, hyper-parameter/inversion grids, task and thread processing
- `src/transunet/`: ViT/TransUNet bottleneck implementation
- `src/data/`: dataset containers and builders (model→model pairs)
- `src/preprocessing/`: sklearn pipelines (scaling, tensorization)
- `src/experiment_tracking/`: MLflow wrapper and skorch callbacks
- `src/visualization/`: figure builders for data, models, and latent diagnostics
- `src/utl/`: shared utilities (logging, serialization, seeds, training callbacks)
- `notebooks/`: prototyping notebooks
- `doc/`: Doxygen configuration; `docs/`: this documentation set
- `scripts/`: versioned experiment launch harness
- `experiments/`: run tree (case directories, logs, manifests — gitignored; see [experiments README](experiments/README.md))

## Data

Supervised **model→model** pairs: X = Y = velocity model (single-channel 2D
image), drawn from the deep-tomography research datasets stored on the
institutional cluster (restricted; not included in this repository). See
[Data availability](docs/data_availability.md).

## Quick start

Experiments run in the same reference container as spatial-support-vmb
(Ubuntu 22.04, CPython 3.11.9, PyTorch 2.3.0+cu121, CUDA 12.1.1, cuDNN 8.9.0)
on A100 GPUs; code and data are bind-mounted, not baked into the image:

```bash
singularity run --nv -e \
    --bind /path/to/repo:/workspace \
    --bind /gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data:/data \
    dt2_v4_reconstructed.sif bash -i
```

Minimal smoke run (see [Tutorials](docs/tutorials.md)):

```bash
cd /workspace
PYTHONPATH=$PWD python src/apps/grid_search_parallel.py --label_id=SMOKE \
    --dataset_type=deep --dataset_resolution=high --max_epochs=1 \
    --dataset_max_samples=8 --seed=0 --device=cuda:0
```

## Documentation

- [Installation](docs/installation.md)
- [User guide](docs/user_guide.md)
- [Experimental protocol](docs/experimental_protocol.md)
- [Data availability](docs/data_availability.md)
- [Tutorials](docs/tutorials.md)
- [Experiments harness](docs/experiments.md)
- [Reproducibility checklist](docs/reproducibility_checklist.md)
- [Computational environment](environment/README.md)

MLflow server operational templates (temporary): [MLFLOW_TEMPLATES.md](MLFLOW_TEMPLATES.md)

## Known limitations (early stage)

- The restart feature is declared by `--execution_id` but only re-attaches the
  MLflow run; task/model resumption is not implemented.
- Some inversion constants are still hardcoded for the `deep` dataset
  configuration (latent shape, inverted test-model indices).
