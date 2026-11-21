# Installation and Computational Requirements

## Supported environment

- Linux x86_64 with an NVIDIA GPU (CUDA 12.1, cuDNN 8) for training and
  evaluation.
- Python 3.11 (the reference container ships CPython 3.11.9).

The reference environment is **shared with the
[spatial-support-vmb](https://github.com/ancpazambuja/spatial-support-vmb)
project**: the same container image (`dt2_v4_reconstructed.sif`, built from
that repository's `configs/Dockerfile`) is reused here. See
[Computational environment](../environment/README.md) for the component
table.

## Option A — container (recommended)

Build/deploy the image as described in the
[spatial-support-vmb installation guide](https://github.com/ancpazambuja/spatial-support-vmb/blob/main/docs/installation.md)
(offline wheel cache + SHA256 verification + `podman build`/`singularity
build`). The image contains the complete pinned Python environment; source
code and data are provided through bind mounts:

```bash
# Docker (NVIDIA Container Toolkit required):
docker run --rm -it --gpus all \
    -v /path/to/ls-fwi:/workspace \
    -v /gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data:/data \
    ls-fwi-env bash -i

# Apptainer/Singularity (the runtime used for the original experiments):
singularity run --nv -e \
    --bind /path/to/ls-fwi:/workspace \
    --bind /gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data:/data \
    dt2_v4_reconstructed.sif bash -i
```

## Option B — local installation (without a container)

The applications are executed as plain scripts with `PYTHONPATH`; no package
installation is required:

```bash
cd ls-fwi
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn==1.7.0 skorch==1.1.0 mlflow==2.21.3 ml_collections==1.1.0 \
    numpy==1.26.4 scipy==1.12.0 matplotlib==3.8.3 seaborn==0.13.2 \
    pandas==2.2.3 pillow==10.3.0 torchinfo==1.8.0
```

## Experiment tracking (MLflow)

- The CLI default points to the cluster tracking server
  (`--experiment_tracking_uri`, experiment `deep_tomography_2.0_fallback`).
  A local `file://` store can be used for smoke tests.
- The tracking server is being **migrated to a new version installed
  elsewhere**; operational templates (run creation, probe-timeout patch, run
  termination) are kept in [`MLFLOW_TEMPLATES.md`](../MLFLOW_TEMPLATES.md).

## Datasets

The research datasets live on the institutional cluster (bind-mounted
`/gemini/...` paths above); they are **not** included in the repository. See
[Data availability](data_availability.md).
