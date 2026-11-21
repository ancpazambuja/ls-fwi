# Computational environment

The reference environment is **shared with the
[spatial-support-vmb](https://github.com/ancpazambuja/spatial-support-vmb)
project**: the same container image (`dt2_v4_reconstructed.sif`, built from
that repository's `configs/Dockerfile`) is reused here. The image was
reconstructed from the original research image and validated (`pip freeze`
identical, 292/292 packages).

| Component | Value |
|---|---|
| Base image | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` |
| OS | Ubuntu 22.04 |
| Python | CPython 3.11.9 |
| Deep-learning framework | PyTorch 2.3.0+cu121 (torchvision 0.18.0) |
| CUDA / cuDNN | 12.1.1 / 8.9.0 |
| GPUs (reference runs) | NVIDIA A100-SXM4-80GB (4× per node, `--device cuda:all`) |
| Key libraries | scikit-learn 1.7.0, skorch 1.1.0, mlflow 2.21.3, numpy 1.26.4, scipy 1.12.0 |
| Container runtime | Apptainer/Singularity (`--nv`), Docker/Podman documented |

## Reproducibility caveats

- The container pins the **software** stack; it cannot pin hardware behavior.
  Bitwise reproduction across GPU models/drivers is not guaranteed.
- Residual training non-determinism is documented in the code
  (`src/apps/grid_search_parallel.py` header: thread race conditions after
  the data split).

## MLflow

The tracking server is being migrated to a new version installed elsewhere.
Operational templates for the new server (run creation, probe-timeout patch,
run termination) are kept in [`../MLFLOW_TEMPLATES.md`](../MLFLOW_TEMPLATES.md).
