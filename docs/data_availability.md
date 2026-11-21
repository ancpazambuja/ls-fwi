# Data availability

## Research datasets

The datasets used in this project are the deep-tomography research datasets
produced in the group's earlier work (the same data family used by
[spatial-support-vmb](https://github.com/ancpazambuja/spatial-support-vmb)).
They are stored on the institutional cluster and are **restricted**; they are
**not included** in this repository.

| Layout | Cluster path (bind-mounted) | Used by |
|---|---|---|
| legacy (`shallow`) | `.../deeptomograph_data/data/` | `--dataset_type shallow` |
| newer, deeper (`deep`) | `.../deeptomograph_data/data_new_and_bigger/` | `--dataset_type deep` (default) |

## What this project uses

This project trains **model→model autoencoders**: both X and Y are velocity
models (single-channel 2D images, z = 680 in the newer dataset, cropped to
672×800 for the 5-level encoder-decoder). The `--dataset_resolution` option
selects the model family used as the pair:

| `--dataset_resolution` | Model family |
|---|---|
| `high` | true models |
| `mid` | medium-resolution models |
| `low` | low-resolution models |

No seismic gathers are used in the current configuration (the gather-loading
code path from the original project remains in `src/data/data_builder.py`
but is bypassed).

## Access

Access to the datasets follows the same restrictions as the original
research (institutional cluster accounts). The repository contains no data,
secrets, or local-only assets.
