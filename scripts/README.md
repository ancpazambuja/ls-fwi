# Scripts — experiment launch harness

Versioned launchers for the experiment stages. Working directories and logs
go to the gitignored [`../experiments/`](../experiments/README.md) tree — see
[`../docs/experiments.md`](../docs/experiments.md) for the full guide.

| Script | Purpose |
|---|---|
| `variables.inc` | Shared configuration (paths, MLflow URI, MCCV seeds). `REPO_ROOT` is derived from this file's location — nothing is hardcoded to a checkout path. |
| `lib.sh` | Shared helpers: fail-fast guards (dataset dir, GPU, MLflow reachability), launch manifest writer (`run_info.txt`), git summary. |
| `run_grid_search.sh` | HPO stage (`src/apps/grid_search_parallel.py`). |
| `run_refit.sh` | Refit stage (`src/apps/refit.py`). |
| `run_inference.sh` | Latent-space characterization (`src/apps/inference.py`). |
| `run_inversion.sh` | Direct latent recovery (`src/apps/inversion.py`). |
| `run_lot_inversion.sh` | Inversion sweep driver (epochs × lr × latent-init), GPU round-robin, optional parallel dispatch, TSV summary. |
| `make_dir_structure.sh` | Pre-creates the case-tree directories. |
| `seeds.txt` | Monte Carlo cross-validation seeds (7, 79, 181, 307, 433, 571, 701, 853, 997, 1129). |
| `run.sh` | Legacy informal launcher, kept for reference only. |

All knobs are environment variables (defaults documented in each script
header); `DRY_RUN=1` prints commands without running, `SKIP_CHECKS=1`
bypasses the guards (useful for dry runs on machines without the cluster
mounts).

Historical note: the first experiments of this project (then named
`vel_encdec`) were launched by positional-argument scripts that lived under
`results/vel_encdec/<family>/scripts/`; that tree now lives (untracked)
under `../experiments/<family>/scripts/` and is superseded by this harness.
