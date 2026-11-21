# `experiments/` — run tree

This directory holds the **experiment working tree**: case directories,
console logs, launch manifests, and sweep summaries for every experiment run.

**The run outputs are NOT part of the git repository** (see `.gitignore`):
git tracks only this `README.md`, which documents the schema. The launch
harness itself is versioned under [`../scripts/`](../scripts/).

## Case-tree schema

| Stage | Path pattern |
|---|---|
| grid search (HPO) | `grid_search/<arch>/<res>res/<k>x<k>/<ds>er/seed_<seed>/` |
| refit (training) | `training/<arch>/<res>res/<k>x<k>/<ds>er/random_seeds/seed_<seed>/` |
| inference | `inference/logs/<res>res/<ds>er/<refit_run_id>/` |
| inversion | `inversion/logs/<res>res/<ds>er/<refit_run_id>/` |

Conventions inherited from the original experiments (`results/vel_encdev`,
pre-harness era):

- `<ds>er` encodes the dataset as `deep`→`deeper`, `shallow`→`shallower`.
- `<res>res` is `highres` / `medres` / `lowres`.
- `k<k>x<k>` is the convolution kernel size (`3x3` … `13x13`).
- Refit seeds follow the Monte Carlo cross-validation list
  (`../scripts/seeds.txt`): 7, 79, 181, 307, 433, 571, 701, 853, 997, 1129.

## What a case directory contains

- `<label>.out` / `<label>.err` — console logs (stdout/stderr) of the run;
- `run_info.txt` — launch manifest written by the harness: timestamp, host,
  GPU, **git commit (and dirty state)**, stage, label, MLflow uri/experiment,
  the stage's key arguments, and the exact app path.
- Sweep summaries: `inversion/logs/.../sweep_<refit8>.tsv` — one line per
  case dispatched by `run_lot_inversion.sh`.

Provenance therefore lives in two complementary places: the **local
manifest** (survives MLflow server migrations) and the **MLflow run**
(metrics, figures, checkpoints).

## Legacy content

The pre-harness launchers used to run the first experiments (when the
project was named `vel_encdec`) are preserved untracked under
`<family>/scripts/` for historical reference. They are superseded by
`../scripts/run_<stage>.sh` and should not be used for new runs.

A compatibility symlink may exist at the old location
(`../../results/vel_encdec` → this directory) so that old notes and shell
history keep resolving.
