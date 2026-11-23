# Experiments harness

Experiments are launched from the **versioned** harness in
[`scripts/`](../scripts/); their working directories live in the
**gitignored** [`experiments/`](../experiments/README.md) tree (schema
documented there). Each run writes a `run_info.txt` manifest (git commit,
args, MLflow coordinates) next to its console logs, so provenance survives
MLflow server migrations.

## Launchers

| Script | Stage | Key knobs |
|---|---|---|
| `scripts/run_grid_search.sh` | HPO of the autoencoder | `ARCH RES K DATASET SEED MAX_EPOCHS TEST_PERC …` |
| `scripts/run_refit.sh` | Final AE training | `+ BEST_RUN_ID BEST_RUN_EXPERIMENT SEED MAX_EPOCHS=500 …` |
| `scripts/run_inference.sh` | Latent-space characterization | `+ REFIT_RUN_ID REFIT_EXPERIMENT …` |
| `scripts/run_inversion.sh` | Direct latent recovery | `+ REFIT_RUN_ID REFIT_EXPERIMENT SEED LATENT_INIT LR MAX_EPOCHS …` |
| `scripts/run_lot_inversion.sh` | Inversion sweep (epochs × lr × init) | `EPOCHS_LIST LR_LIST INIT_LIST GPUS PARALLEL DRY_RUN` |
| `scripts/make_dir_structure.sh` | Pre-create the case tree | `STAGE ARCH_LIST KERNELS DATASETS …` |

All knobs are environment variables with the defaults actually used in the
first experiments (see each script header). `DRY_RUN=1` prints the command;
`SKIP_CHECKS=1` bypasses dataset/MLflow guards; `FORCE=1` proceeds on an
unreachable MLflow server.

## Examples

```bash
# HPO: kernel-size study, one case
ARCH=encdec RES=high K=5 DATASET=deep SEED=0 bash scripts/run_grid_search.sh

# Refit: best GS config, MCCV seed
ARCH=encdec RES=high K=5 DATASET=deep SEED=7 \
BEST_RUN_ID=<gs-best-run-id> BEST_RUN_EXPERIMENT=encdec.highres.gs \
    bash scripts/run_refit.sh

# Inference + inversion of that refit
ARCH=encdec RES=high DATASET=deep SEED=7 \
REFIT_RUN_ID=<refit-run-id> REFIT_EXPERIMENT=encdec.highres.refit.seed.7 \
    bash scripts/run_inference.sh

# Inversion sweep over epochs x lr x latent-init on 4 GPUs (parallel)
ARCH=encdec RES=high DATASET=deep SEED=7 \
REFIT_RUN_ID=<refit-run-id> REFIT_EXPERIMENT=encdec.highres.refit.seed.7 \
GPUS=0,1,2,3 PARALLEL=1 \
    bash scripts/run_lot_inversion.sh
```

## Running on the cluster (GPU nodes)

Experiments run inside the reference container on allocated GPU nodes (see
[Installation](installation.md)). The cluster storage and the repository's
parent directory are bind-mounted at the **same absolute paths inside the
container**, so every path in the harness (`/gemini/...`, the repository
itself) works unchanged; all launchers self-locate the repository
(`REPO_ROOT`), so nothing is hardcoded to a checkout location.

Two invocation modes:

```bash
# interactive shell
singularity run --nv -e --bind <storage-binds> <image.sif> bash -i

# scripted / non-interactive — note: `run` does not forward arguments to a
# command; use `exec`:
singularity exec --nv -e --bind <storage-binds> <image.sif> bash -c 'bash <repo>/scripts/...'
```

Long-running stages should be launched from a persistent session (`tmux`/
`screen`) or with `nohup`: the console output goes to `<label_id>.out` /
`<label_id>.err` inside the case directory, so progress can be followed
with `tail -f` from any machine sharing the filesystem.

### End-to-end recipe (a real experiment)

1. **Grid search** — one case per configuration:
   ```bash
   ARCH=encdec RES=high K=5 DATASET=deep SEED=0 bash scripts/run_grid_search.sh
   ```
   Results are logged as MLflow **subruns** (`hpset_<i>`, one per grid
   element) under the task run.
2. **Pick the best configuration** — in the MLflow UI, open the grid-search
   experiment and select the `hpset_*` subrun with the highest
   `mean_valid_score` (the negated cross-validation loss, so higher is
   better); its run id is the `gs_best_run_id`. Programmatically:
   ```python
   import mlflow
   mlflow.set_tracking_uri("<tracking-uri>")
   c = mlflow.tracking.MlflowClient()
   e = c.get_experiment_by_name("encdec.highres.gs")
   runs = c.search_runs([e.experiment_id],
                        filter_string="tags.`mlflow.runName` LIKE 'hpset%'",
                        order_by=["metrics.mean_valid_score DESC"])
   print(runs[0].info.run_id)  # -> gs_best_run_id
   ```
3. **Refit** — once per MCCV seed (the ten seeds live in
   `scripts/seeds.txt`); the refit run stores all architecture attributes
   consumed downstream:
   ```bash
   for SEED in $(cat scripts/seeds.txt); do
     ARCH=encdec RES=high K=5 DATASET=deep SEED=$SEED \
     BEST_RUN_ID=<gs_best_run_id> BEST_RUN_EXPERIMENT=encdec.highres.gs \
         bash scripts/run_refit.sh
   done
   ```
4. **Inference** (latent-space characterization) — one call per refit run:
   ```bash
   ARCH=encdec RES=high DATASET=deep SEED=<refit-seed> \
   REFIT_RUN_ID=<refit-run-id> \
   REFIT_EXPERIMENT=encdec.highres.refit.seed.<refit-seed> \
       bash scripts/run_inference.sh
   ```
5. **Inversion** (direct latent recovery) — single cases with
   `scripts/run_inversion.sh`, or the epochs × lr × latent-init sweep with
   `scripts/run_lot_inversion.sh` (see its header for knobs; `GPUS=0,1,2,3
   PARALLEL=1` dispatches one concurrent job per GPU).

## MLflow naming

Experiments follow `encdec.highres.<stage>[.seed.<s>]`:
`<arch>.<res>.gs`, `<arch>.<res>.refit.seed.<seed>`,
`<arch>.<res>.inference`, `<arch>.<res>.inversion` — overridable via
`GS_EXPERIMENT` / `REFIT_EXPERIMENT` / `INF_EXPERIMENT` / `INV_EXPERIMENT`.

### Smoke experiments (project convention)

All smoke/validation runs log to the `*smoke*` experiments:
`<arch>.<res>res.smoke`, `<arch>.<res>res.smoke.refit`,
`<arch>.<res>res.smoke.inference`, `<arch>.<res>res.smoke.inversion`
(default `SMOKE_TAG=smoke` in `scripts/validate_smoke.sh`).

These experiments are the project's **permanent test sandbox**: they are
kept (never deleted), and runs from repeated validations accumulate there.
They are clearly separated from the real experiments, which never carry the
`smoke` tag. Smoke refit checkpoints are likewise identifiable by their
`REFIT__ENCDEC_REFIT_...` names and timestamps inside the checkpoint
directory.

## Layout and history

The harness replaced the pre-harness launchers that lived under
`results/vel_encdec/` (project's previous name). That tree was moved into
`experiments/`; old launchers are preserved untracked under
`experiments/<family>/scripts/` for reference. A compatibility symlink at
the old location keeps historical notes valid.
