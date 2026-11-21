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

## MLflow naming

Experiments follow `encdec.highres.<stage>[.seed.<s>]`:
`<arch>.<res>.gs`, `<arch>.<res>.refit.seed.<seed>`,
`<arch>.<res>.inference`, `<arch>.<res>.inversion` — overridable via
`GS_EXPERIMENT` / `REFIT_EXPERIMENT` / `INF_EXPERIMENT` / `INV_EXPERIMENT`.

## Layout and history

The harness replaced the pre-harness launchers that lived under
`results/vel_encdec/` (project's previous name). That tree was moved into
`experiments/`; old launchers are preserved untracked under
`experiments/<family>/scripts/` for reference. A compatibility symlink at
the old location keeps historical notes valid.
