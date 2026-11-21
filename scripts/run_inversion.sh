#!/bin/bash
# =============================================================================
# run_inversion.sh — direct latent recovery (reparameterized inversion)
# =============================================================================
# All knobs are environment variables (defaults in brackets):
#   ARCH=encdec RES=high DATASET=deep SEED=7 \
#   REFIT_RUN_ID=<mlflow-run-id> REFIT_EXPERIMENT=encdec.highres.refit.seed.7 \
#   LATENT_INIT=trained LR=0.001 MAX_EPOCHS=1000 \
#   bash scripts/run_inversion.sh
#
# Required: REFIT_RUN_ID REFIT_EXPERIMENT SEED (must match the refit seed —
#           the app asserts this against the refit run attributes)
# Knobs:
#   ARCH [encdec] RES [high] DATASET [deep]
#   K              optional, included in the label only
#   LATENT_INIT    random | randomn | zeros | trained        [trained]
#   LATENT_IDX     train-example index for 'trained'         [10]
#   LR             Adam learning rate for the latent         [0.001]
#   MAX_EPOCHS     latent-optimization iterations            [1000]
#   MODEL_PATH     checkpoint source dir                     [$MODEL_LOAD_PATH]
#   INV_EXPERIMENT MLflow experiment   [<arch>.<res>.inversion]
#   DEVICE cuda:all | CUDA_ID | EXTRA_ARGS | VERBOSE_LEVEL | PYTHON
#   DRY_RUN=1 / SKIP_CHECKS=1 — see run_grid_search.sh header
# =============================================================================
set -euo pipefail
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh variables.inc
. "$SCRIPTS_DIR/lib.sh"
. "$SCRIPTS_DIR/variables.inc"

# ---- knobs -------------------------------------------------------------------
ARCH="${ARCH:-encdec}"
RES="${RES:-high}"
DATASET="${DATASET:-deep}"
K="${K:-}"
SEED="${SEED:-}"
LATENT_INIT="${LATENT_INIT:-trained}"
LATENT_IDX="${LATENT_IDX:-10}"
LR="${LR:-0.001}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
MODEL_PATH="${MODEL_PATH:-$MODEL_LOAD_PATH}"
INV_EXPERIMENT="${INV_EXPERIMENT:-${ARCH}.${RES}res.inversion}"
DEVICE="${DEVICE:-cuda:all}"
CUDA_ID="${CUDA_ID:-}"
PYTHON="${PYTHON:-python}"
VERBOSE_LEVEL="${VERBOSE_LEVEL:-1}"
DRY_RUN="${DRY_RUN:-0}"

require_env REFIT_RUN_ID "MLflow run id of the refit run to invert from"
require_env REFIT_EXPERIMENT "MLflow experiment containing REFIT_RUN_ID"
require_env SEED "SEED (must match the refit run seed)"

case "$LATENT_INIT" in
    random|randomn|zeros|trained) : ;;
    *) die "LATENT_INIT must be random | randomn | zeros | trained (got '$LATENT_INIT')" ;;
esac

# ---- guards -------------------------------------------------------------------
if [ "${SKIP_CHECKS:-0}" != "1" ]; then
    require_dir "$(data_basedir "$DATASET")" "Dataset basedir (DATASET=$DATASET)"
    require_dir "$MODEL_PATH" "Checkpoint source dir (MODEL_PATH)"
    mlflow_guard "$MLFLOW_URI"
fi
gpu_guard "$DEVICE" "$CUDA_ID"

# ---- paths and label -----------------------------------------------------------
RES_DIR="${RES}res"
DS_DIR="${DATASET}er"
OUTDIR="$EXPERIMENTS_ROOT/inversion/logs/${RES_DIR}/${DS_DIR}/${REFIT_RUN_ID}"
ARCH_UC="$(echo "$ARCH" | tr '[:lower:]' '[:upper:]')"
REFIT_SHORT="${REFIT_RUN_ID:0:8}"
LABEL_ID="${ARCH_UC}_INV_${RES_DIR}${K:+_k${K}}_${DS_DIR}_seed${SEED}_${LATENT_INIT}_lr${LR}_e${MAX_EPOCHS}_r${REFIT_SHORT}"

mkdir -p "$OUTDIR"

APP="$SRCPATH/src/apps/inversion.py"
APP_ARGS=(--model_path="$MODEL_PATH"
          --refit_run_id="$REFIT_RUN_ID"
          --refit_experiment_tracking_name="$REFIT_EXPERIMENT"
          --label_id="$LABEL_ID"
          --experiment_tracking_uri="$MLFLOW_URI"
          --experiment_tracking_name="$INV_EXPERIMENT"
          --dataset_max_samples="${MAX_SAMPLES:--1}"
          --dataset_type="$DATASET" --dataset_resolution="$RES"
          --model_arch="$ARCH" --seed="$SEED"
          --max_epochs="$MAX_EPOCHS" --lr="$LR"
          --latent_init="$LATENT_INIT" --latent_init_idx="$LATENT_IDX"
          --device="$DEVICE" --verbose_level="$VERBOSE_LEVEL")

[ -n "$CUDA_ID" ] && export CUDA_VISIBLE_DEVICES="$CUDA_ID"

write_manifest "$OUTDIR" "inversion" "$LABEL_ID" \
    "arch : $ARCH" "dataset : $DATASET" "refit_run_id : $REFIT_RUN_ID" \
    "refit_experiment : $REFIT_EXPERIMENT" "latent_init : $LATENT_INIT" \
    "latent_idx : $LATENT_IDX" "lr : $LR" "max_epochs : $MAX_EPOCHS" \
    "model_path : $MODEL_PATH" "app : $APP"

echo "[ RUN ] $LABEL_ID"
echo "[ OUT ] $OUTDIR"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] PYTHONPATH=$SRCPATH $PYTHON $APP ${APP_ARGS[*]} ${EXTRA_ARGS:-}"
    exit 0
fi

( cd "$OUTDIR" && PYTHONPATH="$SRCPATH" "$PYTHON" "$APP" "${APP_ARGS[@]}" ${EXTRA_ARGS:-} \
    1>"$LABEL_ID.out" 2>"$LABEL_ID.err" )
echo "[ DONE ] logs: $OUTDIR/$LABEL_ID.out(.err)"
