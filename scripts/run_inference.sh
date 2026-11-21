#!/bin/bash
# =============================================================================
# run_inference.sh — latent-space characterization of a refit model
# =============================================================================
# All knobs are environment variables (defaults in brackets):
#   ARCH=encdec RES=high DATASET=deep \
#   REFIT_RUN_ID=<mlflow-run-id> REFIT_EXPERIMENT=encdec.highres.refit.seed.7 \
#   bash scripts/run_inference.sh
#
# Required: REFIT_RUN_ID REFIT_EXPERIMENT
# Knobs:
#   ARCH [encdec] RES [high] DATASET [deep]
#   K                optional, included in the label only
#   SEED             [-1] (non-deterministic diagnostics)
#   MAX_SAMPLES  -1 = all                                     [-1]
#   MODEL_PATH   checkpoint source dir                        [$MODEL_LOAD_PATH]
#   INF_EXPERIMENT  MLflow experiment  [<arch>.<res>.inference]
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
SEED="${SEED:--1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MODEL_PATH="${MODEL_PATH:-$MODEL_LOAD_PATH}"
INF_EXPERIMENT="${INF_EXPERIMENT:-${ARCH}.${RES}res.inference}"
DEVICE="${DEVICE:-cuda:all}"
CUDA_ID="${CUDA_ID:-}"
PYTHON="${PYTHON:-python}"
VERBOSE_LEVEL="${VERBOSE_LEVEL:-1}"
DRY_RUN="${DRY_RUN:-0}"

require_env REFIT_RUN_ID "MLflow run id of the refit run to analyse"
require_env REFIT_EXPERIMENT "MLflow experiment containing REFIT_RUN_ID"

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
OUTDIR="$EXPERIMENTS_ROOT/inference/logs/${RES_DIR}/${DS_DIR}/${REFIT_RUN_ID}"
ARCH_UC="$(echo "$ARCH" | tr '[:lower:]' '[:upper:]')"
REFIT_SHORT="${REFIT_RUN_ID:0:8}"
LABEL_ID="${ARCH_UC}_INFERENCE_${RES_DIR}${K:+_k${K}}_${DS_DIR}_${REFIT_SHORT}"

mkdir -p "$OUTDIR"

APP="$SRCPATH/src/apps/inference.py"
APP_ARGS=(--model_path="$MODEL_PATH"
          --refit_run_id="$REFIT_RUN_ID"
          --refit_experiment_tracking_name="$REFIT_EXPERIMENT"
          --label_id="$LABEL_ID"
          --experiment_tracking_uri="$MLFLOW_URI"
          --experiment_tracking_name="$INF_EXPERIMENT"
          --dataset_max_samples="$MAX_SAMPLES"
          --dataset_type="$DATASET" --dataset_resolution="$RES"
          --model_arch="$ARCH" --seed="$SEED"
          --device="$DEVICE" --verbose_level="$VERBOSE_LEVEL")

[ -n "$CUDA_ID" ] && export CUDA_VISIBLE_DEVICES="$CUDA_ID"

write_manifest "$OUTDIR" "inference" "$LABEL_ID" \
    "arch : $ARCH" "dataset : $DATASET" "refit_run_id : $REFIT_RUN_ID" \
    "refit_experiment : $REFIT_EXPERIMENT" "model_path : $MODEL_PATH" "app : $APP"

echo "[ RUN ] $LABEL_ID"
echo "[ OUT ] $OUTDIR"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] PYTHONPATH=$SRCPATH $PYTHON $APP ${APP_ARGS[*]} ${EXTRA_ARGS:-}"
    exit 0
fi

( cd "$OUTDIR" && PYTHONPATH="$SRCPATH" "$PYTHON" "$APP" "${APP_ARGS[@]}" ${EXTRA_ARGS:-} \
    1>"$LABEL_ID.out" 2>"$LABEL_ID.err" )
echo "[ DONE ] logs: $OUTDIR/$LABEL_ID.out(.err)"
