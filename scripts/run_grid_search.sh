#!/bin/bash
# =============================================================================
# run_grid_search.sh — HPO stage: autoencoder hyper-parameter grid search
# =============================================================================
# All knobs are environment variables (defaults in brackets):
#   ARCH=encdec RES=high K=13 DATASET=deep SEED=0 bash scripts/run_grid_search.sh
#
# Knobs:
#   ARCH            encdec | transencdec                       [encdec]
#   RES             high | mid | low                           [high]
#   K               convolution kernel size (K x K)            [13]
#   DATASET         shallow | deep                             [deep]
#   SEED                                                       [0]
#   MAX_EPOCHS                                                 [100]
#   MAX_SAMPLES     -1 = all                                   [-1]
#   TEST_PERC       HPO subset fraction (protocol: 0.5)        [0.5]
#   DILATION, TCONV, UNEMBED                                [1, 2, paper]
#   DEVICE          cpu | cuda:N | cuda:all                    [cuda:all]
#   CUDA_ID         value exported as CUDA_VISIBLE_DEVICES     [unset]
#   GS_EXPERIMENT   MLflow experiment name  [<arch>.<res>.gs]
#   MLFLOW_URI, MODEL_* , EXTRA_ARGS, VERBOSE_LEVEL, PYTHON
#   DRY_RUN=1 prints the command without running it
#   SKIP_CHECKS=1 bypasses dataset/mlflow guards (DRY_RUN on odd machines)
# =============================================================================
set -euo pipefail
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh variables.inc
. "$SCRIPTS_DIR/lib.sh"
. "$SCRIPTS_DIR/variables.inc"

# ---- knobs -------------------------------------------------------------------
ARCH="${ARCH:-encdec}"
RES="${RES:-high}"
K="${K:-13}"
DATASET="${DATASET:-deep}"
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
TEST_PERC="${TEST_PERC:-0.5}"
DILATION="${DILATION:-1}"
TCONV="${TCONV:-2}"
UNEMBED="${UNEMBED:-paper}"
DEVICE="${DEVICE:-cuda:all}"
CUDA_ID="${CUDA_ID:-}"
PYTHON="${PYTHON:-python}"
VERBOSE_LEVEL="${VERBOSE_LEVEL:-1}"
MLFLOW_EXPERIMENT="${GS_EXPERIMENT:-${ARCH}.${RES}res.gs}"
DRY_RUN="${DRY_RUN:-0}"

# ---- guards -------------------------------------------------------------------
if [ "${SKIP_CHECKS:-0}" != "1" ]; then
    require_dir "$(data_basedir "$DATASET")" "Dataset basedir (DATASET=$DATASET)"
    mlflow_guard "$MLFLOW_URI"
fi
gpu_guard "$DEVICE" "$CUDA_ID"

# ---- paths and label -----------------------------------------------------------
RES_DIR="${RES}res"
DS_DIR="${DATASET}er"
OUTDIR="$EXPERIMENTS_ROOT/grid_search/${ARCH}/${RES_DIR}/${K}x${K}/${DS_DIR}/seed_${SEED}"
ARCH_UC="$(echo "$ARCH" | tr '[:lower:]' '[:upper:]')"
LABEL_ID="${ARCH_UC}_GS_${RES}_${K}x${K}_${DS_DIR}_seed${SEED}"

mkdir -p "$OUTDIR"

APP="$SRCPATH/src/apps/grid_search_parallel.py"
APP_ARGS=(--label_id="$LABEL_ID"
          --experiment_tracking_uri="$MLFLOW_URI"
          --experiment_tracking_name="$MLFLOW_EXPERIMENT"
          --dataset_type="$DATASET" --dataset_resolution="$RES"
          --dataset_max_samples="$MAX_SAMPLES" --max_epochs="$MAX_EPOCHS"
          --test_perc="$TEST_PERC" --seed="$SEED"
          --model_arch="$ARCH" --unembed_type="$UNEMBED"
          --conv_filter_size_x="$K" --conv_filter_size_z="$K"
          --conv_dilation_size_x="$DILATION" --conv_dilation_size_z="$DILATION"
          --transposed_conv_filter_size_x="$TCONV" --transposed_conv_filter_size_z="$TCONV"
          --device="$DEVICE" --verbose_level="$VERBOSE_LEVEL")

[ -n "$CUDA_ID" ] && export CUDA_VISIBLE_DEVICES="$CUDA_ID"

write_manifest "$OUTDIR" "grid_search" "$LABEL_ID" \
    "arch : $ARCH" "kernel : ${K}x${K}" "dilation : ${DILATION}" \
    "dataset : $DATASET" "seed : $SEED" "max_epochs : $MAX_EPOCHS" \
    "test_perc : $TEST_PERC" "app : $APP"

echo "[ RUN ] $LABEL_ID"
echo "[ OUT ] $OUTDIR"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] PYTHONPATH=$SRCPATH $PYTHON $APP ${APP_ARGS[*]} ${EXTRA_ARGS:-}"
    exit 0
fi

( cd "$OUTDIR" && PYTHONPATH="$SRCPATH" "$PYTHON" "$APP" "${APP_ARGS[@]}" ${EXTRA_ARGS:-} \
    1>"$LABEL_ID.out" 2>"$LABEL_ID.err" )
echo "[ DONE ] logs: $OUTDIR/$LABEL_ID.out(.err)"
