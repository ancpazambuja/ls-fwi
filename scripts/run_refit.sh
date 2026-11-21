#!/bin/bash
# =============================================================================
# run_refit.sh — refit stage: final autoencoder training (best GS configuration)
# =============================================================================
# All knobs are environment variables (defaults in brackets):
#   ARCH=encdec RES=high K=5 DATASET=deep SEED=7 \
#   BEST_RUN_ID=<mlflow-run-id> BEST_RUN_EXPERIMENT=encdec.highres.gs \
#   bash scripts/run_refit.sh
#
# Required: ARCH RES K DATASET SEED BEST_RUN_ID BEST_RUN_EXPERIMENT
# Knobs:
#   DILATION, TCONV                                            [1, 2]
#   ATT_NUM_LAYERS   transformer depth override (-1 = keep GS value)  [-1]
#   MAX_EPOCHS                                                  [500]
#   MAX_SAMPLES     -1 = all                                    [-1]
#   TEST_PERC                                                  [0.1]
#   UNEMBED                                                    [paper]
#   REFIT_EXPERIMENT  target MLflow experiment  [<arch>.<res>.refit.seed.<seed>]
#   MODEL_PATH       checkpoint destination                    [$MODEL_SAVE_PATH]
#   DEVICE cuda:all | CUDA_ID | EXTRA_ARGS | VERBOSE_LEVEL | PYTHON
#   DRY_RUN=1 / SKIP_CHECKS=1 — see run_grid_search.sh header
# =============================================================================
set -euo pipefail
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh variables.inc
. "$SCRIPTS_DIR/lib.sh"
. "$SCRIPTS_DIR/variables.inc"

# ---- knobs -------------------------------------------------------------------
ARCH="${ARCH:-?}"
RES="${RES:-?}"
K="${K:-?}"
DATASET="${DATASET:-?}"
SEED="${SEED:-?}"
BEST_RUN_ID="${BEST_RUN_ID:-}"
BEST_RUN_EXPERIMENT="${BEST_RUN_EXPERIMENT:-}"
[ "$ARCH"    != "?" ] || die "ARCH is required (encdec | transencdec)"
[ "$RES"     != "?" ] || die "RES is required (high | mid | low)"
[ "$K"       != "?" ] || die "K is required (convolution kernel size, e.g. 5)"
[ "$DATASET" != "?" ] || die "DATASET is required (shallow | deep)"
[ "$SEED"    != "?" ] || die "SEED is required"
require_env BEST_RUN_ID "MLflow run id of the best grid-search configuration"
require_env BEST_RUN_EXPERIMENT "MLflow experiment containing BEST_RUN_ID"

DILATION="${DILATION:-1}"
TCONV="${TCONV:-2}"
ATT_NUM_LAYERS="${ATT_NUM_LAYERS:--1}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
TEST_PERC="${TEST_PERC:-0.1}"
UNEMBED="${UNEMBED:-paper}"
REFIT_EXPERIMENT="${REFIT_EXPERIMENT:-${ARCH}.${RES}res.refit.seed.${SEED}}"
MODEL_PATH="${MODEL_PATH:-$MODEL_SAVE_PATH}"
DEVICE="${DEVICE:-cuda:all}"
CUDA_ID="${CUDA_ID:-}"
PYTHON="${PYTHON:-python}"
VERBOSE_LEVEL="${VERBOSE_LEVEL:-1}"
DRY_RUN="${DRY_RUN:-0}"

# ---- guards -------------------------------------------------------------------
if [ "${SKIP_CHECKS:-0}" != "1" ]; then
    require_dir "$(data_basedir "$DATASET")" "Dataset basedir (DATASET=$DATASET)"
    require_dir "$MODEL_PATH" "Checkpoint destination dir (MODEL_PATH)"
    mlflow_guard "$MLFLOW_URI"
fi
gpu_guard "$DEVICE" "$CUDA_ID"

# ---- paths and label -----------------------------------------------------------
RES_DIR="${RES}res"
DS_DIR="${DATASET}er"
OUTDIR="$EXPERIMENTS_ROOT/training/${ARCH}/${RES_DIR}/${K}x${K}/${DS_DIR}/random_seeds/seed_${SEED}"
ARCH_UC="$(echo "$ARCH" | tr '[:lower:]' '[:upper:]')"
LABEL_ID="${ARCH_UC}_REFIT_${RES}_${K}x${K}_${DS_DIR}_seed${SEED}"

mkdir -p "$OUTDIR"

APP="$SRCPATH/src/apps/refit.py"
APP_ARGS=(--refit --model_path="$MODEL_PATH"
          --gs_best_run_id="$BEST_RUN_ID"
          --gs_best_run_experiment_tracking_name="$BEST_RUN_EXPERIMENT"
          --label_id="$LABEL_ID"
          --experiment_tracking_uri="$MLFLOW_URI"
          --experiment_tracking_name="$REFIT_EXPERIMENT"
          --dataset_max_samples="$MAX_SAMPLES" --max_epochs="$MAX_EPOCHS"
          --test_perc="$TEST_PERC" --seed="$SEED"
          --dataset_type="$DATASET" --dataset_resolution="$RES"
          --model_arch="$ARCH" --unembed_type="$UNEMBED"
          --conv_filter_size_x="$K" --conv_filter_size_z="$K"
          --conv_dilation_size_x="$DILATION" --conv_dilation_size_z="$DILATION"
          --transposed_conv_filter_size_x="$TCONV" --transposed_conv_filter_size_z="$TCONV"
          --att_num_layers="$ATT_NUM_LAYERS"
          --device="$DEVICE" --verbose_level="$VERBOSE_LEVEL")

[ -n "$CUDA_ID" ] && export CUDA_VISIBLE_DEVICES="$CUDA_ID"

write_manifest "$OUTDIR" "refit" "$LABEL_ID" \
    "arch : $ARCH" "kernel : ${K}x${K}" "dataset : $DATASET" "seed : $SEED" \
    "best_run_id : $BEST_RUN_ID" "best_run_experiment : $BEST_RUN_EXPERIMENT" \
    "refit_experiment : $REFIT_EXPERIMENT" "max_epochs : $MAX_EPOCHS" \
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
