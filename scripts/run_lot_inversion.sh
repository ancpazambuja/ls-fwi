#!/bin/bash
# =============================================================================
# run_lot_inversion.sh — sweep driver for the direct latent recovery
# =============================================================================
# Runs run_inversion.sh over the cartesian product of
#   MAX_EPOCHS x LR x LATENT_INIT
# dispatching cases round-robin over GPUS (optionally in parallel, one
# concurrent job per GPU), and appends one line per case to a summary TSV.
#
# Required:
#   ARCH RES DATASET SEED REFIT_RUN_ID REFIT_EXPERIMENT
# Knobs:
#   EPOCHS_LIST   "50 100 500 1000"
#   LR_LIST       "1 0.5 0.1 0.01 0.001"
#   INIT_LIST     "random randomn zeros trained"
#   TRAINED_EXAMPLE_ID                                  [10]
#   K             optional (label only)
#   GPUS          csv of gpu ids for round-robin        ["0"]
#   PARALLEL=1    one concurrent job per GPU            [0 = sequential]
#   DRY_RUN=1     print the commands without running them
#
# Example:
#   ARCH=encdec RES=high DATASET=deep SEED=7 \
#   REFIT_RUN_ID=6d115baa522d44ccb0a570ac42caf302 \
#   REFIT_EXPERIMENT=encdec.highres.refit.seed.7 \
#   GPUS=0,1,2,3 PARALLEL=1 bash scripts/run_lot_inversion.sh
# =============================================================================
set -euo pipefail
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh variables.inc
. "$SCRIPTS_DIR/lib.sh"
. "$SCRIPTS_DIR/variables.inc"

# ---- knobs -------------------------------------------------------------------
EPOCHS_LIST="${EPOCHS_LIST:-50 100 500 1000}"
LR_LIST="${LR_LIST:-1 0.5 0.1 0.01 0.001}"
INIT_LIST="${INIT_LIST:-random randomn zeros trained}"
TRAINED_EXAMPLE_ID="${TRAINED_EXAMPLE_ID:-10}"
GPUS_CSV="${GPUS:-0}"
PARALLEL="${PARALLEL:-0}"
DRY_RUN="${DRY_RUN:-0}"

require_env ARCH "architecture (encdec | transencdec)"
require_env RES "resolution (high | mid | low)"
require_env DATASET "dataset (shallow | deep)"
require_env SEED "seed (must match the refit seed)"
require_env REFIT_RUN_ID "MLflow run id of the refit run"
require_env REFIT_EXPERIMENT "MLflow experiment containing REFIT_RUN_ID"

read -r -a GPU_LIST <<< "${GPUS_CSV//,/ }"
read -r -a EPOCHS <<< "$EPOCHS_LIST"
read -r -a LRS <<< "$LR_LIST"
read -r -a INITS <<< "$INIT_LIST"

DS_DIR="${DATASET}er"
RES_DIR="${RES}res"
ARCH_UC="$(echo "$ARCH" | tr '[:lower:]' '[:upper:]')"
REFIT_SHORT="${REFIT_RUN_ID:0:8}"
SUMMARY_DIR="$EXPERIMENTS_ROOT/inversion/logs/${RES_DIR}/${DS_DIR}"
SUMMARY="$SUMMARY_DIR/sweep_${REFIT_SHORT}.tsv"
mkdir -p "$SUMMARY_DIR"

if [ "$DRY_RUN" != "1" ]; then
    if [ ! -e "$SUMMARY" ]; then
        printf 'label\tlatent_init\tlr\tmax_epochs\toutdir\n' > "$SUMMARY"
    fi
fi

# ---- dispatch one case ---------------------------------------------------------
dispatch() {  # dispatch <gpu_id> <epochs> <lr> <init>
    local gpu="$1" e="$2" lr="$3" init="$4"
    local label="${ARCH_UC}_INV_${RES_DIR}${K:+_k${K}}_${DS_DIR}_seed${SEED}_${init}_lr${lr}_e${e}_r${REFIT_SHORT}"
    echo "[ INFO ] gpu=$gpu epochs=$e lr=$lr init=$init"
    if [ "$DRY_RUN" = "1" ]; then
        CUDA_ID="$gpu" MAX_EPOCHS="$e" LR="$lr" LATENT_INIT="$init" LATENT_IDX="$TRAINED_EXAMPLE_ID" \
            DRY_RUN=1 SKIP_CHECKS="${SKIP_CHECKS:-1}" \
            bash "$SCRIPTS_DIR/run_inversion.sh"
        return 0
    fi
    CUDA_ID="$gpu" MAX_EPOCHS="$e" LR="$lr" LATENT_INIT="$init" LATENT_IDX="$TRAINED_EXAMPLE_ID" \
        bash "$SCRIPTS_DIR/run_inversion.sh"
    local resdir
    resdir="$EXPERIMENTS_ROOT/inversion/logs/${RES_DIR}/${DS_DIR}/${REFIT_RUN_ID}"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$init" "$lr" "$e" "$resdir" >> "$SUMMARY"
}

# ---- sweep ---------------------------------------------------------------------
n_gpus=${#GPU_LIST[@]}
running=0
declare -A BUSY

i=0
for e in "${EPOCHS[@]}"; do
  for lr in "${LRS[@]}"; do
    for init in "${INITS[@]}"; do
      gpu="${GPU_LIST[$((i % n_gpus))]}"

      if [ "$PARALLEL" != "1" ]; then
          dispatch "$gpu" "$e" "$lr" "$init"
          i=$((i + 1))
          continue
      fi

      # parallel: at most one running job per gpu
      while :; do
          placed=0
          for g in "${GPU_LIST[@]}"; do
              pid="${BUSY[$g]:-}"
              if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                  dispatch "$g" "$e" "$lr" "$init" &
                  BUSY[$g]=$!
                  placed=1
                  break
              fi
          done
          [ "$placed" = "1" ] && break
          sleep 10
      done
      i=$((i + 1))
    done
  done
done

# wait for the tail of parallel jobs
if [ "$PARALLEL" = "1" ]; then
    for g in "${GPU_LIST[@]}"; do
        pid="${BUSY[$g]:-}"
        [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
    done
fi

echo "[ DONE ] sweep complete"
[ "$DRY_RUN" = "1" ] || echo "[ INFO ] summary: $SUMMARY"
