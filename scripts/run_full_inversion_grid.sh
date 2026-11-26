#!/bin/bash
# =============================================================================
# run_full_inversion_grid.sh — full inversion sweep for the four campaign refits
# =============================================================================
# Runs run_lot_inversion.sh (the epochs x lr x init sweep) once per refit
# model — the same four refit models as the 2026-11 campaign — sequentially;
# each sweep parallelizes internally over GPUS (one job per GPU).
#
# The four refits are baked in below (override the list with REFITS="..."
# lines of arch|seed|refit_run_id|refit_experiment, or filter with ONLY=).
#
# Knobs (forwarded to run_lot_inversion.sh / run_inversion.sh):
#   RES            [high]        DATASET       [deep]
#   K              optional, label only
#   EPOCHS_LIST    ["50 100 500 1000"]
#   LR_LIST        ["1 0.5 0.1 0.01 0.001"]
#   INIT_LIST      ["random randomn zeros trained"]
#   TRAINED_EXAMPLE_ID [10]
#   GPUS           ["0,1,2,3"]   PARALLEL [1]
#   INV_EXPERIMENT [encdec.highres.inversion] — the original campaign
#                  experiment; rename the OLD campaign first (or point this
#                  somewhere else) so pre-fix and post-fix runs never mix.
#   ONLY           <refit_short> — run a single refit's sweep
#   DRY_RUN=1 / SKIP_CHECKS=1 / FORCE=1 — forwarded
#
# Example (the full re-run of the campaign grid with the eval fix):
#   bash scripts/run_full_inversion_grid.sh
# =============================================================================
set -uo pipefail   # no -e: one failed refit must not kill the remaining ones
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh variables.inc
. "$SCRIPTS_DIR/lib.sh"
. "$SCRIPTS_DIR/variables.inc"

# ---- knobs -------------------------------------------------------------------
RES="${RES:-high}"
DATASET="${DATASET:-deep}"
K="${K:-}"
GPUS_CSV="${GPUS:-0,1,2,3}"
PARALLEL="${PARALLEL:-1}"
INV_EXPERIMENT="${INV_EXPERIMENT:-encdec.highres.inversion}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"

# ---- the four campaign refits: arch|seed|refit_run_id|refit_experiment -------
DEFAULT_REFITS="\
encdec|7|6d115baa522d44ccb0a570ac42caf302|encdec.highres.refit.seed.7
encdec|79|9247757e7e0f49459ee4fe8e897fcd14|encdec.highres.refit.seed.79
encdec|571|b01fdd281f6d465082da2cb433f9c989|encdec.highres.refit.seed.571
transencdec|571|32409900943c44f7ae798d3cf2d57ed1|transencdec.highres.refit.seed.571"
REFITS="${REFITS:-$DEFAULT_REFITS}"

echo "[ INFO ] experiment : $INV_EXPERIMENT (uri ${MLFLOW_URI:-<unset>})"
echo "[ INFO ] grid       : EPOCHS_LIST='${EPOCHS_LIST:-50 100 500 1000}' | LR_LIST='${LR_LIST:-1 0.5 0.1 0.01 0.001}' | INIT_LIST='${INIT_LIST:-random randomn zeros trained}'"
echo "[ INFO ] gpus       : $GPUS_CSV (PARALLEL=$PARALLEL) | DRY_RUN=$DRY_RUN"

TSV_DIR="$EXPERIMENTS_ROOT/inversion/logs/${RES}res/${DATASET}er"
failed=()
n_ok=0

while IFS='|' read -r arch seed refit_id refit_exp; do
    [ -n "$arch" ] || continue
    case "$arch" in \#*) continue ;; esac
    short="${refit_id:0:8}"
    if [ -n "$ONLY" ] && [ "$short" != "$ONLY" ]; then
        echo "[ SKIP ] refit $short ($arch seed $seed) — ONLY=$ONLY"
        continue
    fi

    echo "==================================================================="
    echo "[ RUN ] refit $short — $arch, seed $seed -> $INV_EXPERIMENT"
    echo "==================================================================="
    if ARCH="$arch" SEED="$seed" REFIT_RUN_ID="$refit_id" \
       REFIT_EXPERIMENT="$refit_exp" INV_EXPERIMENT="$INV_EXPERIMENT" \
       RES="$RES" DATASET="$DATASET" K="$K" GPUS="$GPUS_CSV" \
       PARALLEL="$PARALLEL" DRY_RUN="$DRY_RUN" \
           bash "$SCRIPTS_DIR/run_lot_inversion.sh"; then
        n_ok=$((n_ok + 1))
        echo "[ DONE ] refit $short — sweep tsv: $TSV_DIR/sweep_${short}.tsv"
    else
        failed+=("$short ($arch, seed $seed)")
        echo "[ERROR] refit $short failed — continuing with the next refit" >&2
    fi
done <<< "$REFITS"

echo "==================================================================="
echo "[ SUMMARY ] refits ok=$n_ok failed=${#failed[@]}"
[ -n "${K:-}" ] && true
for f in "${failed[@]:-}"; do
    [ -n "$f" ] && echo "  FAILED: $f"
done
[ "$n_ok" -gt 0 ] && echo "[ INFO ] per-refit sweep tsvs in: $TSV_DIR/"
[ "${#failed[@]}" -eq 0 ] || exit 1
