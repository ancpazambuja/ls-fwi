#!/bin/bash
# =============================================================================
# make_dir_structure.sh — create the experiment case-tree (directories only)
# =============================================================================
# Knobs (defaults reproduce the original layout):
#   STAGE      grid_search | training               [grid_search]
#   ARCH_LIST  space-separated archs                ["encdec transencdec"]
#   RES_LIST   space-separated resolutions          ["high"]
#   KERNELS    space-separated kernel sizes         ["3 5 7 9 13"]
#   DATASETS   space-separated datasets             ["deep"]
#   SEED       seed for grid_search dirs            [0]
#   SEEDS_FILE seeds file for training dirs         [scripts/seeds.txt]
#   DRY_RUN=1  print the directories without creating them
#
# Tree schemas:
#   grid_search/<arch>/<res>res/<k>x<k>/<ds>er/seed_<seed>
#   training/<arch>/<res>res/<k>x<k>/<ds>er/random_seeds/seed_<seed>
# =============================================================================
set -euo pipefail
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh variables.inc
. "$SCRIPTS_DIR/lib.sh"
. "$SCRIPTS_DIR/variables.inc"

STAGE="${STAGE:-grid_search}"
ARCH_LIST="${ARCH_LIST:-encdec transencdec}"
RES_LIST="${RES_LIST:-high}"
KERNELS="${KERNELS:-3 5 7 9 13}"
DATASETS="${DATASETS:-deep}"
SEED="${SEED:-0}"
SEEDS_FILE="${SEEDS_FILE:-$SCRIPTS_DIR/seeds.txt}"
DRY_RUN="${DRY_RUN:-0}"

case "$STAGE" in
    grid_search|training) : ;;
    *) die "STAGE must be grid_search or training (got '$STAGE')" ;;
esac

created=0
for arch in $ARCH_LIST; do
  for res in $RES_LIST; do
    for k in $KERNELS; do
      for ds in $DATASETS; do
        ds_dir="${ds}er"
        if [ "$STAGE" = "grid_search" ]; then
            dirs="$EXPERIMENTS_ROOT/grid_search/${arch}/${res}res/${k}x${k}/${ds_dir}/seed_${SEED}"
        else
            dirs=""
            for s in $(cat "$SEEDS_FILE"); do
                dirs="$dirs $EXPERIMENTS_ROOT/training/${arch}/${res}res/${k}x${k}/${ds_dir}/random_seeds/seed_${s}"
            done
        fi
        for d in $dirs; do
            if [ "$DRY_RUN" = "1" ]; then
                echo "[DRY-RUN] mkdir -p $d"
            else
                mkdir -p "$d"
            fi
            created=$((created + 1))
        done
      done
    done
  done
done

echo "[ DONE ] $STAGE: $created case director(ies) ${DRY_RUN:+(dry-run)} under $EXPERIMENTS_ROOT"
