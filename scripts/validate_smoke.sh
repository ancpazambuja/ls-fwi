#!/bin/bash
# =============================================================================
# validate_smoke.sh — end-to-end smoke validation of the experiment harness
# =============================================================================
# Intended to run INSIDE the reference container on a GPU node, e.g.:
#
#   singularity run --nv -e \
#     --bind /gemini:/gemini,/bw1nfs1:/bw1nfs1,/env_modules:/env_modules,/tgdesenv/src/proj0/ntbs:/tgdesenv \
#     /gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/docker/dt2_v4.sif \
#     bash /tgdesenv/doutorado/5semestre.2024.2/estudo.orientado/projetos/foundation_models/ls-fwi/scripts/validate_smoke.sh
#
# Stages (STAGES env; default "sanity dry gs refit inference inversion"):
#   sanity     container / GPU / library / MLflow checks
#   dry        dry-run of the four stage launchers
#   gs         tiny grid search   (MAX_SAMPLES, GS_EPOCHS epochs, cv=2)
#   refit      tiny refit of the gs best run (REFIT_EPOCHS epochs)
#   inference  latent-space characterization of the refit run
#   inversion  tiny latent recovery (INV_EPOCHS epochs)
#
# Knobs:
#   ARCH RES K DATASET SEED          [encdec high 3 deep 0]
#   MAX_SAMPLES  TEST_PERC           [220  0.1]   (>=220 keeps the hardcoded
#                                     inversion test indices [1,10,20] valid)
#   DEVICE  CUDA_ID                  [cuda:all  unset]
#   GS_EPOCHS REFIT_EPOCHS INV_EPOCHS [1 1 2]
#   SMOKE_TAG    MLflow experiment suffix ["smoke"]
#   STAGES       stages to run
#   FALLBACK_REFIT_RUN_ID / FALLBACK_SEED
#                if the gs->refit chain fails, downstream stages use this
#                already-existing refit run instead
#   MODEL_PATH   checkpoint dir override for the smoke refit
#                [experiments/smoke_models]
#
# Exit code: non-zero if any executed stage failed.
# =============================================================================
set -uo pipefail
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh variables.inc
. "$SCRIPTS_DIR/lib.sh"
. "$SCRIPTS_DIR/variables.inc"

# ---- knobs -------------------------------------------------------------------
ARCH="${ARCH:-encdec}";  RES="${RES:-high}";  K="${K:-3}"
DATASET="${DATASET:-deep}"; SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-220}"; TEST_PERC="${TEST_PERC:-0.1}"
DEVICE="${DEVICE:-cuda:all}"; CUDA_ID="${CUDA_ID:-}"
GS_EPOCHS="${GS_EPOCHS:-1}"; REFIT_EPOCHS="${REFIT_EPOCHS:-1}"; INV_EPOCHS="${INV_EPOCHS:-2}"
SMOKE_TAG="${SMOKE_TAG:-smoke}"
PYTHON="${PYTHON:-python}"
STAGES="${STAGES:-sanity dry gs refit inference inversion}"
FALLBACK_REFIT_RUN_ID="${FALLBACK_REFIT_RUN_ID:-6d115baa522d44ccb0a570ac42caf302}"
FALLBACK_SEED="${FALLBACK_SEED:-7}"
FALLBACK_REFIT_EXPERIMENT="${FALLBACK_REFIT_EXPERIMENT:-${ARCH}.${RES}res.refit.seed.${FALLBACK_SEED}}"

GS_EXPERIMENT="${ARCH}.${RES}res.${SMOKE_TAG}"
REFIT_EXPERIMENT="${ARCH}.${RES}res.${SMOKE_TAG}.refit"
INF_EXPERIMENT="${ARCH}.${RES}res.${SMOKE_TAG}.inference"
INV_EXPERIMENT="${ARCH}.${RES}res.${SMOKE_TAG}.inversion"

PASS_LIST=""; FAIL_LIST=""; SKIP_LIST=""
GS_RUN_ID=""; REFIT_RUN_ID=""; SEED_EFF="$SEED"

banner() { echo; echo "################################################################"; echo "##  $*"; echo "################################################################"; echo; }
record()    { PASS_LIST="$PASS_LIST $1"; echo "[PASS] $1 ${2:-}"; }
record_fail(){ FAIL_LIST="$FAIL_LIST $1"; echo "[FAIL] $1 ${2:-}"; }
record_skip(){ SKIP_LIST="$SKIP_LIST $1"; echo "[SKIP] $1 ${2:-}"; }
has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ---- one-time: MLflow reachability (fall back to a local file store) ----------
if ! mlflow_guard "$MLFLOW_URI" 2>/dev/null; then
    warn "MLflow server unreachable from this node; falling back to a local file store"
    export MLFLOW_URI="file://$EXPERIMENTS_ROOT/mlruns_smoke"
    mkdir -p "$EXPERIMENTS_ROOT/mlruns_smoke"
fi

# ---- helper: latest run id of an MLflow experiment -----------------------------
latest_run() {  # latest_run <experiment_name> [uri] — newest run WITH params
    local exp="$1" uri="${2:-$MLFLOW_URI}"
    "$PYTHON" - "$exp" "$uri" <<'PYEOF' 2>/dev/null
import sys
import mlflow
name, uri = sys.argv[1], sys.argv[2]
mlflow.set_tracking_uri(uri)
c = mlflow.tracking.MlflowClient()
e = c.get_experiment_by_name(name)
if e is None:
    sys.exit(1)
# Grid-search results live in SUBRUNS (hpset_N, with params) under param-less
# parent runs; the refit stage consumes a subrun, so pick the newest run that
# actually carries parameters.
runs = c.search_runs([e.experiment_id], order_by=["attributes.start_time DESC"], max_results=50)
for r in runs:
    if len(r.data.params) > 0:
        print(r.info.run_id)
        break
PYEOF
}

# ===============================================================================
# STAGE: sanity
# ===============================================================================
if has_stage sanity; then
    banner "STAGE: sanity"
    ok=1
    echo "[ INFO ] repo        : $REPO_ROOT"
    echo "[ INFO ] git         : $(git_summary)"
    echo "[ INFO ] MLflow uri  : $MLFLOW_URI"
    echo "[ INFO ] data (deep) : $DATA_BASEDIR_DEEP"
    echo "[ INFO ] models save : $MODEL_SAVE_PATH"
    echo "[ INFO ] models load : $MODEL_LOAD_PATH"
    [ -d "$REPO_ROOT" ] || { echo "[FAIL] repo dir not visible"; ok=0; }
    [ -d "$DATA_BASEDIR_DEEP" ] || { echo "[FAIL] dataset dir not visible"; ok=0; }
    "$PYTHON" - <<'PYEOF' || ok=0
import sys
import numpy, sklearn, skorch, torch, mlflow
print("[ INFO ] python :", sys.version.split()[0])
print("[ INFO ] torch  :", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| n_gpus:", torch.cuda.device_count())
print("[ INFO ] numpy  :", numpy.__version__, "| sklearn:", sklearn.__version__,
      "| skorch:", skorch.__version__, "| mlflow:", mlflow.__version__)
assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "no CUDA device visible"
PYEOF
    if [ "$ok" = "1" ]; then record sanity; else record_fail sanity; fi
fi

# ===============================================================================
# STAGE: dry
# ===============================================================================
if has_stage dry; then
    banner "STAGE: dry (launchers print their command; nothing executes)"
    ok=1
    DRY_ROOT="${TMPDIR:-/tmp}/validate_smoke_dry.$$/experiments"
    ARCH="$ARCH" RES="$RES" K="$K" DATASET="$DATASET" SEED="$SEED" \
        DRY_RUN=1 SKIP_CHECKS=1 EXPERIMENTS_ROOT="$DRY_ROOT" \
        bash "$SCRIPTS_DIR/run_grid_search.sh" || ok=0
    ARCH="$ARCH" RES="$RES" K="$K" DATASET="$DATASET" SEED="$SEED" \
        BEST_RUN_ID=dryrun0000 BEST_RUN_EXPERIMENT="$GS_EXPERIMENT" \
        DRY_RUN=1 SKIP_CHECKS=1 EXPERIMENTS_ROOT="$DRY_ROOT" \
        bash "$SCRIPTS_DIR/run_refit.sh" || ok=0
    ARCH="$ARCH" RES="$RES" DATASET="$DATASET" REFIT_RUN_ID=dryrun0000 REFIT_EXPERIMENT="$REFIT_EXPERIMENT" \
        DRY_RUN=1 SKIP_CHECKS=1 EXPERIMENTS_ROOT="$DRY_ROOT" \
        bash "$SCRIPTS_DIR/run_inference.sh" || ok=0
    ARCH="$ARCH" RES="$RES" DATASET="$DATASET" SEED="$SEED" REFIT_RUN_ID=dryrun0000 REFIT_EXPERIMENT="$REFIT_EXPERIMENT" \
        DRY_RUN=1 SKIP_CHECKS=1 EXPERIMENTS_ROOT="$DRY_ROOT" \
        bash "$SCRIPTS_DIR/run_inversion.sh" || ok=0
    rm -rf "${TMPDIR:-/tmp}/validate_smoke_dry.$$"
    if [ "$ok" = "1" ]; then record dry; else record_fail dry; fi
fi

# ===============================================================================
# STAGE: gs
# ===============================================================================
if has_stage gs; then
    banner "STAGE: gs — tiny grid search ($MAX_SAMPLES samples, $GS_EPOCHS epoch(s), cv=2)"
    if ARCH="$ARCH" RES="$RES" K="$K" DATASET="$DATASET" SEED="$SEED" \
       MAX_EPOCHS="$GS_EPOCHS" MAX_SAMPLES="$MAX_SAMPLES" TEST_PERC="$TEST_PERC" \
       GS_EXPERIMENT="$GS_EXPERIMENT" DEVICE="$DEVICE" CUDA_ID="$CUDA_ID" EXTRA_ARGS="--cv=2" \
          bash "$SCRIPTS_DIR/run_grid_search.sh"; then
        GS_RUN_ID="$(latest_run "$GS_EXPERIMENT")"
        if [ -n "$GS_RUN_ID" ]; then
            record gs "(run id $GS_RUN_ID)"
        else
            record_fail gs "(completed but no MLflow run id found)"
        fi
    else
        record_fail gs
    fi
fi

# ===============================================================================
# STAGE: refit
# ===============================================================================
if has_stage refit; then
    banner "STAGE: refit — tiny refit of the gs best run ($REFIT_EPOCHS epoch(s))"
    if [ -z "$GS_RUN_ID" ]; then
        record_skip refit "(no gs run id available)"
    elif ARCH="$ARCH" RES="$RES" K="$K" DATASET="$DATASET" SEED="$SEED" \
       BEST_RUN_ID="$GS_RUN_ID" BEST_RUN_EXPERIMENT="$GS_EXPERIMENT" \
       REFIT_EXPERIMENT="$REFIT_EXPERIMENT" \
       MAX_EPOCHS="$REFIT_EPOCHS" MAX_SAMPLES="$MAX_SAMPLES" TEST_PERC="$TEST_PERC" \
       DEVICE="$DEVICE" CUDA_ID="$CUDA_ID" \
          bash "$SCRIPTS_DIR/run_refit.sh"; then
        REFIT_RUN_ID="$(latest_run "$REFIT_EXPERIMENT")"
        if [ -n "$REFIT_RUN_ID" ]; then
            record refit "(run id $REFIT_RUN_ID)"
        else
            record_fail refit "(completed but no MLflow run id found)"
        fi
    else
        record_fail refit
    fi
fi

# ---- downstream stages need a refit run ----------------------------------------
REFIT_EXP_EFF="$REFIT_EXPERIMENT"
if has_stage inference || has_stage inversion; then
    if [ -z "$REFIT_RUN_ID" ]; then
        if [ -n "$FALLBACK_REFIT_RUN_ID" ]; then
            warn "no refit run from this validation; using FALLBACK_REFIT_RUN_ID=$FALLBACK_REFIT_RUN_ID (seed $FALLBACK_SEED)"
            REFIT_RUN_ID="$FALLBACK_REFIT_RUN_ID"
            SEED_EFF="$FALLBACK_SEED"
            REFIT_EXP_EFF="$FALLBACK_REFIT_EXPERIMENT"
        else
            warn "no refit run available; skipping downstream stages"
        fi
    fi
fi

# ===============================================================================
# STAGE: inference
# ===============================================================================
if has_stage inference; then
    banner "STAGE: inference — latent-space characterization (refit $REFIT_RUN_ID)"
    if [ -z "$REFIT_RUN_ID" ]; then
        record_skip inference "(no refit run id)"
    elif ARCH="$ARCH" RES="$RES" DATASET="$DATASET" SEED="$SEED_EFF" \
       REFIT_RUN_ID="$REFIT_RUN_ID" REFIT_EXPERIMENT="$REFIT_EXP_EFF" \
       INF_EXPERIMENT="$INF_EXPERIMENT" MAX_SAMPLES="$MAX_SAMPLES" \
       DEVICE="$DEVICE" CUDA_ID="$CUDA_ID" \
          bash "$SCRIPTS_DIR/run_inference.sh"; then
        record inference
    else
        record_fail inference
    fi
fi

# ===============================================================================
# STAGE: inversion
# ===============================================================================
if has_stage inversion; then
    banner "STAGE: inversion — tiny latent recovery ($INV_EPOCHS epoch(s), latent_init=trained)"
    if [ -z "$REFIT_RUN_ID" ]; then
        record_skip inversion "(no refit run id)"
    elif ARCH="$ARCH" RES="$RES" DATASET="$DATASET" SEED="$SEED_EFF" \
       REFIT_RUN_ID="$REFIT_RUN_ID" REFIT_EXPERIMENT="$REFIT_EXP_EFF" \
       INV_EXPERIMENT="$INV_EXPERIMENT" LATENT_INIT=trained LR=0.001 MAX_EPOCHS="$INV_EPOCHS" \
       DEVICE="$DEVICE" CUDA_ID="$CUDA_ID" \
          bash "$SCRIPTS_DIR/run_inversion.sh"; then
        record inversion
    else
        record_fail inversion
    fi
fi

# ===============================================================================
# summary
# ===============================================================================
banner "VALIDATION SUMMARY"
echo "  PASS :${PASS_LIST:- none}"
echo "  FAIL :${FAIL_LIST:- none}"
echo "  SKIP :${SKIP_LIST:- none}"
[ -z "$FAIL_LIST" ] && [ -n "$PASS_LIST$SKIP_LIST" ] && echo "  ==> VALIDATION OK" || echo "  ==> VALIDATION FINISHED WITH FAILURES"
[ -z "$FAIL_LIST" ]
