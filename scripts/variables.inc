# =============================================================================
# variables.inc — shared experiment configuration (sourced by all launchers)
# =============================================================================
# Every variable has an env-var override: exporting it before invoking a
# launcher wins over the defaults here. Nothing in this file is hardcoded to
# a particular checkout location: REPO_ROOT is derived from where THIS file
# lives, so the harness keeps working if the repository is moved or renamed.

SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPTS_DIR/.." && pwd)"
export SRCPATH="${SRCPATH:-$REPO_ROOT}"

# --- Cluster storage (bind-mounted in the reference container) ---------------
GEMINI_DATA_ROOT="${GEMINI_DATA_ROOT:-/gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data}"
DATA_BASEDIR_SHALLOW="${DATA_BASEDIR_SHALLOW:-$GEMINI_DATA_ROOT/data}"
DATA_BASEDIR_DEEP="${DATA_BASEDIR_DEEP:-$GEMINI_DATA_ROOT/data_new_and_bigger}"

# Refit checkpoints destination (run_refit.sh --model_path)
MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-$GEMINI_DATA_ROOT/models}"

# Checkpoints source for inference/inversion (--model_path)
# NOTE: the real checkpoint files (REFIT__*__params.pt etc.) live directly in
# .../deeptomograph_results/predictions.encdec/ . The historical nested
# path .../predictions.encdec/predictions.encdec is a dangling self-referential
# symlink (kept only for backwards compatibility with old run notes).
MODEL_LOAD_PATH="${MODEL_LOAD_PATH:-$GEMINI_DATA_ROOT/data_new_and_bigger/deeptomograph_results/predictions.encdec}"

# --- Experiment tracking (MLflow) --------------------------------------------
MLFLOW_URI="${MLFLOW_URI:-http://gr14sm02.ep.petrobras.com.br:31947/}"

# --- Run-tree root (gitignored content; see experiments/README.md) -----------
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$REPO_ROOT/experiments}"

# --- Conventions ---------------------------------------------------------------
# Monte Carlo cross-validation seeds (same list as scripts/seeds.txt and the
# spatial-support-vmb protocol).
MCCV_SEEDS="7 79 181 307 433 571 701 853 997 1129"

# The repository lives on NFS and may be edited from another machine between
# runs; stale __pycache__ bytecode has already bitten once (the node kept
# loading an old classes.pyc after an edit). Do not write bytecode by default.
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
