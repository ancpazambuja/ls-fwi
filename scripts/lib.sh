# =============================================================================
# lib.sh — shared helpers for the experiment launchers (sourced, not executed)
# =============================================================================
# Guards fail fast BEFORE a long run starts. Set SKIP_CHECKS=1 to bypass the
# dataset/mlflow checks (e.g., for DRY_RUN on a machine without the mounts),
# and FORCE=1 to proceed even if the MLflow server is unreachable.

die()  { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN]  $*" >&2; }

require_env() {                       # require_env <VAR_NAME> <what>
    local var="$1" what="${2:-$1}"
    [ -n "${!var:-}" ] || die "$what is required (export $var or pass it inline: $var=... $0)"
}

require_dir() {                       # require_dir <path> <what>
    [ -d "$1" ] || die "$2 not found: $1 (check the mount or override the variable; SKIP_CHECKS=1 bypasses)"
}

warn_dir() {
    [ -d "$1" ] || warn "$2 not found: $1"
}

data_basedir() {                      # data_basedir <dataset_type> -> path
    case "$1" in
        shallow) echo "$DATA_BASEDIR_SHALLOW" ;;
        deep)    echo "$DATA_BASEDIR_DEEP" ;;
        *) die "DATASET must be 'shallow' or 'deep' (got '$1')" ;;
    esac
}

gpu_guard() {                         # gpu_guard <device> [cuda_visible_devices]
    command -v nvidia-smi >/dev/null 2>&1 || { warn "nvidia-smi not available; skipping GPU check"; return 0; }
    local cvd="${2:-}" id
    [ -z "$cvd" ] && return 0
    for id in $(echo "$cvd" | tr ',' ' '); do
        nvidia-smi -i "$id" >/dev/null 2>&1 || die "GPU '$id' (CUDA_ID) not visible to nvidia-smi"
    done
}

mlflow_guard() {                      # mlflow_guard <uri>
    case "$1" in
        file://*)  return 0 ;;
        http*)     : ;;
        *)         warn "unrecognized tracking uri: $1"; return 0 ;;
    esac
    [ "${SKIP_CHECKS:-0}" = "1" ] && return 0
    curl -fsS --max-time 5 -o /dev/null "$1" 2>/dev/null && return 0
    curl -fsS --max-time 5 -o /dev/null "$1" 2>/dev/null && return 0
    [ "${FORCE:-0}" = "1" ] && { warn "MLflow server unreachable, proceeding (FORCE=1): $1"; return 0; }
    die "MLflow server not reachable: $1 (check VPN/server; FORCE=1 proceeds anyway)"
}

git_summary() {                       # "<commit> [dirty: N]" or fallback
    if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        local commit dirty
        commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
        dirty="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no | wc -l)"
        if [ "$dirty" -gt 0 ]; then
            echo "$commit (dirty: $dirty tracked file(s) modified)"
        else
            echo "$commit"
        fi
    else
        echo "not-a-git-repo"
    fi
}

write_manifest() {                    # write_manifest <outdir> <stage> <label> ["key : value"]...
    local outdir="$1" stage="$2" label="$3"; shift 3
    {
        echo "timestamp : $(date --iso-8601=seconds)"
        echo "host      : $(hostname)"
        echo "gpu       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} device=${DEVICE:-<default>}"
        echo "git       : $(git_summary)"
        echo "stage     : $stage"
        echo "label     : $label"
        echo "outdir    : $outdir"
        echo "mlflow    : uri=${MLFLOW_URI:-} experiment=${MLFLOW_EXPERIMENT:-}"
        local kv
        for kv in "$@"; do echo "$kv"; done
        echo "args      : ${APP_ARGS:-}"
    } > "$outdir/run_info.txt"
    echo "[ INFO ] manifest: $outdir/run_info.txt"
}
