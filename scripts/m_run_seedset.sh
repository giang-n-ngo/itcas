#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# m_run_seedset.sh — execute ONE job set from the JSON spec on a single GPU
# (cluster "m", A100 GPUs).
#
# Sibling of run_seedset.sh for the "m" cluster; run_seedset.sh remains the
# per-set launcher for the "f" cluster (L40S/V100s) and is unmodified. This
# file is otherwise cluster-agnostic — no functional changes vs. run_seedset.sh.
#
# A "set" = (method, quality?, problem, difficulty, seed range, n_concurrent).
# This launcher runs the seed range for that identity using at
# most n_concurrent seeds in flight at once, draining the pending list in
# batches until every seed is complete. Seeds whose summary already exists are
# skipped (the Python entry point self-skips, and we also pre-filter), so the
# job is safe to requeue/restart and "completes the sequence" incrementally.
#
# This file wraps the Scientific Coder's Python entry point
# (python -m itcas.cli); it contains NO science.
#
# Usage:
#     # Under Slurm (array): the set index is SLURM_ARRAY_TASK_ID
#     bash scripts/m_run_seedset.sh <jobs.json>
#     # Standalone (pick a specific set):
#     JOB_INDEX=0 bash scripts/m_run_seedset.sh <jobs.json>
#     # As one GPU worker inside m_run_pool.sh's multi-GPU job:
#     JOB_INDEX=<claimed set> CPUS_OVERRIDE=<cpus/gpu> \
#         CUDA_VISIBLE_DEVICES=<local gpu id> bash scripts/m_run_seedset.sh <jobs.json>
# ---------------------------------------------------------------------------
set -u -o pipefail

JOBS_JSON="${1:?usage: m_run_seedset.sh <jobs.json>}"
JOB_INDEX="${JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${JOB_INDEX}" ]]; then
    echo "ERROR: no JOB_INDEX / SLURM_ARRAY_TASK_ID provided." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${JOBS_JSON}" ]]; then
    echo "ERROR: jobs spec '${JOBS_JSON}' not found." >&2
    exit 2
fi

NOTIFY_EMAIL="${NOTIFY_EMAIL:-g.ngo@deakin.edu.au}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"   # retry passes for transient (e.g. OOM) failures

# Expand jobs from either legacy array or new split schema:
#   jobs: [ ... ]
#   jobs: { separate: [ ... ], group: [ ... ] }
# Group entries are expanded as cartesian products of methods x problems x
# difficulties, and quality is only crossed for method=itcas.
JOBS_FILTER='
def arr_or_single($v):
    if ($v | type) == "array" then $v
    elif $v == null then []
    else [$v]
    end;

def expand_group($g):
    ($g | del(.methods, .problems, .difficulties, .qualities, .method, .problem, .difficulty, .quality)) as $base
    | (arr_or_single($g.methods // $g.method)) as $methods
    | (arr_or_single($g.problems // $g.problem)) as $problems
    | (arr_or_single($g.difficulties // $g.difficulty)) as $difficulties
    | ($methods[]?) as $m
    | ($problems[]?) as $p
    | ((if ($difficulties | length) > 0 then $difficulties else [null] end)[]) as $d
    | ((if ($m == "itcas" or $m == "itcas_seq")
            then (arr_or_single($g.qualities // $g.quality) | if length > 0 then . else [null] end)
            else [null]
            end)[]) as $q
    | $base
        + {method: $m, problem: $p}
        + (if $d == null then {} else {difficulty: $d} end)
        + (if (($m == "itcas" or $m == "itcas_seq") and $q != null) then {quality: $q} else {} end);

def expanded_jobs:
    if (.jobs | type) == "array" then .jobs
    elif (.jobs | type) == "object" then
        ((.jobs.separate // []) + [(.jobs.group // [])[]? | expand_group(.)])
    else []
    end;

expanded_jobs
'

# --- Resolve a field for this set, falling back to defaults ----------------
# jq returns the job value if present and non-null, else the default, else null.
jget() {
    local key="$1"
    jq -r --argjson i "${JOB_INDEX}" \
        ". as \$root | (${JOBS_FILTER}) as \$jobs | (\$jobs[\$i][\$k]) // (\$root.defaults[\$k]) // empty" \
        --arg k "${key}" "${JOBS_JSON}"
}

N_JOBS=$(jq "${JOBS_FILTER} | length" "${JOBS_JSON}")
if (( JOB_INDEX < 0 || JOB_INDEX >= N_JOBS )); then
    echo "ERROR: JOB_INDEX=${JOB_INDEX} out of range [0, ${N_JOBS})." >&2
    exit 2
fi

METHOD="$(jget method)"
QUALITY="$(jget quality)"
PROBLEM="$(jget problem)"
DIFFICULTY="$(jget difficulty)"
SEEDS="$(jget seeds)"
N_CONCURRENT="$(jget n_concurrent)"
DEVICE="$(jget device)"
THRESHOLDS_PATH="$(jget thresholds_path)"
RESULTS_ROOT="$(jget results_root)"
EXPERIMENTS_PATH="$(jget experiments_path)"

: "${METHOD:?missing 'method'}" "${PROBLEM:?missing 'problem'}" "${SEEDS:?missing 'seeds'}"
N_CONCURRENT="${N_CONCURRENT:-1}"
DEVICE="${DEVICE:-cuda}"
RESULTS_ROOT="${RESULTS_ROOT:-results/sweep}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-configs/experiments.json}"

# Accept the "method/quality" display form used in configs/final_methods.json
# (e.g. "itcas_seq/ndig") as well as separate method+quality fields. Without
# this, a compound string here would flow straight into RUN_NAME_BASE/OUT_DIR
# and put a stray "/" in the middle of a log filename, breaking every run.
if [[ "${METHOD}" == */* ]]; then
    [[ -z "${QUALITY}" ]] && QUALITY="${METHOD#*/}"
    METHOD="${METHOD%%/*}"
fi
if [[ ( "${METHOD}" == "itcas" || "${METHOD}" == "itcas_seq" ) && -z "${QUALITY}" ]]; then
    QUALITY="roi_mi"
fi

if [[ ! -f "${EXPERIMENTS_PATH}" ]]; then
    echo "ERROR: experiments sizing config '${EXPERIMENTS_PATH}' not found." >&2
    exit 2
fi

# Filesystem-safe difficulty tag (e.g. 0.05 -> p0_05). "none"/empty -> default.
if [[ -z "${DIFFICULTY}" || "${DIFFICULTY}" == "none" || "${DIFFICULTY}" == "null" ]]; then
    DIFF_TAG="default"
    DIFF_KEY="defaults"
    USE_THRESHOLD=0
else
    DIFF_TAG="p$(echo "${DIFFICULTY}" | tr '.' '_')"
    DIFF_KEY="${DIFFICULTY}"
    USE_THRESHOLD=1
fi

# --- Resolve budget/batch_size/n_init (problem+difficulty dependent) --------
# These scale with the problem and difficulty, so they come from the sizing
# config (configs/experiments.json), NOT jobs.json. Resolution per key:
#   problems[<problem>][<difficulty>] -> problems[<problem>].defaults -> defaults
xget() {
    local key="$1"
    jq -r --arg p "${PROBLEM}" --arg d "${DIFF_KEY}" --arg k "${key}" '
        (.problems[$p][$d][$k])
        // (.problems[$p].defaults[$k])
        // (.defaults[$k])
        // empty
    ' "${EXPERIMENTS_PATH}"
}

BUDGET="$(xget budget)"
BATCH_SIZE="$(xget batch_size)"
N_INIT="$(xget n_init)"
EPS_ARCHIVE="$(xget eps_archive)"

: "${BUDGET:?missing 'budget' for ${PROBLEM}/${DIFF_KEY} in ${EXPERIMENTS_PATH}}"
: "${BATCH_SIZE:?missing 'batch_size' for ${PROBLEM}/${DIFF_KEY} in ${EXPERIMENTS_PATH}}"
: "${N_INIT:?missing 'n_init' for ${PROBLEM}/${DIFF_KEY} in ${EXPERIMENTS_PATH}}"

if [[ "${METHOD}" == "itcas" || "${METHOD}" == "itcas_seq" ]]; then
    QUALITY_TAG="__${QUALITY:-roi_mi}"
    METHOD_DIR="${METHOD}/${QUALITY:-roi_mi}"
else
    QUALITY_TAG=""
    METHOD_DIR="${METHOD}"
fi

RUN_NAME_BASE="${PROBLEM}__${METHOD}${QUALITY_TAG}__${DIFF_TAG}"
OUT_DIR="${RESULTS_ROOT}/${PROBLEM}/${DIFF_TAG}/${METHOD_DIR}"
mkdir -p "${OUT_DIR}"

# --- Environment: load Anaconda3 + activate the conda env ------------------
# Mirrors the cluster's recommended pattern (module load Anaconda3; conda
# activate <env>). Set CONDA_ENV to the project environment name. Alternatively
# point ITCAS_VENV at a venv. If neither is set, the system Python is used.
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE:-Anaconda3}" 2>/dev/null || true
    # 'source activate' makes 'conda activate' available in non-login shells.
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" || {
        echo "ERROR: failed to 'conda activate ${CONDA_ENV}'." >&2; exit 3; }
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi

# --- Thread pinning: split the task's CPUs across the concurrent seeds ------
# Also isolate from ~/.local user-site so the conda env's (CUDA-matched) torch
# is used, never a stray ~/.local build.
export PYTHONNOUSERSITE=1
export PIP_USER=0
# CPUS_OVERRIDE lets a multi-GPU pool launcher (m_run_pool.sh) tell this
# process its per-GPU share of the job's CPUs. Without it, every GPU worker
# in the same Slurm job would read the job's FULL $SLURM_CPUS_PER_TASK and
# each spawn that many OMP/MKL threads, oversubscribing the CPUs by a factor
# of GPUS_PER_JOB.
CPUS="${CPUS_OVERRIDE:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 1)}}"
PER_SEED_THREADS=$(( CPUS / N_CONCURRENT ))
(( PER_SEED_THREADS < 1 )) && PER_SEED_THREADS=1
export OMP_NUM_THREADS="${PER_SEED_THREADS}"
export MKL_NUM_THREADS="${PER_SEED_THREADS}"
export OPENBLAS_NUM_THREADS="${PER_SEED_THREADS}"
export NUMEXPR_NUM_THREADS="${PER_SEED_THREADS}"

echo "===================================================================="
echo "ITCAS seed-set | set ${JOB_INDEX}/${N_JOBS} | array ${SLURM_ARRAY_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA}"
echo "method=${METHOD} quality=${QUALITY:-none} problem=${PROBLEM} difficulty=${DIFFICULTY:-default} seeds=${SEEDS}"
echo "budget=${BUDGET} batch_size=${BATCH_SIZE} n_init=${N_INIT} eps_archive=${EPS_ARCHIVE:-default} (from ${EXPERIMENTS_PATH})"
echo "n_concurrent=${N_CONCURRENT} device=${DEVICE} threads/seed=${PER_SEED_THREADS}"
echo "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-NA} out_dir=${OUT_DIR}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Helpers ---------------------------------------------------------------
# Pending seeds = target range minus already-completed runs. Reuses the
# Scientific Coder's parser/completion logic so naming stays consistent.
compute_pending() {
    python - "$SEEDS" "$OUT_DIR" "$RUN_NAME_BASE" <<'PY'
import sys
from itcas.utils.seeds import parse_seed_spec, pending_seeds
spec, out_dir, base = sys.argv[1], sys.argv[2], sys.argv[3]
print(" ".join(str(s) for s in pending_seeds(parse_seed_spec(spec), out_dir, base)))
PY
}

run_one_seed() {
    local s="$1"
    local log="${OUT_DIR}/${RUN_NAME_BASE}_seed${s}.run.log"
    local quality_args=()
    local thr=()
    local eps_args=()
    if [[ -n "${EPS_ARCHIVE:-}" && "${EPS_ARCHIVE}" != "null" ]]; then
        eps_args=(--eps_archive "${EPS_ARCHIVE}")
    fi
    if [[ "${USE_THRESHOLD}" -eq 1 ]]; then
        thr=(--threshold_pct "${DIFFICULTY}" --thresholds_path "${THRESHOLDS_PATH}")
    fi
    echo "[seed ${s}] start $(date '+%H:%M:%S') -> ${log}"
    if [[ "${METHOD}" == "itcas" || "${METHOD}" == "itcas_seq" ]]; then
        quality_args=(--quality "${QUALITY:-roi_mi}")
    fi
    python -m itcas.cli \
        --method   "${METHOD}" \
        --problem  "${PROBLEM}" \
        --budget   "${BUDGET}" \
        --batch_size "${BATCH_SIZE}" \
        --n_init   "${N_INIT}" \
        "${quality_args[@]}" \
        "${eps_args[@]}" \
        --device   "${DEVICE}" \
        --seeds    "${s}" \
        "${thr[@]}" \
        --out_dir  "${OUT_DIR}" \
        --run_name "${RUN_NAME_BASE}" \
        > "${log}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        echo "[seed ${s}] done  $(date '+%H:%M:%S')"
    else
        echo "[seed ${s}] FAILED rc=${rc} (see ${log})" >&2
    fi
    return ${rc}
}

# --- Optional e-mail at set completion -------------------------------------
send_email() {
    local status="$1" pend_left="$2" runtime="$3"
    command -v mail >/dev/null 2>&1 || return 0
    [[ -z "${NOTIFY_EMAIL}" ]] && return 0
    local subject body
    subject="[ITCAS] ${status} — ${METHOD}/${PROBLEM} diff=${DIFFICULTY:-default} seeds=${SEEDS}"
    body="$(cat <<EOF
ITCAS seed-set notification
===========================
Status        : ${status}
Method        : ${METHOD}
Problem       : ${PROBLEM}
Difficulty    : ${DIFFICULTY:-default}
Seed range    : ${SEEDS}
Concurrency   : ${N_CONCURRENT} seeds/GPU
Seeds left    : ${pend_left}
Runtime       : ${runtime}s

Slurm job     : ${SLURM_JOB_ID:-NA} (array ${SLURM_ARRAY_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-${JOB_INDEX}})
Node          : $(hostname)   GPU(s): ${CUDA_VISIBLE_DEVICES:-NA}
Output dir    : ${OUT_DIR}
EOF
)"
    printf '%s\n' "${body}" | mail -s "${subject}" "${NOTIFY_EMAIL}" \
        && echo "[run_seedset] e-mail sent to ${NOTIFY_EMAIL} (${status})" \
        || echo "[run_seedset] WARNING: 'mail' failed." >&2
}

# --- Drain pending seeds in N-wide batches, retrying transient failures -----
START_TS=$(date +%s)
attempt=0
while :; do
    read -r -a PENDING <<< "$(compute_pending)"
    if [[ ${#PENDING[@]} -eq 0 ]]; then
        break
    fi
    attempt=$(( attempt + 1 ))
    if (( attempt > MAX_ATTEMPTS )); then
        echo "[run_seedset] giving up after ${MAX_ATTEMPTS} pass(es); still pending: ${PENDING[*]}" >&2
        break
    fi
    echo "[run_seedset] pass ${attempt}/${MAX_ATTEMPTS}: ${#PENDING[@]} pending seed(s): ${PENDING[*]}"

    inflight=0
    for s in "${PENDING[@]}"; do
        run_one_seed "${s}" &
        inflight=$(( inflight + 1 ))
        if (( inflight >= N_CONCURRENT )); then
            wait -n || true
            inflight=$(( inflight - 1 ))
        fi
    done
    wait
done

# --- Final accounting ------------------------------------------------------
read -r -a LEFT <<< "$(compute_pending)"
RUNTIME=$(( $(date +%s) - START_TS ))
echo "===================================================================="
if [[ ${#LEFT[@]} -eq 0 ]]; then
    echo "[run_seedset] COMPLETED set ${JOB_INDEX}: all seeds in '${SEEDS}' done (${RUNTIME}s)."
    send_email "COMPLETED" "0" "${RUNTIME}"
    echo "===================================================================="
    exit 0
else
    echo "[run_seedset] INCOMPLETE set ${JOB_INDEX}: ${#LEFT[@]} seed(s) still pending: ${LEFT[*]} (${RUNTIME}s)." >&2
    send_email "FAILED" "${#LEFT[@]}" "${RUNTIME}"
    echo "===================================================================="
    exit 1
fi
