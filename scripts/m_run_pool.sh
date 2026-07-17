#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# m_run_pool.sh — drain ALL sets in the JSON spec using a pool of GPU workers
# inside ONE Slurm job (cluster "m", A100s).
#
# Cluster "m" convention: submit one big multi-GPU job instead of many
# single-GPU array jobs (the "f" cluster's model, kept in run_seedset.sh /
# submit_jobs.sbatch / submit.sh, unmodified). m_submit_jobs.sbatch requests
# GPUS_PER_JOB GPUs on a single node and calls this script once; this script
# spawns GPUS_PER_JOB background workers, one per allocated GPU, each of
# which repeatedly claims the next unclaimed "set" (one method/problem/
# difficulty over a seed range) from the JSON spec and runs it to completion
# via m_run_seedset.sh before claiming the next one — so all GPUs stay busy
# until every set is done, regardless of how sets outnumber GPUs.
#
# Claiming uses atomic `mkdir` locks under a per-job claim directory, so
# concurrent workers (and, if NUM_BIG_JOBS>1, concurrent array tasks sharing
# the same claim directory via SLURM_ARRAY_JOB_ID) never double-process a
# set. Claims are scoped to one Slurm submission (keyed by job/array ID) and
# are NOT required for correctness across resubmits: m_run_seedset.sh already
# skips seeds with existing results, so a fresh submission simply re-derives
# "pending" work from what's actually on disk.
#
# Usage (invoked by m_submit_jobs.sbatch, not run standalone):
#     bash scripts/m_run_pool.sh <jobs.json> <gpus_per_job>
# ---------------------------------------------------------------------------
set -u -o pipefail

JOBS_JSON="${1:?usage: m_run_pool.sh <jobs.json> <gpus_per_job>}"
GPUS_PER_JOB="${2:?usage: m_run_pool.sh <jobs.json> <gpus_per_job>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${JOBS_JSON}" ]]; then
    echo "ERROR: jobs spec '${JOBS_JSON}' not found." >&2
    exit 2
fi

# Same set-expansion filter as m_submit.sh / m_run_seedset.sh, duplicated
# here to keep each script standalone (no shared-sourcing across the
# f/m script families).
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

N_JOBS=$(jq "${JOBS_FILTER} | length" "${JOBS_JSON}")
if [[ -z "${N_JOBS}" || "${N_JOBS}" -le 0 ]]; then
    echo "ERROR: '${JOBS_JSON}' expands to 0 sets." >&2
    exit 2
fi

# --- Resolve the GPUs actually visible to this job/step ---------------------
# Slurm sets CUDA_VISIBLE_DEVICES to the device id(s) granted to this job
# (cgroup-remapped to "0..N-1" on most configs, but read it rather than
# assume, in case a site leaves physical ids e.g. "2,5,6,7"). Fall back to
# 0..GPUS_PER_JOB-1 if unset (e.g. manual non-Slurm testing).
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
    VISIBLE_GPUS=()
    for ((i = 0; i < GPUS_PER_JOB; i++)); do VISIBLE_GPUS+=("${i}"); done
fi
if [[ "${#VISIBLE_GPUS[@]}" -lt "${GPUS_PER_JOB}" ]]; then
    echo "ERROR: only ${#VISIBLE_GPUS[@]} GPU(s) visible (${VISIBLE_GPUS[*]:-none}) but GPUS_PER_JOB=${GPUS_PER_JOB}." >&2
    exit 2
fi

# --- Per-worker CPU share ----------------------------------------------------
# The job's whole --cpus-per-task is for ALL GPUS_PER_JOB workers combined;
# split evenly so each worker's OMP/MKL thread count (set inside
# m_run_seedset.sh via CPUS_OVERRIDE) doesn't oversubscribe the CPUs.
JOB_CPUS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo "${GPUS_PER_JOB}")}"
CPUS_PER_WORKER=$(( JOB_CPUS / GPUS_PER_JOB ))
(( CPUS_PER_WORKER < 1 )) && CPUS_PER_WORKER=1

# --- Claim directory (shared by every worker of this job, and by every ------
# array task of this submission if NUM_BIG_JOBS>1) --------------------------
CLAIM_DIR="${REPO_ROOT}/slurm_logs/m_claims/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-standalone}}"
mkdir -p "${CLAIM_DIR}"

claim_next_set() {
    local idx
    for ((idx = 0; idx < N_JOBS; idx++)); do
        if mkdir "${CLAIM_DIR}/set_${idx}.claim" 2>/dev/null; then
            echo "${idx}"
            return 0
        fi
    done
    return 1
}

echo "===================================================================="
echo "ITCAS GPU pool | job ${SLURM_JOB_ID:-NA} array ${SLURM_ARRAY_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA}"
echo "spec=${JOBS_JSON} sets=${N_JOBS} gpus_per_job=${GPUS_PER_JOB} visible_gpus=${VISIBLE_GPUS[*]}"
echo "cpus/job=${JOB_CPUS} cpus/worker=${CPUS_PER_WORKER} host=$(hostname) claim_dir=${CLAIM_DIR}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# Reduce CUDA allocator fragmentation when multiple seeds share a single GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

worker() {
    local local_idx="$1"
    local gpu_id="${VISIBLE_GPUS[${local_idx}]}"
    local result_file="${CLAIM_DIR}/worker_${local_idx}.failed_sets"
    : > "${result_file}"
    local claimed n_done=0
    while claimed=$(claim_next_set); do
        echo "[pool worker ${local_idx}] gpu=${gpu_id} claims set ${claimed}/${N_JOBS} at $(date '+%H:%M:%S')"
        CUDA_VISIBLE_DEVICES="${gpu_id}" CPUS_OVERRIDE="${CPUS_PER_WORKER}" \
            JOB_INDEX="${claimed}" bash "${SCRIPT_DIR}/m_run_seedset.sh" "${JOBS_JSON}"
        if [[ $? -ne 0 ]]; then
            echo "${claimed}" >> "${result_file}"
        fi
        n_done=$(( n_done + 1 ))
    done
    echo "[pool worker ${local_idx}] gpu=${gpu_id} done: processed ${n_done} set(s), no sets left to claim."
}

PIDS=()
for ((w = 0; w < GPUS_PER_JOB; w++)); do
    worker "${w}" &
    PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAIL=1
done

# --- Aggregate failures across all workers -----------------------------------
FAILED_SETS=()
for ((w = 0; w < GPUS_PER_JOB; w++)); do
    f="${CLAIM_DIR}/worker_${w}.failed_sets"
    [[ -s "${f}" ]] && FAILED_SETS+=($(cat "${f}"))
done

echo "===================================================================="
if [[ "${#FAILED_SETS[@]}" -eq 0 && "${FAIL}" -eq 0 ]]; then
    echo "[pool] COMPLETED: all ${N_JOBS} set(s) finished (across ${GPUS_PER_JOB} GPU workers)."
    echo "===================================================================="
    exit 0
else
    echo "[pool] INCOMPLETE: ${#FAILED_SETS[@]} set(s) still failing: ${FAILED_SETS[*]:-none}." >&2
    echo "===================================================================="
    exit 1
fi
