#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# m_submit.sh — main entry (cluster "m", A100 GPUs): read the JSON job spec
# and submit the GPU sweep as one big multi-GPU job (or a small array of
# them), NOT one job per set.
#
# Sibling of submit.sh for the "m" cluster (A100s). submit.sh remains the
# entry point for the "f" cluster (L40S/V100s), where every set gets its own
# single-GPU array task — that model is intentionally NOT used here. Cluster
# "m" convention favors fewer, bigger jobs: this script submits ONE Slurm job
# requesting GPUS_PER_JOB A100s on a single node (or, if NUM_BIG_JOBS>1, a
# small array of such jobs). m_submit_jobs.sbatch -> m_run_pool.sh then
# spawns one background worker per GPU inside that job; each worker
# repeatedly claims and fully drains the next unclaimed set from the JSON
# spec until every set is done, so GPUS_PER_JOB GPUs stay busy across
# however many sets exist, however that compares to GPUS_PER_JOB.
#
# GPUS_PER_JOB default (4, i.e. half an A100 node): the "m" cluster's 3 A100
# nodes have 8 GPUs each but are usually only partially idle (mixed use by
# other jobs), so requesting a full node (8) tends to queue far longer than
# requesting half a node, which Slurm can co-schedule alongside other jobs on
# the same node. 4 GPUs is still a "big job" (4x fewer submissions than one
# job per set) while leaving headroom. Bump to 8 for a quiet-cluster burst
# run, or drop to 1-2 if the cluster is very busy; check current headroom
# with `sinfo -N -o "%N %P %G"` / `scontrol show node <a100 node>` first.
#
# Usage:
#     scripts/m_submit.sh <jobs.json> [extra sbatch args...]
#
# Environment overrides:
#     GPUS_PER_JOB=4          # A100 GPUs requested per big job. Default 4.
#     NUM_BIG_JOBS=1          # how many such jobs to submit concurrently
#                              # (as a small array); total GPUs in flight =
#                              # GPUS_PER_JOB * NUM_BIG_JOBS. Default 1.
#     PARTITION=gpu           # GPU partition. Default 'gpu'.
#     QOS=batch-short         # batch-short (<=5d) or batch-long (<=10d).
#     TIME=3-00:00:00         # per-job wallclock limit (D-HH:MM:SS). Scale
#                              # with ceil(num_sets / GPUS_PER_JOB) waves of
#                              # ~3-5h each (one set's worth of seeds); use
#                              # QOS=batch-long for very large sweeps.
#     CONDA_ENV=itcas         # conda env to activate — both in THIS script
#                              # (so its own `jq` calls below resolve; installs
#                              # happen only in install_env.sbatch's compute
#                              # job, never here, but activating an already-
#                              # built env is a cheap, login-node-safe PATH
#                              # change) and, via --export=ALL, inside the
#                              # submitted job itself.
#     ACCOUNT=...             # only if your site requires one (not needed here).
#     DRY_RUN=1               # print the sbatch command without submitting.
#
# Examples:
#     CONDA_ENV=itcas scripts/m_submit.sh scripts/jobs.json
#     GPUS_PER_JOB=8 CONDA_ENV=itcas scripts/m_submit.sh scripts/jobs.json
#     GPUS_PER_JOB=4 NUM_BIG_JOBS=2 CONDA_ENV=itcas scripts/m_submit.sh scripts/jobs.json
# ---------------------------------------------------------------------------
set -euo pipefail

JOBS_JSON="${1:?usage: m_submit.sh <jobs.json> [extra sbatch args...]}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

[[ -f "${JOBS_JSON}" ]] || { echo "ERROR: '${JOBS_JSON}' not found." >&2; exit 2; }

# Activate CONDA_ENV in THIS shell (not just forward it to the job) so `jq`
# — installed into the env by install_env.sbatch's compute-node job, per
# policy: package installs never run on the login node — is on PATH for the
# preflight check and job-listing below, which run here on the login node
# before anything is submitted. Mirrors the activation block in
# m_run_seedset.sh. Because sbatch is invoked later in this same process
# with --export=ALL, the resulting PATH (env bin/ prepended) also reaches
# the job, so m_run_pool.sh's own `jq` calls resolve there too.
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE:-Anaconda3}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" || {
        echo "ERROR: failed to 'conda activate ${CONDA_ENV}'." >&2; exit 3; }
fi

command -v jq >/dev/null 2>&1 || { echo "ERROR: jq is required (set CONDA_ENV=<env with jq installed>, e.g. via install_env.sbatch)." >&2; exit 2; }

# Validate the spec and count sets after expanding jobs (purely informational
# here — unlike submit.sh, this count no longer sizes the Slurm array; it's
# printed so you can sanity-check GPUS_PER_JOB/NUM_BIG_JOBS/TIME against it).
# Supported schemas:
#   1) legacy: jobs is an array of explicit entries
#   2) new:    jobs.separate (explicit) + jobs.group (cartesian expansion)
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
    | ((if $m == "itcas"
            then (arr_or_single($g.qualities // $g.quality) | if length > 0 then . else [null] end)
            else [null]
            end)[]) as $q
    | $base
        + {method: $m, problem: $p}
        + (if $d == null then {} else {difficulty: $d} end)
        + (if ($m == "itcas" and $q != null) then {quality: $q} else {} end);

def expanded_jobs:
    if (.jobs | type) == "array" then .jobs
    elif (.jobs | type) == "object" then
        ((.jobs.separate // []) + [(.jobs.group // [])[]? | expand_group(.)])
    else []
    end;

expanded_jobs
'

K=$(jq -e "${JOBS_FILTER} | length" "${JOBS_JSON}") || {
        echo "ERROR: '${JOBS_JSON}' has invalid '.jobs' schema." >&2; exit 2; }
(( K > 0 )) || { echo "ERROR: '.jobs' is empty." >&2; exit 2; }

GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
NUM_BIG_JOBS="${NUM_BIG_JOBS:-1}"
PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-batch-short}"
TIME="${TIME:-3-00:00:00}"

# Per-GPU CPU/mem budget matches the "f" cluster's per-set budget (16 cpus,
# 32G mem for n_concurrent=8 seeds/GPU with 2 OMP threads each), scaled by
# GPUS_PER_JOB since this job holds that many GPUs at once.
CPUS_PER_JOB=$(( 16 * GPUS_PER_JOB ))
MEM_PER_JOB_GB=$(( 32 * GPUS_PER_JOB ))

mkdir -p slurm_logs

# Snapshot the resolved spec into an immutable copy for this submission.
# m_run_pool.sh re-reads its JSON argument fresh every time a worker claims a
# new set (workers can start claiming hours into a long job), so if the
# caller keeps editing/overwriting JOBS_JSON in place after submitting, later
# claims would read a different job count/content than the pool was sized
# for. Submitting a private snapshot instead of JOBS_JSON directly makes the
# job immune to that race.
SPEC_DIR="slurm_logs/job_specs"
mkdir -p "${SPEC_DIR}"

# Assemble sbatch directives. Account is only added when explicitly provided.
SBATCH_ARGS=(
    --partition="${PARTITION}"
    --qos="${QOS}"
    --gres="gpu:${GPUS_PER_JOB}"
    --cpus-per-task="${CPUS_PER_JOB}"
    --mem="${MEM_PER_JOB_GB}G"
    --time="${TIME}"
)
# Only wrap in an array when submitting more than one big job; a single big
# job stays a plain (non-array) submission, matching the "m" cluster's "one
# big job" convention literally in the common case.
if (( NUM_BIG_JOBS > 1 )); then
    SBATCH_ARGS+=( --array="0-$((NUM_BIG_JOBS-1))" )
fi
[[ -n "${ACCOUNT:-}" ]] && SBATCH_ARGS+=( --account="${ACCOUNT}" )

# Forward CONDA_ENV (and optional module name) to the job environment.
EXPORTS="ALL"
[[ -n "${CONDA_ENV:-}"    ]] && EXPORTS="${EXPORTS},CONDA_ENV=${CONDA_ENV}"
[[ -n "${CONDA_MODULE:-}" ]] && EXPORTS="${EXPORTS},CONDA_MODULE=${CONDA_MODULE}"
SBATCH_ARGS+=( --export="${EXPORTS}" )

TOTAL_GPUS=$(( GPUS_PER_JOB * NUM_BIG_JOBS ))
echo "[m_submit] spec=${JOBS_JSON}  sets=${K}"
echo "[m_submit] gpus_per_job=${GPUS_PER_JOB}  num_big_jobs=${NUM_BIG_JOBS}  total_gpus_in_flight=${TOTAL_GPUS}"
echo "[m_submit] partition=${PARTITION}  qos=${QOS}  time=${TIME}  cpus/job=${CPUS_PER_JOB}  mem/job=${MEM_PER_JOB_GB}G"
echo "[m_submit] conda_env=${CONDA_ENV:-<unset: using system Python>}  account=${ACCOUNT:-<none>}"
jq -r "${JOBS_FILTER} | to_entries[] | \"  [\\(.key)] \\(.value.method) / \\(.value.problem) quality=\\(.value.quality // \"default\") diff=\\(.value.difficulty // \"default\") seeds=\\(.value.seeds) N=\\(.value.n_concurrent // \"default\")\"" "${JOBS_JSON}"

if [[ -n "${DRY_RUN:-}" ]]; then
    SPEC_SNAPSHOT="${SPEC_DIR}/jobs_<snapshot>.json"
    CMD=( sbatch "${SBATCH_ARGS[@]}" "$@" "${SCRIPT_DIR}/m_submit_jobs.sbatch" "${SPEC_SNAPSHOT}" "${GPUS_PER_JOB}" )
    echo "[m_submit] DRY_RUN — '${JOBS_JSON}' would be snapshotted to '${SPEC_SNAPSHOT}' and this run:"
    printf '  %q ' "${CMD[@]}"; echo
    exit 0
fi

if [[ -z "${CONDA_ENV:-}" ]]; then
    echo "[m_submit] NOTE: CONDA_ENV unset; the job will use the node's system Python." >&2
fi

SPEC_SNAPSHOT="$(mktemp --suffix=.json "${SPEC_DIR}/jobs_XXXXXX")"
cp "${JOBS_JSON}" "${SPEC_SNAPSHOT}"
echo "[m_submit] snapshotted spec -> ${SPEC_SNAPSHOT} (immune to later edits of '${JOBS_JSON}')"

CMD=( sbatch "${SBATCH_ARGS[@]}" "$@" "${SCRIPT_DIR}/m_submit_jobs.sbatch" "${SPEC_SNAPSHOT}" "${GPUS_PER_JOB}" )

echo "[m_submit] submitting..."
SBATCH_OUT="$("${CMD[@]}")"
echo "${SBATCH_OUT}"

JOBID="$(grep -oE '[0-9]+' <<<"${SBATCH_OUT}" | head -1)"
if [[ -n "${JOBID}" ]]; then
    ln -s "$(basename "${SPEC_SNAPSHOT}")" "${SPEC_DIR}/jobs_${JOBID}.json"
    echo "[m_submit] spec snapshot aliased -> ${SPEC_DIR}/jobs_${JOBID}.json"
fi
