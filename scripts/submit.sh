#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# submit.sh — main entry: read the JSON job spec and submit the GPU sweep.
#
# Reads <jobs.json>, counts the sets, and submits ONE Slurm array where each
# array task runs one set (one method/problem/difficulty over a seed range) on
# a single GPU. The array is throttled so no more than MAX_GPUS sets run at
# once.
#
# Usage:
#     scripts/submit.sh <jobs.json> [extra sbatch args...]
#
# Environment overrides:
#     MAX_GPUS=48            # max concurrent array tasks (=GPUs). Default 48.
#     PARTITION=gpu         # GPU partition. Default 'gpu'.
#     QOS=batch-short       # batch-short (<=5d) or batch-long (<=10d).
#     TIME=1-00:00:00       # per-job wallclock limit (D-HH:MM:SS).
#     GPUS=1                # GPUs per job; type-qualify as v100:1 if required.
#     CONDA_ENV=itcas       # conda env to activate on the node (recommended).
#     ACCOUNT=...           # only if your site requires one (not needed here).
#     DRY_RUN=1             # print the sbatch command without submitting.
#
# Examples:
#     CONDA_ENV=itcas scripts/submit.sh scripts/jobs.json
#     PARTITION=gpu GPUS=v100:1 MAX_GPUS=2 CONDA_ENV=itcas scripts/submit.sh scripts/jobs.json
# ---------------------------------------------------------------------------
set -euo pipefail

JOBS_JSON="${1:?usage: submit.sh <jobs.json> [extra sbatch args...]}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

[[ -f "${JOBS_JSON}" ]] || { echo "ERROR: '${JOBS_JSON}' not found." >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "ERROR: jq is required." >&2; exit 2; }

# Validate the spec and count sets after expanding jobs.
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

MAX_GPUS="${MAX_GPUS:-48}"
PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-batch-short}"
TIME="${TIME:-1-00:00:00}"
GPUS="${GPUS:-1}"

mkdir -p slurm_logs

# Snapshot the resolved spec into an immutable copy for this submission.
# run_seedset.sh re-reads its JSON argument fresh on every array-task start
# (pending tasks can start hours/days after submission), so if the caller
# keeps editing/overwriting JOBS_JSON in place after submitting, later tasks
# read a different job count/content than the array was sized for and die
# with "JOB_INDEX out of range". Submitting a private snapshot instead of
# JOBS_JSON directly makes the array immune to that race.
SPEC_DIR="slurm_logs/job_specs"
mkdir -p "${SPEC_DIR}"

# Assemble sbatch directives. Account is only added when explicitly provided.
SBATCH_ARGS=(
    --array="0-$((K-1))%${MAX_GPUS}"
    --partition="${PARTITION}"
    --qos="${QOS}"
    --gpus="${GPUS}"
    --time="${TIME}"
)
[[ -n "${ACCOUNT:-}" ]] && SBATCH_ARGS+=( --account="${ACCOUNT}" )

# Forward CONDA_ENV (and optional module name) to the job environment.
EXPORTS="ALL"
[[ -n "${CONDA_ENV:-}"    ]] && EXPORTS="${EXPORTS},CONDA_ENV=${CONDA_ENV}"
[[ -n "${CONDA_MODULE:-}" ]] && EXPORTS="${EXPORTS},CONDA_MODULE=${CONDA_MODULE}"
SBATCH_ARGS+=( --export="${EXPORTS}" )

echo "[submit] spec=${JOBS_JSON}  sets=${K}"
echo "[submit] array=0-$((K-1))%${MAX_GPUS}  partition=${PARTITION}  qos=${QOS}  gpus=${GPUS}  time=${TIME}"
echo "[submit] conda_env=${CONDA_ENV:-<unset: using system Python>}  account=${ACCOUNT:-<none>}"
jq -r "${JOBS_FILTER} | to_entries[] | \"  [\\(.key)] \\(.value.method) / \\(.value.problem) quality=\\(.value.quality // \"default\") diff=\\(.value.difficulty // \"default\") seeds=\\(.value.seeds) N=\\(.value.n_concurrent // \"default\")\"" "${JOBS_JSON}"

if [[ -n "${DRY_RUN:-}" ]]; then
    SPEC_SNAPSHOT="${SPEC_DIR}/jobs_<snapshot>.json"
    CMD=( sbatch "${SBATCH_ARGS[@]}" "$@" "${SCRIPT_DIR}/submit_jobs.sbatch" "${SPEC_SNAPSHOT}" )
    echo "[submit] DRY_RUN — '${JOBS_JSON}' would be snapshotted to '${SPEC_SNAPSHOT}' and this run:"
    printf '  %q ' "${CMD[@]}"; echo
    exit 0
fi

if [[ -z "${CONDA_ENV:-}" ]]; then
    echo "[submit] NOTE: CONDA_ENV unset; the job will use the node's system Python." >&2
fi

SPEC_SNAPSHOT="$(mktemp --suffix=.json "${SPEC_DIR}/jobs_XXXXXX")"
cp "${JOBS_JSON}" "${SPEC_SNAPSHOT}"
echo "[submit] snapshotted spec -> ${SPEC_SNAPSHOT} (immune to later edits of '${JOBS_JSON}')"

CMD=( sbatch "${SBATCH_ARGS[@]}" "$@" "${SCRIPT_DIR}/submit_jobs.sbatch" "${SPEC_SNAPSHOT}" )

echo "[submit] submitting..."
SBATCH_OUT="$("${CMD[@]}")"
echo "${SBATCH_OUT}"

JOBID="$(grep -oE '[0-9]+' <<<"${SBATCH_OUT}" | head -1)"
if [[ -n "${JOBID}" ]]; then
    ln -s "$(basename "${SPEC_SNAPSHOT}")" "${SPEC_DIR}/jobs_${JOBID}.json"
    echo "[submit] spec snapshot aliased -> ${SPEC_DIR}/jobs_${JOBID}.json"
fi
