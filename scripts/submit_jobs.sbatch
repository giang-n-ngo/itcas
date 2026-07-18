#!/usr/bin/env bash
#SBATCH --job-name=itcas_gpu
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/itcas_gpu_%A_%a.out
#SBATCH --error=slurm_logs/itcas_gpu_%A_%a.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# submit_jobs.sbatch — ITCAS GPU sweep as a Slurm job ARRAY.
#
#   * One array task == one "set" from the JSON spec == one (method, problem,
#     difficulty) run over a seed range.
#   * Each task requests EXACTLY 1 GPU (--gpus=1) and runs n_concurrent seeds
#     at a time on that GPU, iterating batches until the range is done.
#   * The array index selects jobs[<index>] in the JSON.
#
# Resource rationale:
#   - --partition=gpu, --gpus=1 : each job is capped at 1 GPU (request a typed
#     GPU with GPUS=v100:1 via submit.sh if needed, matching --gpus=<type>:N).
#   - MAX_GPUS (default 48 in submit.sh) controls the %N array throttle; set it
#     to the maximum number of concurrent GPU jobs your quota allows.
#   - --qos=batch-short : batch jobs up to 5 days; --time=1-00:00:00 is a safe
#     upper bound for 20 seeds with n_concurrent=8 (~3-5h per job).
#   - --cpus-per-task=16 / --mem=32G : supports n_concurrent=8 seeds in flight
#     with 2 OMP threads each (16/8=2); 32G gives 4 GB per seed for GP models.
#
# Environment: the launcher (run_seedset.sh) loads Anaconda3 + activates the
#   conda env named by CONDA_ENV (mirrors the cluster's recommended pattern:
#   module load Anaconda3; conda activate <env>). Set CONDA_ENV before submit.
#
# Submit (preferred — auto-sizes the array AND snapshots the JSON spec so
# in-flight/pending tasks are immune to later edits of scripts/jobs.json):
#     scripts/submit.sh scripts/jobs.json
# or manually (NOTE: no snapshot — if you edit/overwrite the JSON path you
# pass here while the array still has pending tasks, those tasks will read
# the new file and can fail with "JOB_INDEX out of range"; pass a copy of
# the JSON instead of the live path if you need to keep editing it):
#     K=$(jq '
#       def arr_or_single($v): if ($v|type)=="array" then $v elif $v==null then [] else [$v] end;
#       def expand_group($g):
#         ($g | del(.methods,.problems,.difficulties,.qualities,.method,.problem,.difficulty,.quality)) as $base
#         | (arr_or_single($g.methods // $g.method)) as $ms
#         | (arr_or_single($g.problems // $g.problem)) as $ps
#         | (arr_or_single($g.difficulties // $g.difficulty)) as $ds
#         | ($ms[]?) as $m | ($ps[]?) as $p
#         | ((if ($ds|length)>0 then $ds else [null] end)[]) as $d
#         | ((if $m=="itcas" then (arr_or_single($g.qualities // $g.quality) | if length>0 then . else [null] end) else [null] end)[]) as $q
#         | $base + {method:$m,problem:$p}
#             + (if $d==null then {} else {difficulty:$d} end)
#             + (if ($m=="itcas" and $q!=null) then {quality:$q} else {} end);
#       if (.jobs|type)=="array" then .jobs
#       elif (.jobs|type)=="object" then ((.jobs.separate // []) + [(.jobs.group // [])[]? | expand_group(.)])
#       else [] end
#       | length' scripts/jobs.json)
#     sbatch --array=0-$((K-1))%2 scripts/submit_jobs.sbatch scripts/jobs.json
# ---------------------------------------------------------------------------
set -u -o pipefail

JOBS_JSON="${1:?usage: sbatch submit_jobs.sbatch <jobs.json>}"

# Slurm copies the batch script to a spool dir, so BASH_SOURCE does NOT point
# at the repo. Use SLURM_SUBMIT_DIR (the dir sbatch was invoked from) as the
# repo root; fall back to BASH_SOURCE for direct (non-sbatch) execution.
# SCRIPT_DIR must therefore be derived from the repo, not BASH_SOURCE, so the
# srun call below finds run_seedset.sh in the real checkout (not the spool).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/scripts"
cd "${REPO_ROOT}"
mkdir -p slurm_logs 2>/dev/null || \
    echo "[submit_jobs] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

echo "===================================================================="
echo "ITCAS GPU sweep | array ${SLURM_ARRAY_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA}"
echo "spec=${JOBS_JSON} host=$(hostname) partition=${SLURM_JOB_PARTITION:-NA}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

export JOB_INDEX="${SLURM_ARRAY_TASK_ID}"
# Reduce CUDA allocator fragmentation when multiple seeds share a single GPU
# (n_concurrent > 1). Without this, large transient kernel-matrix allocations
# can fail to find a contiguous block even when total free memory is sufficient.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
srun --kill-on-bad-exit=1 --gres=gpu:1 bash "${SCRIPT_DIR}/run_seedset.sh" "${JOBS_JSON}"
RC=$?

echo "===================================================================="
echo "set ${SLURM_ARRAY_TASK_ID:-NA} finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
exit "${RC}"
