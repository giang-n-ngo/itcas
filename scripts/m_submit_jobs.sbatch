#!/usr/bin/env bash
#SBATCH --job-name=itcas_gpu_m
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --constraint=gpu-a100
#SBATCH --output=slurm_logs/itcas_gpu_m_%j.out
#SBATCH --error=slurm_logs/itcas_gpu_m_%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# m_submit_jobs.sbatch — ITCAS GPU sweep as ONE multi-GPU Slurm job, cluster
# "m" (A100 GPUs).
#
# Sibling of submit_jobs.sbatch for the "m" cluster. submit_jobs.sbatch
# remains the entry point for the "f" cluster (L40S/V100s, one array task ==
# one set on one GPU) and is unmodified.
#
# Cluster "m" convention is the opposite: submit ONE big job holding
# GPUS_PER_JOB GPUs on a single node (not many single-GPU jobs). This script
# requests that GPU count (via --gres/--cpus-per-task/--mem passed on the
# CLI by m_submit.sh, NOT hardcoded here — they scale with GPUS_PER_JOB) and
# hands off to m_run_pool.sh, which spawns one background worker per GPU;
# each worker repeatedly claims and fully drains the next unclaimed set
# (method/problem/difficulty over a seed range, n_concurrent seeds at a time)
# until every set in the JSON spec is done. If m_submit.sh's NUM_BIG_JOBS>1,
# this script is submitted as a small array where each task independently
# requests GPUS_PER_JOB GPUs and all tasks cooperate over the same set list
# via a shared claim directory (keyed by SLURM_ARRAY_JOB_ID).
#
# Resource rationale (see m_submit.sh for how these scale with GPUS_PER_JOB):
#   - --constraint=gpu-a100 : pins the job to A100 nodes (the "m" cluster
#     also has H100/H200 nodes in a separate 'gpu-large' partition that we
#     do not want here).
#   - --gres/--cpus-per-task/--mem are set via sbatch CLI args in
#     m_submit.sh: 16 cpus and 32G mem PER GPU (same per-GPU budget as the
#     "f" cluster), scaled by GPUS_PER_JOB. m_run_pool.sh further splits
#     --cpus-per-task evenly across its GPUS_PER_JOB workers so OMP/MKL
#     threading doesn't oversubscribe.
#   - --qos=batch-short : batch jobs up to 5 days. --time is also passed via
#     CLI (default set in m_submit.sh) — size it for
#     ceil(num_sets / GPUS_PER_JOB) waves of ~3-5h each (one set's worth of
#     seeds), not just one set, since each GPU now drains multiple sets
#     serially instead of getting exactly one via the job array.
#
# Environment: m_run_pool.sh / m_run_seedset.sh load Anaconda3 + activate the
#   conda env named by CONDA_ENV (mirrors the cluster's recommended pattern:
#   module load Anaconda3; conda activate <env>). Set CONDA_ENV before submit.
#
# Submit (preferred — sizes GPUS_PER_JOB/NUM_BIG_JOBS and snapshots the JSON
# spec so in-flight/pending tasks are immune to later edits of
# scripts/jobs.json):
#     scripts/m_submit.sh scripts/jobs.json
# or manually (NOTE: no snapshot; pass a copy of the JSON instead of the live
# path if you need to keep editing it while the job runs):
#     sbatch --gres=gpu:4 --cpus-per-task=64 --mem=128G \
#         scripts/m_submit_jobs.sbatch scripts/jobs.json 4
# ---------------------------------------------------------------------------
set -u -o pipefail

JOBS_JSON="${1:?usage: sbatch m_submit_jobs.sbatch <jobs.json> <gpus_per_job>}"
GPUS_PER_JOB="${2:?usage: sbatch m_submit_jobs.sbatch <jobs.json> <gpus_per_job>}"

# Slurm copies the batch script to a spool dir, so BASH_SOURCE does NOT point
# at the repo. Use SLURM_SUBMIT_DIR (the dir sbatch was invoked from) as the
# repo root; fall back to BASH_SOURCE for direct (non-sbatch) execution.
# SCRIPT_DIR must therefore be derived from the repo, not BASH_SOURCE, so the
# call below finds m_run_pool.sh in the real checkout (not the spool).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/scripts"
cd "${REPO_ROOT}"
mkdir -p slurm_logs 2>/dev/null || \
    echo "[m_submit_jobs] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

echo "===================================================================="
echo "ITCAS GPU pool job | job ${SLURM_JOB_ID:-NA} array ${SLURM_ARRAY_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA}"
echo "spec=${JOBS_JSON} gpus_per_job=${GPUS_PER_JOB} host=$(hostname) partition=${SLURM_JOB_PARTITION:-NA}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

bash "${SCRIPT_DIR}/m_run_pool.sh" "${JOBS_JSON}" "${GPUS_PER_JOB}"
RC=$?

echo "===================================================================="
echo "pool job finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
exit "${RC}"
