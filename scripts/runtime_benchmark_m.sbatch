#!/usr/bin/env bash
#SBATCH --job-name=itcas_runtime_benchmark_m
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/itcas_runtime_benchmark_m-%j.out
#SBATCH --error=slurm_logs/itcas_runtime_benchmark_m-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
# ---------------------------------------------------------------------------
# runtime_benchmark_m.sbatch — MONOLITHIC single job (no array), cluster "m"
# (A100 GPUs).
#
# Sibling of runtime_benchmark.sbatch for the "m" cluster. runtime_benchmark
# .sbatch remains the entry point for the "f" cluster (L40S/V100s) and is
# unmodified. Unlike submit_jobs.sbatch/m_submit_jobs.sbatch (where "m"
# cluster convention favors ONE big multi-GPU pool job over many single-GPU
# jobs), this benchmark has no seed-parallelism to feed a multi-GPU pool with
# — it is inherently one sequential, single-GPU job (one seed per
# (method, problem) cell, cells run one after another in-process) on both
# clusters. So this sibling stays a single monolithic job just like the
# original, simply re-targeted at "m" cluster conventions below; it does NOT
# use the m_run_pool.sh pattern.
#
# Runs `python -m itcas.reporting.runtime_benchmark`, which calls
# `run_experiment(...)` in-process, ONE seed=0 run per (method, problem) cell,
# for (by default) 11 methods x 15 synthetic problems = 165 cells, at each
# problem's real resolved budget (100 or 200 evaluations, from
# configs/experiments.json), and times each cell's wall-clock with
# time.perf_counter(). It prints two markdown tables to stdout (grouped by
# budget=100 vs budget=200) and, with --output, also writes them to a file.
#
# NOTHING is persisted to results/ by the science code itself: every timed run
# writes to a tempfile.mkdtemp() dir that the script deletes via
# shutil.rmtree in a `finally` block right after each cell finishes. The ONLY
# durable artifact of this job is the printed/written markdown table (i.e.
# whatever --output points at, plus this job's own slurm_logs/*.out capture of
# stdout as a fallback).
#
# GPU REQUIRED: the script defaults --device cuda (matching scripts/jobs.json's
# cluster convention) so, unlike the CPU-only report jobs (synthetic_comparison
# .sbatch et al., which only read already-logged results), this needs a real
# GPU allocation to fit GP models / optimize acquisitions per iteration.
#
# COST — NOT YET MEASURED AT FULL SCALE. See runtime_benchmark.sbatch's header
# for the "f" cluster's CPU-login-node smoke-test numbers and per-cell cost
# reasoning (not repeated here); the same caveats apply on "m": expect the
# real 100-200-eval/cell x 165-cell run to take substantial wall-clock time on
# an actual A100, and --time is sized generously (1 day, the full
# batch-short QoS ceiling) as a conservative upper bound until a real
# measurement on this cluster exists.
#
# *** VALIDATE FIRST, ON A SMALL --methods/--problems SUBSET, BEFORE THE FULL
# 11x15 SWEEP. *** This job has never been run on the "m" cluster. Use
# METHODS= and PROBLEMS= env overrides (see below) to submit e.g. a 2x2 cell
# smoke test and confirm it completes cleanly end-to-end on a real A100 node
# before committing to the full-scale (and expensive, previously-unmeasured)
# run.
#
# (For context on a launch-path incident seen on the "f" cluster's first
# validation attempt of the sibling job — job 3451622, instant CANCELLED
# with no logs, ruled transient/cluster-side — see runtime_benchmark.sbatch's
# header. --requeue below is kept as the same cheap mitigation here.)
#
# Resource sizing:
#   - --constraint=gpu-a100: pins the job to the "m" cluster's A100 nodes.
#     The "m" cluster also has H100/H200 nodes in a separate 'gpu-large'
#     partition, which this job must NOT land on (untested there, and not
#     the intended target per the "m" cluster's A100-first convention used
#     by m_submit_jobs.sbatch).
#   - --gres=gpu:1, --cpus-per-task=8, --mem=32G: one (method,problem) cell
#     runs a single seed on a single GPU (no seed-parallelism in this script,
#     unlike run_seedset.sh/m_run_seedset.sh's n_concurrent), so 1 GPU
#     suffices; cpus/mem mirror the per-GPU allocation used elsewhere on this
#     cluster (scaled down on cpus since there is no multi-seed concurrency
#     here to feed).
#   - --qos=batch-short, --time=1-00:00:00: the full 5-day batch-short
#     ceiling is available if needed; 1 day is a conservative first cap given
#     the task is unmeasured at full scale on this cluster — raise if the run
#     doesn't finish (or switch to --qos=batch-long, 10-day ceiling) once real
#     per-cell timings from the validation run are known.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                              # conda env to activate
#   CONDA_MODULE=Anaconda3                       # module providing conda
#   METHODS=""                                   # space-separated --methods override (default: all 11)
#   PROBLEMS=""                                  # space-separated --problems override (default: all 15)
#   SEED=0                                        # --seed
#   DEVICE=cuda                                   # --device
#   EXPERIMENTS_CONFIG=configs/experiments.json   # --experiments-config
#   PROBLEMS_CONFIG=configs/final_problems.json   # --problems-config
#   OUTPUT=results/runtime_benchmark_m.md         # --output (markdown report path)
#
# Submit (full 11x15 sweep — DO NOT submit until a small validation run has
# passed; this is an expensive, previously-unmeasured job on this cluster):
#   sbatch scripts/runtime_benchmark_m.sbatch
#
# Submit (small validation run first — RECOMMENDED before the above). Note
# --gres=gpu:1 is requested directly via the #SBATCH directives above (as on
# the "f" cluster); this is still the correct way to request a single GPU on
# "m" for a plain (non-pool) sbatch submission like this one — the CLI-passed
# --gres pattern in m_submit.sh/m_submit_jobs.sbatch only applies to the
# multi-GPU pool jobs, which this script is not:
#   METHODS="random cas_eci" PROBLEMS="zdt3_6d alpine_12d" \
#   OUTPUT=results/runtime_benchmark_m_validation.md \
#     sbatch --export=ALL,METHODS,PROBLEMS,OUTPUT scripts/runtime_benchmark_m.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs 2>/dev/null || true

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
METHODS="${METHODS:-}"
PROBLEMS="${PROBLEMS:-}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG:-configs/experiments.json}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
OUTPUT="${OUTPUT:-results/runtime_benchmark_m.md}"

echo "===================================================================="
echo "ITCAS runtime benchmark (m) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "methods=${METHODS:-<all 11 default>} problems=${PROBLEMS:-<all 15 default>}"
echo "seed=${SEED} device=${DEVICE} experiments_config=${EXPERIMENTS_CONFIG} problems_config=${PROBLEMS_CONFIG}"
echo "output=${OUTPUT}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[runtime_benchmark_m] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Build args ---------------------------------------------------------
ARGS=(--seed "${SEED}" --device "${DEVICE}" \
      --experiments-config "${EXPERIMENTS_CONFIG}" \
      --problems-config "${PROBLEMS_CONFIG}" \
      --output "${OUTPUT}")
if [[ -n "${METHODS}" ]]; then
    # shellcheck disable=SC2206
    METHODS_ARR=(${METHODS})
    ARGS+=(--methods "${METHODS_ARR[@]}")
fi
if [[ -n "${PROBLEMS}" ]]; then
    # shellcheck disable=SC2206
    PROBLEMS_ARR=(${PROBLEMS})
    ARGS+=(--problems "${PROBLEMS_ARR[@]}")
fi

mkdir -p "$(dirname "${OUTPUT}")" 2>/dev/null || true

srun --kill-on-bad-exit=1 python -m itcas.reporting.runtime_benchmark "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[runtime_benchmark_m] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[runtime_benchmark_m] markdown report (if rc=0): ${OUTPUT}"
echo "===================================================================="
exit "${RC}"
