#!/usr/bin/env bash
#SBATCH --job-name=itcas_runtime_benchmark
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/itcas_runtime_benchmark-%j.out
#SBATCH --error=slurm_logs/itcas_runtime_benchmark-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
# ---------------------------------------------------------------------------
# runtime_benchmark.sbatch — MONOLITHIC single job (no array).
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
# COST — NOT YET MEASURED AT FULL SCALE. A real end-to-end smoke test on a
# contended CPU-only login node (budget=6, load avg ~46) took 30-50s/cell
# there; that is NOT representative of a real GPU compute-node run, but
# confirms this is genuine GP-fitting + acquisition-optimization compute per
# iteration, not a trivial job. At the real budget (100-200 evals/cell) x 165
# cells on an actual GPU, expect this to take a substantial amount of
# wall-clock time. For calibration, this cluster's `itcas_gpu` sweep
# (submit_jobs.sbatch, up to 20 seeds/task) had a median per-task runtime of
# ~31 min (up to ~7h at the tail; see 3403378_* in sacct) — this job runs only
# 1 seed/cell so each individual cell should be much faster than a 20-seed
# task, but 165 sequential cells in one job can still add up. --time is sized
# generously (1 day, the full batch-short QoS ceiling) as a conservative upper
# bound until a real GPU measurement exists; revise down once the full run
# has been timed once.
#
# *** VALIDATE FIRST, ON A SMALL --methods/--problems SUBSET, BEFORE THE FULL
# 11x15 SWEEP. *** This job has never been run on this cluster. Use METHODS=
# and PROBLEMS_ env overrides (see below) to submit e.g. a 2x2 cell smoke test
# and confirm it completes cleanly end-to-end on a real GPU node before
# committing to the full-scale (and expensive, previously-unmeasured) run.
#
# INCIDENT NOTE (job 3451622, 2026-07-27/28): first validation submission sat
# PENDING ~3h15m then FAILED in ~1s with ExitCode 0:53, .batch step
# CANCELLED, and NO slurm_logs/*.out|.err ever created — i.e. it died in
# Slurm's own launch path before this script's first line ran, not due to a
# bug in this script or the Python benchmark. Investigated and ruled out:
# QOS/time-limit (batch-short caps at 5-00:00:00; our 1-00:00:00 request and
# --partition=gpu/--qos=batch-short/--gres=gpu:1/--mail-user all exactly
# match a previously-COMPLETED GPU job, 3403378, in this same partition/QOS);
# node health (target node v100l-f-04 was/is MIXED, not drained/down per
# `sinfo -R`/`scontrol show node`); disk/quota (no quota set, 167G free on
# /home/giangn). The same "0:53, instant .batch CANCELLED, no logs" signature
# also appears sporadically for OTHER users' GPU jobs going back months
# (unrelated job names/accounts), and co-occurred with a burst of the same
# exit code in a concurrent, unrelated job array on this account during the
# same ~30 min window — consistent with a transient, intermittent
# cluster-side launch glitch (not a static misconfiguration in this script).
# Mitigation added: `--requeue` so Slurm auto-requeues the job if a future
# recurrence kills it via node/launch failure instead of silently failing.
# Plan: simply resubmit the same small validation (transient issue expected
# to have cleared) and confirm it runs end-to-end before the full sweep.
#
# Resource sizing:
#   - --gres=gpu:1, --cpus-per-task=8, --mem=32G: one (method,problem) cell
#     runs a single seed on a single GPU (no seed-parallelism in this script,
#     unlike run_seedset.sh's n_concurrent), so 1 GPU suffices; cpus/mem
#     mirror submit_jobs.sbatch's per-GPU allocation (scaled down on cpus
#     since there is no multi-seed concurrency here to feed).
#   - --qos=batch-short, --time=1-00:00:00: the full 5-day batch-short
#     ceiling is available if needed; 1 day is a conservative first cap given
#     the task is unmeasured at full scale — raise if the run doesn't finish
#     (or switch to --qos=batch-long, 10-day ceiling) once real per-cell
#     timings from the validation run are known.
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
#   OUTPUT=results/runtime_benchmark.md           # --output (markdown report path)
#
# Submit (full 11x15 sweep — DO NOT submit until a small validation run has
# passed; this is an expensive, previously-unmeasured job):
#   sbatch scripts/runtime_benchmark.sbatch
#
# Submit (small validation run first — RECOMMENDED before the above):
#   METHODS="random cas_eci" PROBLEMS="zdt3_6d alpine_12d" \
#   OUTPUT=results/runtime_benchmark_validation.md \
#     sbatch --export=ALL,METHODS,PROBLEMS,OUTPUT scripts/runtime_benchmark.sbatch
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
OUTPUT="${OUTPUT:-results/runtime_benchmark.md}"

echo "===================================================================="
echo "ITCAS runtime benchmark | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
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
        echo "[runtime_benchmark] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
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
echo "[runtime_benchmark] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[runtime_benchmark] markdown report (if rc=0): ${OUTPUT}"
echo "===================================================================="
exit "${RC}"
