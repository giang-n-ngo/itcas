#!/usr/bin/env bash
#SBATCH --job-name=itcas_synth_problem
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/itcas_synth_problem-%j.out
#SBATCH --error=slurm_logs/itcas_synth_problem-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=FAIL
# ---------------------------------------------------------------------------
# synthetic_comparison_problem.sbatch — per-problem worker of the split
# synthetic-comparison pipeline.
#
# Submitted automatically by synthetic_comparison_dispatch.sbatch (the
# orchestrator). Do not submit this script directly unless you know what you
# are doing.
#
# For one problem (PROBLEM) this job runs
# `python -m itcas.summarize --synthetic-comparison-problem "${PROBLEM}"`
# (see itcas.reporting.summary.summarize_synthetic_comparison_problem), which:
#   1. Loads only PROBLEM's own runs (6 methods x 4 difficulties x ~20 seeds)
#      -- never any other problem's .jsonl logs.
#   2. Renders <OUTPUT_DIR>/<difficulty>/synthetic_comparison_<PROBLEM>_vs_evaluations.pdf
#      for each difficulty level found.
#   3. Writes <METRICS_DIR>/<PROBLEM>_synthetic_metrics.json (a small
#      intermediate summary, NOT raw run data) for the aggregate job.
#
# Required environment variables (set by the orchestrator via --export):
#   PROBLEM           synthetic problem name (e.g. zakharov_ackley_10d)
#   INPUT_DIR         sweep root containing run logs
#   PROBLEMS_CONFIG   final-problems config, used only to validate PROBLEM is
#                      a synthetic problem (not spacecraft_formation_flying_a1)
#   OUTPUT_DIR        where per-difficulty PDFs are written
#   METRICS_DIR       where the intermediate metrics JSON is written
#   ALPHA             stats significance level (default: 0.05; unused by this
#                      half of the pipeline, accepted for CLI/signature
#                      symmetry with the aggregate job -- see
#                      summarize_synthetic_comparison_problem's docstring)
#   CONDA_ENV         conda environment name
#   CONDA_MODULE      module providing conda
#
# Resource sizing: NOT measured via `seff` on THIS cluster's Slurm scheduler
# yet for this specific code path -- job 3427762 (the old monolithic
# --synthetic-comparison job) OOM killed at 15.99/16G before revealing any
# per-problem peak-RSS number, and this per-problem entry point is brand new.
# On-disk, one synthetic problem's raw sweep directory is 80-430MB
# (`du -sh results/sweep/*/`, e.g. styblinski_tang_6d ~248M up to
# ellipsoid_rastrigin_20d ~429M, the largest of the 15 synthetic problems);
# the O(T^2) per-step RunSeries reconstruction inflates that by some multiple
# in memory, but critically this job only ever holds ONE problem's data at a
# time (unlike the monolithic job's ~15 simultaneously). The analogous
# single-problem worker in the other split pipeline
# (scripts/summarize_problem.sbatch, which reconstructs the same kind of
# RunSeries data for one problem across all its discovered methods) runs fine
# at --mem=4G. This job covers comparably more data per problem (6 methods x
# 4 difficulties x ~20 seeds). A local (non-Slurm, off-cluster) smoke test of
# `--synthetic-comparison-problem` via `/usr/bin/time -v` measured peak RSS of
# ~1.4GB for styblinski_tang_6d (248M on disk) and ~2.9GB for
# ellipsoid_rastrigin_20d (429M on disk, the largest problem) -- both well
# under 10G, but this was measured off-cluster and is not a substitute for a
# real `seff <jobid>` reading. --mem=10G is chosen as generous headroom above
# both that local measurement and the --mem=4G baseline, rather than an exact
# match. REVISE this number after observing actual `seff <jobid>` numbers
# from the first real per-problem Slurm run on this cluster (the
# ellipsoid_rastrigin_20d job is the one to check for the true worst case).
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
INPUT_DIR="${INPUT_DIR:-results/sweep}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/synthetic_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
ALPHA="${ALPHA:-0.05}"
PROBLEM="${PROBLEM:?PROBLEM env var is required}"

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "===================================================================="
echo "ITCAS synthetic comparison (per-problem) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "problem=${PROBLEM} input_dir=${INPUT_DIR} problems_config=${PROBLEMS_CONFIG}"
echo "output_dir=${OUTPUT_DIR} metrics_dir=${METRICS_DIR} alpha=${ALPHA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[synth_problem] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Render this problem's PDF(s) + save metrics -----------------------------
srun --kill-on-bad-exit=1 python -m itcas.summarize \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --problems-config "${PROBLEMS_CONFIG}" \
    --alpha "${ALPHA}" \
    --synthetic-comparison-problem "${PROBLEM}" \
    --save-metrics "${METRICS_DIR}"
RC=$?

echo "===================================================================="
echo "[synth_problem] finished rc=${RC} problem=${PROBLEM} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[synth_problem] outputs written under: ${OUTPUT_DIR}/<difficulty>/"
echo "[synth_problem] metrics written to: ${METRICS_DIR}/${PROBLEM}_synthetic_metrics.json"
echo "===================================================================="
exit "${RC}"
