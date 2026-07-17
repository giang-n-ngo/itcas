#!/usr/bin/env bash
#SBATCH --job-name=itcas_school_comparison
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_school_comparison-%j.out
#SBATCH --error=slurm_logs/itcas_school_comparison-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# school_comparison.sbatch — single lightweight CPU job (no array, no
# dependency), structurally like summarize_aggregate.sbatch.
#
# Runs itcas/reporting/school_comparison.py, which internally reads run logs
# for all 16 problems in configs/final_problems.json at one fixed difficulty,
# across both LSE bases (straddle_then_sample and bes_then_sample) and their
# associated methods, sequential-only (batch generation was removed), and
# produces exactly 3 PDFs:
#   two_schools_sequential_lse10_vs_evaluations.pdf
#   two_schools_sequential_lse25_vs_evaluations.pdf
#   two_schools_sequential_lse50_vs_evaluations.pdf
#
# Measured locally on real results/sweep data (16 problems x 2 LSE bases):
# 6m35s wall-clock, 7m56s user CPU (~124% CPU, modest multi-threaded BLAS use
# via torch/numpy), ~16.0GB peak RSS (_collect_setting precomputes metric
# curves for every problem and the union of every LSE-proportion's methods
# before any PDF is written, so peak RSS scales with problems x methods x
# seeds). CPU-only, no GPU required.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                             # conda env to activate
#   CONDA_MODULE=Anaconda3                      # module providing conda
#   INPUT_DIR=results/sweep                     # run-log tree to read
#   CONFIG=configs/two_schools_of_thought.json  # method/problem/difficulty config
#   OUTPUT_DIR=<unset>                          # where PDFs are written;
#                                                # if unset, the Python module's
#                                                # own default (results/school_comparison)
#                                                # is used, NOT INPUT_DIR
#
# Submit:
#   sbatch scripts/school_comparison.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep CONFIG=configs/two_schools_of_thought.json \
#   OUTPUT_DIR=results/sweep/_school_comparison \
#     sbatch --export=ALL,INPUT_DIR,CONFIG,OUTPUT_DIR scripts/school_comparison.sbatch
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
CONFIG="${CONFIG:-configs/two_schools_of_thought.json}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "===================================================================="
echo "ITCAS school comparison | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} config=${CONFIG} output_dir=${OUTPUT_DIR:-<script default: results/school_comparison>}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[school_comparison] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Produce the two-schools-of-thought comparison PDFs ---------------------
ARGS=(--input-dir "${INPUT_DIR}" --config "${CONFIG}")
if [[ -n "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

srun --kill-on-bad-exit=1 python -m itcas.reporting.school_comparison "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[school_comparison] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[school_comparison] outputs written under: ${OUTPUT_DIR:-results/school_comparison}"
echo "===================================================================="
exit "${RC}"
