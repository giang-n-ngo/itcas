#!/usr/bin/env bash
#SBATCH --job-name=itcas_school_comparison
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
# for 5 fixed problems (all_valley_8d, alpine_12d, dtlz1_12d, dtlz2_6d,
# dtlz3_8d) at one fixed difficulty, for a small fixed set of methods
# (straddle_then_sample, bes_then_sample, cas_eci/cas_eci_batch,
# moc_cas_hard/moc_cas_hard_batch) and produces exactly 4 PDFs:
#   two_schools_sequential_vs_evaluations.pdf
#   two_schools_sequential_vs_steps.pdf
#   two_schools_batch_vs_evaluations.pdf
#   two_schools_batch_vs_steps.pdf
#
# Measured locally on real results/sweep data: 2m18s wall-clock, 4m48s user
# CPU (modest multi-threaded BLAS use via torch/numpy), ~2.5GB peak RSS.
# CPU-only, no GPU required.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                             # conda env to activate
#   CONDA_MODULE=Anaconda3                      # module providing conda
#   INPUT_DIR=results/sweep                     # run-log tree to read
#   CONFIG=configs/two_schools_of_thought.json  # method/problem/difficulty config
#   OUTPUT_DIR=<unset>                          # where PDFs are written;
#                                                # if unset, writes into INPUT_DIR
#                                                # (matches the script's own default)
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
echo "input_dir=${INPUT_DIR} config=${CONFIG} output_dir=${OUTPUT_DIR:-<input_dir>}"
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
echo "[school_comparison] outputs written under: ${OUTPUT_DIR:-${INPUT_DIR}}"
echo "===================================================================="
exit "${RC}"
