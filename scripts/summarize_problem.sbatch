#!/usr/bin/env bash
#SBATCH --job-name=itcas_summary_problem
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/itcas_summary_problem-%j.out
#SBATCH --error=slurm_logs/itcas_summary_problem-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=FAIL
# ---------------------------------------------------------------------------
# summarize_problem.sbatch — per-problem summary job.
#
# Submitted automatically by summarize.sbatch (the orchestrator).  Do not
# submit this script directly unless you know what you are doing.
#
# For one problem (BENCHMARK) this job:
#   1. Renders <OUTPUT_DIR>/<BENCHMARK>_summary_vs_{evaluations,steps}.pdf
#   2. Writes per-variant stats reports under OUTPUT_DIR
#   3. Saves <METRICS_DIR>/<BENCHMARK>_metrics.json for the aggregate job
#
# Required environment variables (set by the orchestrator via --export):
#   BENCHMARK     problem name (e.g. sphere2_6d)
#   INPUT_DIR     sweep root containing run logs
#   OUTPUT_DIR    where PDFs and stats reports are written
#   METRICS_DIR   where the intermediate metrics JSON is written
#   ALPHA         stats significance level (default: 0.05)
#   CONDA_ENV     conda environment name
#   CONDA_MODULE  module providing conda
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
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/_summary}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
ALPHA="${ALPHA:-0.05}"
BENCHMARK="${BENCHMARK:?BENCHMARK env var is required}"

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "===================================================================="
echo "ITCAS summary (per-problem) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "benchmark=${BENCHMARK} input_dir=${INPUT_DIR} output_dir=${OUTPUT_DIR}"
echo "metrics_dir=${METRICS_DIR} alpha=${ALPHA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[problem] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Render per-problem summary + save metrics ------------------------------
srun --kill-on-bad-exit=1 python -m itcas.summarize \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --benchmark "${BENCHMARK}" \
    --alpha "${ALPHA}" \
    --save-metrics "${METRICS_DIR}"
RC=$?

echo "===================================================================="
echo "[problem] finished rc=${RC} benchmark=${BENCHMARK} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[problem] outputs written under: ${OUTPUT_DIR}"
echo "[problem] metrics written to: ${METRICS_DIR}/${BENCHMARK}_metrics.json"
echo "===================================================================="
exit "${RC}"
