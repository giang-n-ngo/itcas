#!/usr/bin/env bash
#SBATCH --job-name=itcas_summary_aggregate
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/itcas_summary_aggregate-%j.out
#SBATCH --error=slurm_logs/itcas_summary_aggregate-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# summarize_aggregate.sbatch — final aggregate job.
#
# Submitted automatically by summarize.sbatch with --dependency=afterok on
# all per-problem jobs.  Do not submit this script directly.
#
# Reads every <METRICS_DIR>/*_metrics.json written by the per-problem jobs
# and produces:
#   OUTPUT_DIR/all_problems_hypervolume_vs_evaluations.pdf
#   OUTPUT_DIR/all_problems_hypervolume_vs_steps.pdf
#   OUTPUT_DIR/all_problems_<variant>_stats_report.{json,md}
#
# No run logs are re-read; all computation is driven from the small JSON
# files saved by the per-problem jobs.
#
# Required environment variables (set by the orchestrator via --export):
#   METRICS_DIR   directory containing *_metrics.json files
#   OUTPUT_DIR    where combined PDFs and stats reports are written
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
OUTPUT_DIR="${OUTPUT_DIR:-results/sweep/_summary}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
ALPHA="${ALPHA:-0.05}"

mkdir -p "${OUTPUT_DIR}"

echo "===================================================================="
echo "ITCAS summary (aggregate) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "metrics_dir=${METRICS_DIR} output_dir=${OUTPUT_DIR} alpha=${ALPHA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[aggregate] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND="Agg"

# --- Produce combined plots and global stats --------------------------------
srun --kill-on-bad-exit=1 python -m itcas.summarize \
    --aggregate-from "${METRICS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --alpha "${ALPHA}"
RC=$?

echo "===================================================================="
echo "[aggregate] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[aggregate] combined outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
