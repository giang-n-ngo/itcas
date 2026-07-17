#!/usr/bin/env bash
#SBATCH --job-name=itcas_synth_aggregate
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_synth_aggregate-%j.out
#SBATCH --error=slurm_logs/itcas_synth_aggregate-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# synthetic_comparison_aggregate.sbatch — final aggregate worker of the split
# synthetic-comparison pipeline.
#
# Submitted automatically by synthetic_comparison_dispatch.sbatch with
# --dependency=afterok on all per-problem jobs. Do not submit this script
# directly.
#
# Reads every <METRICS_DIR>/*_synthetic_metrics.json written by the
# per-problem jobs (see summarize_synthetic_comparison_problem) -- no
# .jsonl/raw run logs touched at all -- and produces, per difficulty found
# across those files, under <OUTPUT_DIR>/<difficulty>/:
#   synthetic_comparison_avg_rank_vs_evaluations.pdf
#   synthetic_comparison_relative_auc_vs_evaluations.pdf
#   synthetic_comparison_stats_report.{json,md}
#
# Required environment variables (set by the orchestrator via --export):
#   METRICS_DIR   directory containing *_synthetic_metrics.json files
#   OUTPUT_DIR    where combined per-difficulty PDFs and stats reports go
#   ALPHA         stats significance level (default: 0.05)
#   CONDA_ENV     conda environment name
#   CONDA_MODULE  module providing conda
#
# Resource sizing: NOT measured on this cluster yet for this specific code
# path (new; the old monolithic job never got this far before OOM-killing).
# Mirrors the analogous aggregate job in the other split pipeline
# (scripts/summarize_aggregate.sbatch, --mem=2G --time=00:15:00), which reads
# only small per-problem JSON summaries and is cheap. This aggregate does
# slightly more work -- Friedman/Wilcoxon stats plus 2 extra figure renders
# (avg-rank and relative-AUC) per difficulty vs. that script's 1 figure --
# so --mem=4G / --time=00:30:00 is a modest bump above that baseline, not a
# measured number. REVISE after observing actual `seff <jobid>` numbers from
# the first real aggregate run.
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
OUTPUT_DIR="${OUTPUT_DIR:-results/synthetic_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
ALPHA="${ALPHA:-0.05}"

mkdir -p "${OUTPUT_DIR}"

echo "===================================================================="
echo "ITCAS synthetic comparison (aggregate) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
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
        echo "[synth_aggregate] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND="Agg"

# --- Produce combined avg-rank/relative-AUC plots and stats reports ---------
srun --kill-on-bad-exit=1 python -m itcas.summarize \
    --synthetic-comparison-aggregate-from "${METRICS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --alpha "${ALPHA}"
RC=$?

echo "===================================================================="
echo "[synth_aggregate] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[synth_aggregate] combined outputs written under: ${OUTPUT_DIR}/<difficulty>/"
echo "===================================================================="
exit "${RC}"
