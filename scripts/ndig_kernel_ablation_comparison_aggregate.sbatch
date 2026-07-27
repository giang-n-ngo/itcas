#!/usr/bin/env bash
#SBATCH --job-name=itcas_nka_aggregate
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_nka_aggregate-%j.out
#SBATCH --error=slurm_logs/itcas_nka_aggregate-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# ndig_kernel_ablation_comparison_aggregate.sbatch — final aggregate worker
# of the split NDIG QD-DPP diversity-kernel-ablation comparison pipeline
# (proposed method's full batch NDIG acquisition, itcas_ndig, vs. its two
# kernel-ablated siblings ndig_no_kobj_batch/ndig_no_kctx_batch).
#
# Submitted automatically by ndig_kernel_ablation_comparison_dispatch.sbatch
# with --dependency=afterok on all per-problem jobs. Do not submit this
# script directly unless you know what you are doing.
#
# Reads every <METRICS_DIR>/*_ndig_kernel_ablation_metrics.json written by the
# per-problem jobs (see
# itcas.reporting.ndig_kernel_ablation_comparison.summarize_ndig_kernel_ablation_problem)
# -- no .jsonl/raw run logs touched at all -- and produces TWO variants of
# each figure: the base 3-method comparison (itcas_ndig, ndig_no_kobj_batch,
# ndig_no_kctx_batch) and a "_with_seq" variant that additionally includes
# itcas_seq_ndig (the forced-sequential sibling, no QD-DPP batch diversity at
# all) -- see ndig_kernel_ablation_comparison.py's module docstring's "Two
# output pairs" section:
#   <OUTPUT_DIR>/ndig_kernel_ablation_relative_auc_by_difficulty.pdf
#   <OUTPUT_DIR>/ndig_kernel_ablation_normalized_avg_curve_vs_pct_budget.pdf
#   <OUTPUT_DIR>/ndig_kernel_ablation_relative_auc_by_difficulty_with_seq.pdf
#   <OUTPUT_DIR>/ndig_kernel_ablation_normalized_avg_curve_vs_pct_budget_with_seq.pdf
# This pipeline produces NO relative-AUC-vs-evaluations boxplot, NO avg-rank
# plot, and NO Friedman/Wilcoxon stats report (no --alpha flag exists for
# this CLI path at all) -- see ndig_kernel_ablation_comparison.py's module
# docstring for why that's a deliberate scope cut, not an oversight.
#
# Required environment variables (set by the orchestrator via --export):
#   METRICS_DIR    directory containing *_ndig_kernel_ablation_metrics.json files
#   OUTPUT_DIR     where the combined PDFs go
#   CONDA_ENV      conda environment name
#   CONDA_MODULE   module providing conda
#
# Resource sizing: reuses method_group_comparison_aggregate.sbatch's --mem=4G /
# --time=00:30:00 / --cpus-per-task=1 verbatim rather than a fresh guess. This
# aggregate reads the same kind of small per-problem JSON summaries and does
# comparable work to that one -- 4 figure renders (relative-AUC by
# difficulty + normalized-curve, each in 2 variants) vs. that script's 3 --
# so 4G/30min stays a safe, conservative bound here too. REVISE after observing actual
# `seff <jobid>` numbers from the first real aggregate run of THIS pipeline
# on this cluster.
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
OUTPUT_DIR="${OUTPUT_DIR:-results/ndig_kernel_ablation_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"

mkdir -p "${OUTPUT_DIR}"

echo "===================================================================="
echo "ITCAS NDIG kernel-ablation comparison (aggregate) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "metrics_dir=${METRICS_DIR} output_dir=${OUTPUT_DIR}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[nka_aggregate] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND="Agg"

# --- Produce combined relative-AUC/normalized-curve plots -------------------
srun --kill-on-bad-exit=1 python -m itcas.summarize \
    --ndig-kernel-ablation-aggregate-from "${METRICS_DIR}" \
    --output-dir "${OUTPUT_DIR}"
RC=$?

echo "===================================================================="
echo "[nka_aggregate] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[nka_aggregate] combined outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
