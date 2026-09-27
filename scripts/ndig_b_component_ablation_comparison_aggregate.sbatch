#!/usr/bin/env bash
#SBATCH --job-name=itcas_ncab_aggregate
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_ncab_aggregate-%j.out
#SBATCH --error=slurm_logs/itcas_ncab_aggregate-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# ndig_b_component_ablation_comparison_aggregate.sbatch — final aggregate
# worker of the split NDIG-B acquisition-component ablation comparison
# pipeline (proposed method's full batch NDIG acquisition, itcas_ndig, vs.
# its four quality-component ablations itcas_ndig_no_infogain/itcas_edig/
# itcas_efig/itcas_ndig_pof_entropy).
#
# Submitted automatically by ndig_b_component_ablation_comparison_dispatch.sbatch
# with --dependency=afterok on all per-problem jobs. Do not submit this
# script directly unless you know what you are doing.
#
# Reads every <METRICS_DIR>/*_ndig_b_component_ablation_metrics.json written
# by the per-problem jobs (see
# itcas.reporting.ndig_b_component_ablation_comparison.summarize_ndig_b_component_ablation_problem)
# -- no .jsonl/raw run logs touched at all -- and produces the single pair of
# figures documented in that module's docstring (no "_with_seq" variant,
# unlike the QD-DPP kernel-ablation pipeline -- every method compared is
# already a batch run):
#   <OUTPUT_DIR>/ndig_b_component_ablation_relative_auc_by_difficulty.pdf
#   <OUTPUT_DIR>/ndig_b_component_ablation_normalized_avg_curve_vs_pct_budget.pdf
# This pipeline produces NO relative-AUC-vs-evaluations boxplot, NO avg-rank
# plot, and NO Friedman/Wilcoxon stats report (no --alpha flag exists for
# this CLI path at all) -- see ndig_b_component_ablation_comparison.py's
# module docstring for why that's a deliberate scope cut, not an oversight.
#
# Required environment variables (set by the orchestrator via --export):
#   METRICS_DIR    directory containing *_ndig_b_component_ablation_metrics.json files
#   OUTPUT_DIR     where the combined PDFs go
#   CONDA_ENV      conda environment name
#   CONDA_MODULE   module providing conda
#
# Resource sizing: reuses ndig_kernel_ablation_comparison_aggregate.sbatch's
# --mem=4G / --time=00:30:00 / --cpus-per-task=1 verbatim rather than a fresh
# guess. This aggregate reads the same kind of small per-problem JSON
# summaries and renders only 2 figures (vs. that script's 4, since there is
# no "_with_seq" variant here), so 4G/30min stays a safe, conservative bound.
# REVISE after observing actual `seff <jobid>` numbers from the first real
# aggregate run of THIS pipeline on this cluster.
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
OUTPUT_DIR="${OUTPUT_DIR:-results/ndig_b_component_ablation_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"

mkdir -p "${OUTPUT_DIR}"

echo "===================================================================="
echo "ITCAS NDIG-B component-ablation comparison (aggregate) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
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
        echo "[ncab_aggregate] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
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
    --ndig-b-component-ablation-aggregate-from "${METRICS_DIR}" \
    --output-dir "${OUTPUT_DIR}"
RC=$?

echo "===================================================================="
echo "[ncab_aggregate] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ncab_aggregate] combined outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
