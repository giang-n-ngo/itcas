#!/usr/bin/env bash
#SBATCH --job-name=itcas_mgc_aggregate
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_mgc_aggregate-%j.out
#SBATCH --error=slurm_logs/itcas_mgc_aggregate-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# method_group_comparison_aggregate.sbatch — final aggregate worker of the
# split method-group-comparison pipeline (batch-methods-vs-each-other or
# sequential-methods-vs-each-other, selected by METHOD_GROUP).
#
# Submitted automatically by method_group_comparison_dispatch.sbatch with
# --dependency=afterok on all per-problem jobs for the same METHOD_GROUP
# (which also overrides this script's --job-name/-o/-e on the `sbatch`
# command line to bake METHOD_GROUP into the job name and log paths -- the
# #SBATCH header below can't reference ${METHOD_GROUP} itself, it is parsed
# before --export env vars are available). Do not submit this script
# directly unless you know what you are doing.
#
# Reads every <METRICS_DIR>/*_<METHOD_GROUP>_comparison_metrics.json written
# by the per-problem jobs (see
# itcas.reporting.method_group_comparison.summarize_method_group_comparison_problem)
# -- no .jsonl/raw run logs touched at all -- and produces, restricted to
# METHOD_GROUP's own 5 methods:
#   <OUTPUT_DIR>/<difficulty>/<METHOD_GROUP>_comparison_relative_auc_vs_evaluations.pdf
#   <OUTPUT_DIR>/<METHOD_GROUP>_comparison_relative_auc_by_difficulty.pdf
#   <OUTPUT_DIR>/<METHOD_GROUP>_comparison_normalized_avg_curve_vs_pct_budget.pdf
# Unlike synthetic_comparison_aggregate.sbatch, this pipeline produces NO
# avg-rank plot and NO Friedman/Wilcoxon stats report (no --alpha flag exists
# for this CLI path at all) -- see method_group_comparison.py's module
# docstring for why that's a deliberate scope cut, not an oversight.
#
# Required environment variables (set by the orchestrator via --export):
#   METHOD_GROUP   "batch" or "sequential"
#   METRICS_DIR    directory containing *_<METHOD_GROUP>_comparison_metrics.json files
#   OUTPUT_DIR     where combined PDFs go
#   CONDA_ENV      conda environment name
#   CONDA_MODULE   module providing conda
#
# Resource sizing: reuses synthetic_comparison_aggregate.sbatch's --mem=4G /
# --time=00:30:00 / --cpus-per-task=1 verbatim rather than a fresh guess.
# This aggregate reads the same kind of small per-problem JSON summaries and
# does strictly LESS work than that one -- no Friedman/Wilcoxon stats, no
# avg-rank boxplot, only 3 figure renders (relative-AUC box, relative-AUC by
# difficulty, normalized-curve) vs. that script's 2 PDFs plus a stats report
# per difficulty -- so 4G/30min stays a safe, conservative bound here too.
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

METHOD_GROUP="${METHOD_GROUP:-}"
if [[ "${METHOD_GROUP}" != "batch" && "${METHOD_GROUP}" != "sequential" ]]; then
    echo "[mgc_aggregate] ERROR: METHOD_GROUP must be exactly 'batch' or 'sequential'," \
         "got '${METHOD_GROUP}'." >&2
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${METHOD_GROUP}_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"

mkdir -p "${OUTPUT_DIR}"

echo "===================================================================="
echo "ITCAS method-group comparison (aggregate) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "method_group=${METHOD_GROUP}"
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
        echo "[mgc_aggregate] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND="Agg"

# --- Produce combined relative-AUC/normalized-curve plots for METHOD_GROUP --
srun --kill-on-bad-exit=1 python -m itcas.summarize \
    --method-group-comparison-aggregate-from "${METRICS_DIR}" \
    --method-group "${METHOD_GROUP}" \
    --output-dir "${OUTPUT_DIR}"
RC=$?

echo "===================================================================="
echo "[mgc_aggregate] finished rc=${RC} method_group=${METHOD_GROUP} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[mgc_aggregate] combined outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
