#!/usr/bin/env bash
#SBATCH --job-name=itcas_ndig_comparison
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_ndig_comparison-%j.out
#SBATCH --error=slurm_logs/itcas_ndig_comparison-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# ndig_comparison.sbatch — single lightweight CPU job (no array, no
# dependency), structurally identical to batch_improvement_comparison.sbatch.
#
# Runs itcas/reporting/ndig_comparison.py, which has been rewritten as a
# narrow, single-purpose report (the old 3-method, real-world-plus-spacecraft,
# Friedman/Wilcoxon-stats version is gone -- see the module's own docstring).
# It now compares exactly 2 methods (itcas_ndig "NDIG-B" vs itcas_seq_ndig
# "NDIG") across the synthetic problem suite in --problems-config (the two
# real-world problems are excluded, same as batch_improvement_comparison.py),
# but discovers runs for 7 methods total -- those 2 plus the 5 other methods
# in summary.py's SYNTHETIC_METHODS pool (random,
# straddle_then_sample_lse10, bes_then_sample_lse10, cas_eci, moc_cas_hard) --
# since the relative-AUC ratio's denominator is "best AUC among every method
# shown in the synthetic comparison", not just the 2 plotted methods. It
# renders a SINGLE PDF: a 2x2 grid, one boxplot panel per metric (no product
# panel), each with one box per plotted method. No --alpha flag -- this
# report, like batch_improvement_comparison.py, is purely descriptive with no
# Friedman/Wilcoxon significance testing.
#
# CPU-only (matplotlib PDF rendering + numpy aggregation, no GPU, no torch
# model fitting -- just reads already-logged run curves via the same
# disk-backed curve cache ff_comparison.py/casd_comparison.py/
# batch_improvement_comparison.py use).
#
# Resource sizing: no sacct history exists yet for this rewritten, narrower
# report, so this borrows batch_improvement_comparison.sbatch's own
# empirically-derived sizing as the closest comparable job rather than
# guessing from scratch: that job reads the same full synthetic-suite tree
# (14 problems) with a similar method count (10) and no stats overhead, and
# settled on --mem=20G/--time=00:30:00 against a consistently observed
# ~16-16.3 GiB peak (dominated by the initial --input-dir tree scan, not by
# which methods/problems are used downstream -- see that script's own sizing
# comment for the full sacct history it was derived from). This report reads
# 7 methods (fewer than that job's 10) over the same problem tree and renders
# only one figure instead of a 5-family heatmap, so its true peak should sit
# at or below that ~16-16.3 GiB floor; --mem=20G keeps the same ~25% headroom
# rather than trimming further on an unmeasured guess. Revise once an actual
# run completes on this cluster (check `seff <jobid>`).
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                              # conda env to activate
#   CONDA_MODULE=Anaconda3                       # module providing conda
#   INPUT_DIR=results/sweep                      # run-log tree to read
#   PROBLEMS_CONFIG=configs/final_problems.json  # synthetic problem suite config
#   OUTPUT_DIR=results/ndig_comparison           # where the relative-AUC PDF is written
#   AUC_CACHE_DIR=                               # disk-backed curve cache dir;
#                                                 # empty/unset -> module default
#                                                 # (<input-dir's parent>/auc_cache)
#
# Submit:
#   sbatch scripts/ndig_comparison.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep OUTPUT_DIR=results/ndig_comparison/rerun \
#     sbatch --export=ALL,INPUT_DIR,OUTPUT_DIR scripts/ndig_comparison.sbatch
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
OUTPUT_DIR="${OUTPUT_DIR:-results/ndig_comparison}"
AUC_CACHE_DIR="${AUC_CACHE_DIR:-}"

echo "===================================================================="
echo "ITCAS NDIG comparison | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} problems_config=${PROBLEMS_CONFIG} output_dir=${OUTPUT_DIR}"
echo "auc_cache_dir=${AUC_CACHE_DIR:-<default>}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[ndig_comparison] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Produce the relative-AUC-vs-synthetic-benchmarks PDF -------------------
mkdir -p "${OUTPUT_DIR}"
ARGS=(--input-dir "${INPUT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --problems-config "${PROBLEMS_CONFIG}")
if [[ -n "${AUC_CACHE_DIR}" ]]; then
    ARGS+=(--auc-cache-dir "${AUC_CACHE_DIR}")
fi

srun --kill-on-bad-exit=1 python -m itcas.reporting.ndig_comparison "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[ndig_comparison] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ndig_comparison] outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
