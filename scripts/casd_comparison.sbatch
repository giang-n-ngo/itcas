#!/usr/bin/env bash
#SBATCH --job-name=itcas_casd_comparison
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/itcas_casd_comparison-%j.out
#SBATCH --error=slurm_logs/itcas_casd_comparison-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# casd_comparison.sbatch — single lightweight CPU job (no array, no
# dependency), structurally identical to ff_comparison.sbatch (its FF
# counterpart -- casd_llm is the repo's second real-world problem, see
# itcas/reporting/batch_vs_sequential.py's _REAL_WORLD_PROBLEMS).
#
# Runs itcas/reporting/casd_comparison.py, which reads run logs under
# --input-dir for the "CASD problem" (casd_llm -- the LLM decoding
# hyperparameter search problem, the only other problem besides FF with its
# own per-problem difficulty scale, here p1_00..p4_00) and compares 6
# hardcoded methods: itcas_ndig (the proposed method, batch) vs. five
# forced-sequential baselines (bes_then_sample_lse10,
# straddle_then_sample_lse10, random, cas_eci, moc_cas_hard). Like
# ff_comparison.py (unlike batch_vs_sequential.py / summarize.py), this
# module's CLI takes no --methods-config/--problems-config/--families flags
# -- the method list and the single CASD problem are hardcoded in the module
# itself. It renders ONE grid PDF (rows = the 4 difficulty levels + 1
# average-rank summary row, columns = metric curves + product + product-rank)
# keyed on total individual evaluations, runs a Friedman-gated
# Holm-Bonferroni-corrected one-sided paired Wilcoxon dominance test (H1:
# itcas_ndig > each baseline) per difficulty level, and writes a stats report
# (json+md) -- all into --output-dir.
#
# CPU-only (matplotlib PDF rendering + Wilcoxon/Friedman tests, no GPU, no
# torch model fitting, no CASD server involved -- just reads already-logged
# run curves).
#
# Resource sizing: NOT measured locally (no cluster access at authoring
# time) -- copied verbatim from ff_comparison.sbatch's estimate, which itself
# was ratio'd down from batch_vs_sequential.sbatch's measured baseline. This
# job reads exactly 1 problem x 6 methods x 4 difficulty levels x ~20 seeds --
# fewer difficulty levels than FF's 10, so if anything this is an
# upper-bound estimate; --time=00:15:00 / --mem=4G / --cpus-per-task=4 are
# kept identical to ff_comparison.sbatch rather than scaled down further,
# since the fixed per-level Friedman + 5x Holm-Bonferroni-corrected Wilcoxon
# battery overhead dominates at this data volume either way. Revise once
# actual run time is observed on this cluster.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                          # conda env to activate
#   CONDA_MODULE=Anaconda3                   # module providing conda
#   INPUT_DIR=results/sweep                  # run-log tree to read
#   OUTPUT_DIR=results/casd_comparison       # where the PDF/stats report are written
#   ALPHA=0.05                               # Friedman gate / Wilcoxon significance level
#
# Submit:
#   sbatch scripts/casd_comparison.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep OUTPUT_DIR=results/casd_comparison/rerun ALPHA=0.10 \
#     sbatch --export=ALL,INPUT_DIR,OUTPUT_DIR,ALPHA scripts/casd_comparison.sbatch
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
OUTPUT_DIR="${OUTPUT_DIR:-results/casd_comparison}"
ALPHA="${ALPHA:-0.05}"

echo "===================================================================="
echo "ITCAS CASD comparison | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} output_dir=${OUTPUT_DIR} alpha=${ALPHA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[casd_comparison] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Produce the CASD comparison PDF + stats report --------------------------
mkdir -p "${OUTPUT_DIR}"
ARGS=(--input-dir "${INPUT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --alpha "${ALPHA}")

srun --kill-on-bad-exit=1 python -m itcas.reporting.casd_comparison "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[casd_comparison] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[casd_comparison] outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
