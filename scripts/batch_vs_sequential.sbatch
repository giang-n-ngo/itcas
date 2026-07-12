#!/usr/bin/env bash
#SBATCH --job-name=itcas_batch_vs_sequential
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/itcas_batch_vs_sequential-%j.out
#SBATCH --error=slurm_logs/itcas_batch_vs_sequential-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# batch_vs_sequential.sbatch — single lightweight CPU job (no array, no
# dependency), structurally like school_comparison.sbatch / summarize.sbatch.
#
# Runs itcas/reporting/batch_vs_sequential.py, which reads run logs under
# --input-dir, pairs each CAS-family method (eci, moc_cas_hard, straddle by
# default, from the "families" mapping in --methods-config) against its own
# _batch sibling, renders grid PDFs (rows = problems/difficulty levels,
# columns = metric curves + product + product rank) keyed on total
# evaluations, runs one-sided paired Wilcoxon signed-rank tests (H1: batch >
# sequential, Holm-Bonferroni corrected) per family, and writes a SUMMARY.md
# synthesising per-problem / per-difficulty tallies — all into --output-dir.
#
# CPU-only (matplotlib PDF rendering + Wilcoxon tests, no GPU, no torch model
# fitting beyond reading logged curves).
#
# Resource sizing: measured locally against the real results/sweep data (16
# problems, 3 families, 20 seeds each) at ~15 min wall-clock, CPU-only. This
# covers ~3x as many problems and slightly more families than
# school_comparison.sbatch's 5-problem run (2m18s measured -> 30 min / 8G
# allocated there, a ~13x time / ~3x mem headroom). Applying a comparable but
# somewhat lighter headroom multiplier to this job's larger 15 min baseline
# (~4x time, ~3x mem, scaled for the larger data volume) gives --time=01:00:00
# and --mem=16G; --cpus-per-task=4 is kept the same as school_comparison.sbatch
# since PDF rendering/Wilcoxon tests aren't heavily multi-threaded. Adjust if
# profiling on this cluster shows otherwise.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                              # conda env to activate
#   CONDA_MODULE=Anaconda3                       # module providing conda
#   INPUT_DIR=results/sweep                      # run-log tree to read
#   METHODS_CONFIG=configs/final_methods.json    # families mapping config
#   PROBLEMS_CONFIG=configs/final_problems.json  # problem suite config
#   FAMILIES=eci,moc_cas_hard,straddle           # comma-separated family subset
#   OUTPUT_DIR=results/batch_vs_sequential        # where PDFs/SUMMARY.md are written
#   ALPHA=0.05                                    # Wilcoxon significance level
#
# Submit:
#   sbatch scripts/batch_vs_sequential.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep FAMILIES=eci,moc_cas_hard \
#   OUTPUT_DIR=results/batch_vs_sequential/cas_only \
#     sbatch --export=ALL,INPUT_DIR,FAMILIES,OUTPUT_DIR scripts/batch_vs_sequential.sbatch
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
METHODS_CONFIG="${METHODS_CONFIG:-configs/final_methods.json}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
FAMILIES="${FAMILIES:-eci,moc_cas_hard,straddle}"
OUTPUT_DIR="${OUTPUT_DIR:-results/batch_vs_sequential}"
ALPHA="${ALPHA:-0.05}"

echo "===================================================================="
echo "ITCAS batch vs. sequential | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} methods_config=${METHODS_CONFIG} problems_config=${PROBLEMS_CONFIG}"
echo "families=${FAMILIES} output_dir=${OUTPUT_DIR} alpha=${ALPHA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[batch_vs_sequential] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Produce the batch-vs-sequential comparison PDFs + SUMMARY.md ----------
mkdir -p "${OUTPUT_DIR}"
ARGS=(--input-dir "${INPUT_DIR}" \
      --methods-config "${METHODS_CONFIG}" \
      --problems-config "${PROBLEMS_CONFIG}" \
      --families "${FAMILIES}" \
      --output-dir "${OUTPUT_DIR}" \
      --alpha "${ALPHA}")

srun --kill-on-bad-exit=1 python -m itcas.reporting.batch_vs_sequential "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[batch_vs_sequential] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[batch_vs_sequential] outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
