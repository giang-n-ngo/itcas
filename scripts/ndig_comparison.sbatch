#!/usr/bin/env bash
#SBATCH --job-name=itcas_ndig_comparison
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/itcas_ndig_comparison-%j.out
#SBATCH --error=slurm_logs/itcas_ndig_comparison-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# ndig_comparison.sbatch — single lightweight CPU job (no array, no
# dependency), structurally like ff_comparison.sbatch / batch_vs_sequential.sbatch.
#
# Runs itcas/reporting/ndig_comparison.py, which reads run logs under
# --input-dir for the problem suite listed in --problems-config and compares
# 3 hardcoded methods: itcas_ndig (the proposed method, batch) vs. its two
# forced-sequential siblings itcas_seq_ndig and cr_ndig. Unlike
# batch_vs_sequential.py, this module's CLI takes no --methods-config/
# --families flags -- the method list is hardcoded in the module itself. It
# renders one grid PDF per shared "standard" difficulty (rows = problems + 1
# average-rank summary row) plus one grid PDF for
# spacecraft_formation_flying_a1 (rows = its own 10 difficulty levels + 1
# average-rank summary row), columns = metric curves + product (no trailing
# product-rank column), keyed on total individual evaluations, runs a
# Friedman-gated Holm-Bonferroni-corrected one-sided paired Wilcoxon
# dominance test (H1: itcas_ndig > baseline) per (problem, difficulty)
# against each of the two baselines, and writes a stats report (json+md) --
# all into --output-dir.
#
# CPU-only (matplotlib PDF rendering + Wilcoxon/Friedman tests, no GPU, no
# torch model fitting -- just reads already-logged run curves).
#
# Resource sizing: the initial estimate here (--mem=8G, by analogy with
# ff_comparison.sbatch/batch_vs_sequential.sbatch) OOM-killed on the first
# real submission (job 3407326, killed within ~2 min, before any progress
# output) against the full results/sweep tree (~43k run-log jsonl files,
# ~1GB raw, but Python/pandas parsing of that many individual runs inflates
# well past 8G). batch_vs_sequential.sbatch's own measured-safe baseline for
# a comparably-sized read (16 problems x 3 families x 20 seeds) is 16G,
# and this job additionally holds the spacecraft grid's 10 difficulty levels
# in memory on top of the standard-problem grids, so --mem=32G gives a 2x
# margin above that known-safe figure rather than another unverified guess.
# --time=00:45:00 is bumped similarly for headroom; --cpus-per-task=4 is
# kept the same as its siblings since PDF rendering/Wilcoxon tests aren't
# heavily multi-threaded. Revise down once actual peak RSS is observed on
# this cluster (check `seff <jobid>` after a successful run).
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                              # conda env to activate
#   CONDA_MODULE=Anaconda3                       # module providing conda
#   INPUT_DIR=results/sweep                      # run-log tree to read
#   PROBLEMS_CONFIG=configs/final_problems.json  # problem suite config
#   OUTPUT_DIR=results/ndig_comparison           # where PDFs/stats report are written
#   ALPHA=0.05                                   # Friedman gate / Wilcoxon significance level
#
# Submit:
#   sbatch scripts/ndig_comparison.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep OUTPUT_DIR=results/ndig_comparison/rerun ALPHA=0.10 \
#     sbatch --export=ALL,INPUT_DIR,OUTPUT_DIR,ALPHA scripts/ndig_comparison.sbatch
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
ALPHA="${ALPHA:-0.05}"

echo "===================================================================="
echo "ITCAS NDIG comparison | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} problems_config=${PROBLEMS_CONFIG} output_dir=${OUTPUT_DIR} alpha=${ALPHA}"
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

# --- Produce the NDIG comparison PDFs + stats report -------------------------
mkdir -p "${OUTPUT_DIR}"
ARGS=(--input-dir "${INPUT_DIR}" \
      --problems-config "${PROBLEMS_CONFIG}" \
      --output-dir "${OUTPUT_DIR}" \
      --alpha "${ALPHA}")

srun --kill-on-bad-exit=1 python -m itcas.reporting.ndig_comparison "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[ndig_comparison] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ndig_comparison] outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
