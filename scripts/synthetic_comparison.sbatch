#!/usr/bin/env bash
#SBATCH --job-name=itcas_synthetic_comparison
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/itcas_synthetic_comparison-%j.out
#SBATCH --error=slurm_logs/itcas_synthetic_comparison-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# synthetic_comparison.sbatch — MONOLITHIC single job (no array, no
# dependency), structurally like batch_vs_sequential.sbatch/ff_comparison.sbatch.
#
# *** DO NOT SUBMIT THIS SCRIPT FOR FULL-SCALE RUNS. ***
# This is the OLD, single-job code path that loads EVERY synthetic problem's
# full run history into memory simultaneously. It OOM-killed at 16G on job
# 3427762 (15.99/16.00 GB utilized, died ~11 min in, before writing any
# output). It is kept ONLY for quick local/small-scale testing (e.g. a
# handful of problems on a workstation, or a laptop-sized --input-dir). For
# the real full ~15-problem sweep, submit the memory-bounded split pipeline
# instead:
#   sbatch scripts/synthetic_comparison_dispatch.sbatch
# which fans out one Slurm job per problem
# (scripts/synthetic_comparison_problem.sbatch, --synthetic-comparison-problem)
# and then a small aggregate job
# (scripts/synthetic_comparison_aggregate.sbatch, --synthetic-comparison-aggregate-from)
# that only reads the small intermediate metrics JSONs, never raw run logs.
# Both code paths are numerically identical (58/58 tests pass); this
# monolithic one is simply not memory-safe at full scale.
#
# Runs `python -m itcas.summarize --synthetic-comparison` (see
# itcas.reporting.summary.summarize_synthetic_comparison), which reads run
# logs under --input-dir for every problem in --problems-config EXCEPT
# spacecraft_formation_flying_a1 (the one problem backed by a real Basilisk
# simulation rather than a closed-form objective) and compares 6 hardcoded
# methods: itcas_ndig (the proposed method, batch) vs. five forced-sequential
# baselines (random, straddle_then_sample_lse10, bes_then_sample_lse10,
# cas_eci, moc_cas_hard). For each difficulty level (p0_01/p0_05/p0_10/p0_20)
# it writes, into its own <output-dir>/<difficulty>/ subfolder:
#   - one PDF PER SYNTHETIC PROBLEM (metric curves + raw product, no
#     product-rank column), keyed on total individual evaluations only (no
#     steps variant)
#   - one standalone average-rank PDF (each method's average rank, 1 = best,
#     per metric + product, averaged across every problem at that difficulty)
#   - one standalone relative-AUC PDF (each method's AUC relative to the best
#     AUC seen anywhere for that problem/metric, regardless of method or
#     seed, averaged across seeds then problems; 1.0 = matched the best)
#   - a Friedman-gated Holm-Bonferroni-corrected one-sided paired Wilcoxon
#     dominance test (H1: itcas_ndig > each baseline) stats report (json+md)
#
# This is additive: it does not touch or replace the existing sharded
# summarize.sbatch -> summarize_problem.sbatch -> summarize_aggregate.sbatch
# pipeline (the box-plot-over-every-discovered-method summary), which keeps
# working exactly as before via its own --benchmark/--save-metrics/
# --aggregate-from flags.
#
# CPU-only (matplotlib PDF rendering + Wilcoxon/Friedman tests, no GPU, no
# torch model fitting -- just reads already-logged run curves).
#
# Resource sizing: NOT measured on this cluster at authoring time -- sized to
# match batch_vs_sequential.sbatch's measured baseline (16 problems x 3
# families, 20 seeds each, ~15 min wall-clock, --time=01:00:00 --mem=16G
# --cpus-per-task=4), since this job covers a comparable data volume (15
# synthetic problems x 6 methods x 4 difficulties x ~20 seeds). Revise once
# actual run time is observed on this cluster.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                              # conda env to activate
#   CONDA_MODULE=Anaconda3                       # module providing conda
#   INPUT_DIR=results/sweep                      # run-log tree to read
#   PROBLEMS_CONFIG=configs/final_problems.json  # problem suite config
#   OUTPUT_DIR=results/synthetic_comparison       # where <difficulty>/ subfolders are written
#   ALPHA=0.05                                    # Friedman gate / Wilcoxon significance level
#
# Submit:
#   sbatch scripts/synthetic_comparison.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep OUTPUT_DIR=results/synthetic_comparison/rerun ALPHA=0.10 \
#     sbatch --export=ALL,INPUT_DIR,OUTPUT_DIR,ALPHA scripts/synthetic_comparison.sbatch
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
OUTPUT_DIR="${OUTPUT_DIR:-results/synthetic_comparison}"
ALPHA="${ALPHA:-0.05}"

echo "===================================================================="
echo "ITCAS synthetic comparison | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
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
        echo "[synthetic_comparison] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Produce the per-difficulty synthetic comparison PDFs + stats reports ---
mkdir -p "${OUTPUT_DIR}"
ARGS=(--input-dir "${INPUT_DIR}" \
      --problems-config "${PROBLEMS_CONFIG}" \
      --output-dir "${OUTPUT_DIR}" \
      --alpha "${ALPHA}" \
      --synthetic-comparison)

srun --kill-on-bad-exit=1 python -m itcas.summarize "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[synthetic_comparison] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[synthetic_comparison] outputs written under: ${OUTPUT_DIR}/<difficulty>/"
echo "===================================================================="
exit "${RC}"
