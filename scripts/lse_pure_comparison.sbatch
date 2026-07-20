#!/usr/bin/env bash
#SBATCH --job-name=itcas_lse_pure_comparison
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/itcas_lse_pure_comparison-%j.out
#SBATCH --error=slurm_logs/itcas_lse_pure_comparison-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# lse_pure_comparison.sbatch — single lightweight CPU job (no array, no
# dependency), structurally identical to casd_comparison.sbatch (its closest
# sibling here), minus an ALPHA env var since this module has no --alpha
# flag / no stats report (see below).
#
# Runs itcas/reporting/lse_pure_comparison.py, which reads run logs under
# --input-dir for all 15 synthetic problems in --config's "problems" list
# (configs/lse_pure_comparison.json by default; real-world problems
# spacecraft_formation_flying_a1 and casd_llm are excluded, see module
# docstring) at the config's one fixed default_difficulty (p0_05), and for
# each of the two LSE bases (straddle, bes) x two settings (sequential,
# batch) compares that base's pure (single-stage) LSE acquisition against its
# three _then_sample_lse{10,25,50} siblings (4 methods per group). It renders
# one single-row PDF per (setting, base, problem) --
# {setting}_{base}_{problem}_vs_{evaluations,steps}.pdf, up to 2 x 2 x 15 = 60
# PDFs -- plus two standalone cross-cutting summary PDFs
# (lse_pure_comparison_avg_rank.pdf, lse_pure_comparison_relative_auc.pdf)
# aggregating average rank / relative-AUC across every row produced. Unlike
# casd_comparison.py / ff_comparison.py / ndig_comparison.py, this module
# ships NO stats/Wilcoxon report -- there is no natural "proposed" method
# among "pure vs 10%/25%/50% Stage-1 split" -- so its CLI has no --alpha flag
# and this script sets no ALPHA env var.
#
# CPU-only (matplotlib PDF rendering + log parsing, no GPU, no torch model
# fitting -- just reads already-logged run curves).
#
# Resource sizing: initially estimated (without cluster access) at
# --time=00:30:00 / --mem=12G by analogy to casd_comparison.sbatch and
# school_comparison.sbatch, then corrected after a real smoke-test submission
# on this cluster (job 3442373, 2026-07-19) OOM-killed at --mem=12G: peak RSS
# hit ~12.57GB within 6m43s while still collecting runs for the very first
# (setting, base) group (before any PDF was written), i.e. before even
# reaching school_comparison.sbatch's own measured ~16GB peak for a
# comparable 16-problem x 2-base sequential-only sweep -- this job also loads
# the batch setting on top of sequential (roughly 2x school_comparison's
# sequential-only data volume) via four separate _collect_family_runs calls
# (one per (setting, base) group) that are never freed/reused across groups.
# Resized to --mem=32G (matching school_comparison.sbatch's own ceiling, itself
# ~2x its own measured 16GB peak) and --time=00:45:00 (up from the 30 min
# estimate, giving headroom beyond the 6m43s-to-OOM point actually observed).
# Revise further once a full successful run's actual peak RSS/wall-clock is
# observed on this cluster.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                              # conda env to activate
#   CONDA_MODULE=Anaconda3                       # module providing conda
#   INPUT_DIR=results/sweep                      # run-log tree to read
#   OUTPUT_DIR=results/lse_pure_comparison        # where the PDFs are written
#   CONFIG=configs/lse_pure_comparison.json       # default_difficulty/lse_proportions/bases/problems config
#
# Submit:
#   sbatch scripts/lse_pure_comparison.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep OUTPUT_DIR=results/lse_pure_comparison/rerun \
#     sbatch --export=ALL,INPUT_DIR,OUTPUT_DIR scripts/lse_pure_comparison.sbatch
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
OUTPUT_DIR="${OUTPUT_DIR:-results/lse_pure_comparison}"
CONFIG="${CONFIG:-configs/lse_pure_comparison.json}"

echo "===================================================================="
echo "ITCAS LSE pure comparison | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} output_dir=${OUTPUT_DIR} config=${CONFIG}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[lse_pure_comparison] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Produce the LSE pure-vs-then-sample comparison PDFs --------------------
mkdir -p "${OUTPUT_DIR}"
ARGS=(--input-dir "${INPUT_DIR}" \
      --config "${CONFIG}" \
      --output-dir "${OUTPUT_DIR}")

srun --kill-on-bad-exit=1 python -m itcas.reporting.lse_pure_comparison "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[lse_pure_comparison] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[lse_pure_comparison] outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
