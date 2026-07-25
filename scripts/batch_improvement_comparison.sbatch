#!/usr/bin/env bash
#SBATCH --job-name=itcas_batch_improvement_comparison
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_batch_improvement_comparison-%j.out
#SBATCH --error=slurm_logs/itcas_batch_improvement_comparison-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# batch_improvement_comparison.sbatch — single lightweight CPU job (no array,
# no dependency), structurally like ff_comparison.sbatch / batch_vs_sequential.sbatch
# (now deleted; superseded by this report -- see its own module docstring).
#
# Runs itcas/reporting/batch_improvement_comparison.py, which reads run logs
# under --input-dir for the synthetic problem suite in --problems-config (the
# two real-world problems, spacecraft_formation_flying_a1/casd_llm, are
# excluded) and renders a SINGLE heatmap PDF: for each of 5 hardcoded method
# families (itcas, bes, straddle, eci, moc_cas_hard -- 10 methods total, each
# family's sequential vs. batch variant) x 4 shared difficulty levels
# (p0_01/p0_05/p0_10/p0_20) x metric column, the mean percentage AUC change
# from switching sequential -> batch. Unlike batch_vs_sequential.py /
# ndig_comparison.py (both now retired), this module's CLI takes no --alpha
# flag -- it is purely descriptive (no Friedman/Wilcoxon significance
# testing), and the method-family list is hardcoded in the module itself, not
# read from --methods-config.
#
# CPU-only (matplotlib PDF rendering + numpy aggregation, no GPU, no torch
# model fitting -- just reads already-logged run curves via the same
# disk-backed curve cache ff_comparison.py/casd_comparison.py use).
#
# Resource sizing: sized off real `sacct` accounting history for the closest
# already-completed comparable jobs on this cluster (no fresh profiling
# submission needed) rather than an unmeasured guess:
#   - itcas_batch_vs_sequential (3 families x 16 problems incl. real-world,
#     WITH Friedman/Wilcoxon stats; jobs 3329657/3339878/3396695/3398049/
#     3398705): elapsed 12:15-24:57, MaxRSS ~15.98-15.99 GiB against its own
#     16G request -- uncomfortably tight in hindsight.
#   - itcas_ndig_comparison (3 hardcoded methods, real-world-only scope +
#     spacecraft 10-level grid, WITH stats; successful run 3407655): elapsed
#     7:09, MaxRSS ~16.34 GiB against a 32G request -- i.e. despite a much
#     narrower method/problem scope than batch_vs_sequential, it peaked at
#     essentially the same ~16 GiB. This (plus that script's own comment
#     about an earlier 8G attempt, job 3407326, OOM-killing against the
#     ~43k-file results/sweep tree) indicates the memory floor here is
#     dominated by the initial full --input-dir tree scan/parse, not by
#     which methods/problems/difficulties are actually used downstream.
#   - itcas_ff_comparison (1 problem only, WITH stats; several runs): elapsed
#     well under a minute, MaxRSS ~1.9-2.0 GiB against 4G -- confirms the
#     ~16 GiB floor only appears once reading broadly across the tree, not
#     for a single-problem read.
# This report reads across the full synthetic suite (14 problems, comparable
# in breadth to batch_vs_sequential's 16) so it should hit the same ~16 GiB
# tree-scan floor, plus more methods per problem (10 vs. 6) but zero
# Friedman/Wilcoxon overhead (the costliest per-cell step in the sibling
# reports) and only 4 fixed difficulty levels per problem rather than
# per-problem real-world scales. Net: --mem=20G gives ~25% headroom above
# the consistently-observed ~16-16.3 GiB peaks across all 3 job types (more
# margin than batch_vs_sequential's own too-tight 16G), while staying well
# below ndig_comparison's empirically-oversized 32G. --time=00:30:00 gives
# ~1.2x headroom above batch_vs_sequential's worst observed 24:57 elapsed,
# reasonable given comparable-or-larger problem/method breadth even without
# the stats-testing overhead. --cpus-per-task=4 matches every sibling report
# since PDF rendering/aggregation isn't heavily multi-threaded. Revise once
# an actual run completes on this cluster (check `seff <jobid>`).
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                              # conda env to activate
#   CONDA_MODULE=Anaconda3                       # module providing conda
#   INPUT_DIR=results/sweep                      # run-log tree to read
#   PROBLEMS_CONFIG=configs/final_problems.json  # synthetic problem suite config
#   OUTPUT_DIR=results/batch_improvement_comparison  # where the heatmap PDF is written
#   AUC_CACHE_DIR=                               # disk-backed curve cache dir;
#                                                 # empty/unset -> module default
#                                                 # (<input-dir's parent>/auc_cache)
#
# Submit:
#   sbatch scripts/batch_improvement_comparison.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep OUTPUT_DIR=results/batch_improvement_comparison/rerun \
#     sbatch --export=ALL,INPUT_DIR,OUTPUT_DIR scripts/batch_improvement_comparison.sbatch
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
OUTPUT_DIR="${OUTPUT_DIR:-results/batch_improvement_comparison}"
AUC_CACHE_DIR="${AUC_CACHE_DIR:-}"

echo "===================================================================="
echo "ITCAS batch improvement comparison | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
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
        echo "[batch_improvement_comparison] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Produce the batch-improvement heatmap PDF ------------------------------
mkdir -p "${OUTPUT_DIR}"
ARGS=(--input-dir "${INPUT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --problems-config "${PROBLEMS_CONFIG}")
if [[ -n "${AUC_CACHE_DIR}" ]]; then
    ARGS+=(--auc-cache-dir "${AUC_CACHE_DIR}")
fi

srun --kill-on-bad-exit=1 python -m itcas.reporting.batch_improvement_comparison "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[batch_improvement_comparison] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[batch_improvement_comparison] outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
