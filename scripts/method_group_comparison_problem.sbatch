#!/usr/bin/env bash
#SBATCH --job-name=itcas_mgc_problem
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/itcas_mgc_problem-%j.out
#SBATCH --error=slurm_logs/itcas_mgc_problem-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=FAIL
# ---------------------------------------------------------------------------
# method_group_comparison_problem.sbatch — per-problem worker of the split
# method-group-comparison pipeline (batch-methods-vs-each-other or
# sequential-methods-vs-each-other, selected by METHOD_GROUP).
#
# Submitted automatically by method_group_comparison_dispatch.sbatch (the
# orchestrator), which also overrides this script's --job-name/-o/-e on the
# `sbatch` command line to bake METHOD_GROUP into the job name and log paths
# (the #SBATCH header below can't reference ${METHOD_GROUP} itself -- it is
# parsed before --export env vars are available). Do not submit this script
# directly unless you know what you are doing.
#
# For one problem (PROBLEM) and one METHOD_GROUP (batch|sequential) this job
# runs
#   python -m itcas.summarize --method-group-comparison-problem "${PROBLEM}" \
#          --method-group "${METHOD_GROUP}"
# (see itcas.reporting.method_group_comparison.summarize_method_group_comparison_problem),
# which:
#   1. Loads only PROBLEM's own runs, restricted to METHOD_GROUP's 5 methods
#      (see METHOD_GROUPS in method_group_comparison.py; `random` is excluded
#      from both groups) -- never any other problem's .jsonl logs, never the
#      other group's methods.
#   2. Renders NO per-problem plot -- unlike synthetic_comparison_problem.sbatch,
#      this pipeline has no per-problem PDF at all (see
#      method_group_comparison.py's module docstring); --output-dir is passed
#      only for CLI/signature symmetry and is otherwise unused here.
#   3. Writes <METRICS_DIR>/<PROBLEM>_<METHOD_GROUP>_comparison_metrics.json
#      (a small intermediate summary, NOT raw run data) for the aggregate job.
#
# Required environment variables (set by the orchestrator via --export):
#   METHOD_GROUP      "batch" or "sequential"
#   PROBLEM           synthetic problem name (e.g. zakharov_ackley_10d)
#   INPUT_DIR         sweep root containing run logs
#   PROBLEMS_CONFIG   final-problems config, used only to validate PROBLEM is
#                      a synthetic problem (not spacecraft_formation_flying_a1
#                      / casd_llm)
#   OUTPUT_DIR        accepted for CLI/signature symmetry only (unused, see
#                      above)
#   METRICS_DIR       where the intermediate metrics JSON is written
#   CONDA_ENV         conda environment name
#   CONDA_MODULE      module providing conda
#
# Resource sizing: reuses synthetic_comparison_problem.sbatch's --mem=10G /
# --time=01:00:00 / --cpus-per-task=2 verbatim rather than a fresh guess.
# This pipeline covers LESS data per problem than that one -- 5 methods vs.
# 6, and no per-problem PDF is rendered here at all (no
# batch_vs_sequential.plot_group_grid call, unlike
# summarize_synthetic_comparison_problem) -- so synthetic_comparison_problem.sbatch's
# already-conservative 10G/1h (itself generous headroom above a local,
# off-cluster /usr/bin/time -v smoke test that measured ~1.4-2.9GB peak RSS
# across the smallest/largest synthetic problems, see that script's own
# header comment) is safely sufficient here too, not an under-provisioned
# guess. REVISE after observing actual `seff <jobid>` numbers from the first
# real per-problem Slurm run of THIS pipeline on this cluster.
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
    echo "[mgc_problem] ERROR: METHOD_GROUP must be exactly 'batch' or 'sequential'," \
         "got '${METHOD_GROUP}'." >&2
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
INPUT_DIR="${INPUT_DIR:-results/sweep}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${METHOD_GROUP}_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
PROBLEM="${PROBLEM:?PROBLEM env var is required}"

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "===================================================================="
echo "ITCAS method-group comparison (per-problem) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "method_group=${METHOD_GROUP} problem=${PROBLEM}"
echo "input_dir=${INPUT_DIR} problems_config=${PROBLEMS_CONFIG}"
echo "output_dir=${OUTPUT_DIR} metrics_dir=${METRICS_DIR}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[mgc_problem] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Compute this problem's metrics for METHOD_GROUP -------------------------
srun --kill-on-bad-exit=1 python -m itcas.summarize \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --problems-config "${PROBLEMS_CONFIG}" \
    --method-group-comparison-problem "${PROBLEM}" \
    --method-group "${METHOD_GROUP}" \
    --save-metrics "${METRICS_DIR}"
RC=$?

echo "===================================================================="
echo "[mgc_problem] finished rc=${RC} method_group=${METHOD_GROUP} problem=${PROBLEM} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[mgc_problem] metrics written to: ${METRICS_DIR}/${PROBLEM}_${METHOD_GROUP}_comparison_metrics.json"
echo "===================================================================="
exit "${RC}"
