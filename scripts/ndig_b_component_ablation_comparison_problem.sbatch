#!/usr/bin/env bash
#SBATCH --job-name=itcas_ncab_problem
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/itcas_ncab_problem-%j.out
#SBATCH --error=slurm_logs/itcas_ncab_problem-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=FAIL
# ---------------------------------------------------------------------------
# ndig_b_component_ablation_comparison_problem.sbatch — per-problem worker of
# the split NDIG-B acquisition-component ablation comparison pipeline
# (proposed method's full batch NDIG acquisition, itcas_ndig, vs. its four
# quality-component ablations itcas_ndig_no_infogain/itcas_edig/itcas_efig/
# itcas_ndig_pof_entropy).
#
# Submitted automatically by ndig_b_component_ablation_comparison_dispatch.sbatch
# (the orchestrator). Do not submit this script directly unless you know what
# you are doing.
#
# For one problem (PROBLEM) this job runs
#   python -m itcas.summarize --ndig-b-component-ablation-problem "${PROBLEM}"
# (see itcas.reporting.ndig_b_component_ablation_comparison.summarize_ndig_b_component_ablation_problem),
# which:
#   1. Loads only PROBLEM's own runs, restricted to the 5 METHODS (itcas_ndig,
#      itcas_ndig_no_infogain, itcas_edig, itcas_efig,
#      itcas_ndig_pof_entropy) -- never any other problem's .jsonl logs, never
#      any other method.
#   2. Renders NO per-problem plot -- this pipeline has no per-problem PDF at
#      all (see ndig_b_component_ablation_comparison.py's module docstring);
#      --output-dir is passed only for CLI/signature symmetry and is
#      otherwise unused here.
#   3. Writes <METRICS_DIR>/<PROBLEM>_ndig_b_component_ablation_metrics.json
#      (a small intermediate summary, NOT raw run data) for the aggregate job.
#
# Required environment variables (set by the orchestrator via --export):
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
# Resource sizing: reuses ndig_kernel_ablation_comparison_problem.sbatch's
# --mem=10G / --time=01:00:00 / --cpus-per-task=2 verbatim rather than a fresh
# guess. This pipeline covers slightly MORE data per problem than that one --
# 5 methods vs. 4 -- but likewise renders no per-problem PDF, so that
# script's already-conservative 10G/1h stays a reasonable starting bound
# here too, not an under-provisioned guess. REVISE after observing actual
# `seff <jobid>` numbers from the first real per-problem Slurm run of THIS
# pipeline on this cluster.
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
OUTPUT_DIR="${OUTPUT_DIR:-results/ndig_b_component_ablation_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
PROBLEM="${PROBLEM:?PROBLEM env var is required}"

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "===================================================================="
echo "ITCAS NDIG-B component-ablation comparison (per-problem) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "problem=${PROBLEM}"
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
        echo "[ncab_problem] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Compute this problem's ablation-comparison metrics ---------------------
srun --kill-on-bad-exit=1 python -m itcas.summarize \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --problems-config "${PROBLEMS_CONFIG}" \
    --ndig-b-component-ablation-problem "${PROBLEM}" \
    --save-metrics "${METRICS_DIR}"
RC=$?

echo "===================================================================="
echo "[ncab_problem] finished rc=${RC} problem=${PROBLEM} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ncab_problem] metrics written to: ${METRICS_DIR}/${PROBLEM}_ndig_b_component_ablation_metrics.json"
echo "===================================================================="
exit "${RC}"
