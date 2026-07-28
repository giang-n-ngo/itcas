#!/usr/bin/env bash
#SBATCH --job-name=itcas_rtb_dispatch
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_rtb_dispatch-%j.out
#SBATCH --error=slurm_logs/itcas_rtb_dispatch-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# runtime_benchmark_dispatch.sbatch — orchestrator for the split runtime-
# benchmark sweep. Replaces scripts/runtime_benchmark.sbatch (monolithic,
# 165 cells sequential on 1 GPU, too slow for a same-day deadline and only
# wrote output at the very end -- see runtime_benchmark_problem.sbatch's
# header for the full rationale).
#
# For each of the 15 synthetic problems (from PROBLEMS_CONFIG), submits ONE
# runtime_benchmark_problem.sbatch job (1 GPU, 2 concurrent cells/GPU) -- but
# ONLY for methods not already recorded in
# results/runtime_benchmark_cells/<problem>__<method>.csv, and skips
# submitting a job entirely if that problem's 11 cells are all already
# recorded. This is what makes it safe to resubmit: already-recorded cells
# (including ones salvaged from the earlier monolithic job's log) are never
# redone.
#
# After all per-problem jobs finish, submits runtime_benchmark_aggregate.sbatch
# (afterok) to build the final results/runtime_benchmark.md from the
# per-cell CSVs (never from any Slurm .out log).
#
# Submit:
#   sbatch scripts/runtime_benchmark_dispatch.sbatch
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
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG:-configs/experiments.json}"
CELLS_DIR="${CELLS_DIR:-results/runtime_benchmark_cells}"
OUTPUT="${OUTPUT:-results/runtime_benchmark.md}"
N_CONCURRENT="${N_CONCURRENT:-2}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "${CELLS_DIR}"

METHODS_LIST="itcas_ndig itcas_seq_ndig random straddle_then_sample_lse10 straddle_then_sample_lse10_batch bes_then_sample_lse10 bes_then_sample_lse10_batch cas_eci cas_eci_batch moc_cas_hard moc_cas_hard_batch"

echo "===================================================================="
echo "ITCAS runtime benchmark dispatch | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "cells_dir=${CELLS_DIR} problems_config=${PROBLEMS_CONFIG}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[rtb_dispatch] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
fi
export PYTHONNOUSERSITE=1

# --- Discover the 15 synthetic problems (same helper the pipeline uses) -----
PROBLEMS=$(python - <<PYEOF
from itcas.reporting.summary import _synthetic_problems
print("\n".join(_synthetic_problems("${PROBLEMS_CONFIG}")))
PYEOF
)

# --- Submit one per-problem job per problem that still has missing cells ---
JOB_IDS=""
while IFS= read -r PROBLEM; do
    [[ -z "${PROBLEM}" ]] && continue
    MISSING_METHODS=""
    for m in ${METHODS_LIST}; do
        if [[ ! -f "${CELLS_DIR}/${PROBLEM}__${m}.csv" ]]; then
            MISSING_METHODS="${MISSING_METHODS} ${m}"
        fi
    done
    MISSING_METHODS="${MISSING_METHODS# }"
    if [[ -z "${MISSING_METHODS}" ]]; then
        echo "[rtb_dispatch] '${PROBLEM}': all cells already recorded, skipping."
        continue
    fi
    JID=$(sbatch --parsable \
        --export=ALL,PROBLEM="${PROBLEM}",METHODS="${MISSING_METHODS}",N_CONCURRENT="${N_CONCURRENT}",DEVICE="${DEVICE}",EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG}",PROBLEMS_CONFIG="${PROBLEMS_CONFIG}",CELLS_DIR="${CELLS_DIR}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
        "${REPO_ROOT}/scripts/runtime_benchmark_problem.sbatch")
    echo "[rtb_dispatch] Submitted ${JID} for '${PROBLEM}' (${MISSING_METHODS})"
    JOB_IDS="${JOB_IDS}:${JID}"
done <<< "${PROBLEMS}"

JOB_IDS="${JOB_IDS#:}"

if [[ -z "${JOB_IDS}" ]]; then
    echo "[rtb_dispatch] Nothing to run -- all cells already recorded. Submitting aggregate directly."
    AGG_JID=$(sbatch --parsable \
        --export=ALL,CELLS_DIR="${CELLS_DIR}",OUTPUT="${OUTPUT}",EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG}",PROBLEMS_CONFIG="${PROBLEMS_CONFIG}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
        "${REPO_ROOT}/scripts/runtime_benchmark_aggregate.sbatch")
    echo "[rtb_dispatch] Submitted aggregate job ${AGG_JID}"
    exit 0
fi

AGG_JID=$(sbatch --parsable \
    --dependency="afterok:${JOB_IDS}" \
    --export=ALL,CELLS_DIR="${CELLS_DIR}",OUTPUT="${OUTPUT}",EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG}",PROBLEMS_CONFIG="${PROBLEMS_CONFIG}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
    "${REPO_ROOT}/scripts/runtime_benchmark_aggregate.sbatch")

echo "===================================================================="
echo "[rtb_dispatch] Per-problem jobs: ${JOB_IDS}"
echo "[rtb_dispatch] Aggregate job:    ${AGG_JID} (depends on above)"
echo "[rtb_dispatch] Monitor with: squeue -j ${JOB_IDS//:/,},${AGG_JID}"
echo "===================================================================="
