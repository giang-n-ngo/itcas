#!/usr/bin/env bash
#SBATCH --job-name=itcas_rtb_problem
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/itcas_rtb_problem-%j.out
#SBATCH --error=slurm_logs/itcas_rtb_problem-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --requeue
# ---------------------------------------------------------------------------
# runtime_benchmark_problem.sbatch — per-problem worker for the split
# runtime-benchmark sweep (replaces the monolithic scripts/runtime_benchmark.sbatch,
# which ran all 11 methods x 15 problems sequentially on ONE GPU and only
# wrote output after all 165 cells finished -- unrecoverable on any crash and
# far too slow to hit a same-day deadline).
#
# Runs the 11 methods for ONE PROBLEM (seed=0 only -- this is a wall-clock
# timing tool, not the seed sweep), 2 concurrent processes sharing this job's
# single GPU (observed: one cell uses ~9.2/32GB VRAM and ~6% GPU util, so the
# GPU is heavily underused by a single process -- packing 2 concurrent gives
# a ~2x speedup per GPU with comfortable memory headroom). Each cell is its
# own `python -m itcas.reporting.runtime_benchmark` invocation (via
# runtime_benchmark_cell.sh) that persists its own result to
# results/runtime_benchmark_cells/<problem>__<method>.csv immediately --
# never relies on this job's Slurm log as the record of truth, and already-
# recorded cells are skipped so this is safe to requeue.
#
# Required env: PROBLEM
# Optional env: METHODS (space-separated, default: all 11), N_CONCURRENT
#               (default 2), CONDA_ENV, CONDA_MODULE, EXPERIMENTS_CONFIG,
#               PROBLEMS_CONFIG, DEVICE, CELLS_DIR
#
# Submitted automatically by runtime_benchmark_dispatch.sbatch. Can also be
# submitted directly for one problem:
#   sbatch --export=ALL,PROBLEM=zdt3_6d scripts/runtime_benchmark_problem.sbatch
# ---------------------------------------------------------------------------
set -u -o pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs 2>/dev/null || true

PROBLEM="${PROBLEM:?PROBLEM env var is required}"
DEFAULT_METHODS="itcas_ndig itcas_seq_ndig random straddle_then_sample_lse10 straddle_then_sample_lse10_batch bes_then_sample_lse10 bes_then_sample_lse10_batch cas_eci cas_eci_batch moc_cas_hard moc_cas_hard_batch"
METHODS="${METHODS:-${DEFAULT_METHODS}}"
N_CONCURRENT="${N_CONCURRENT:-2}"
CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
export EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG:-configs/experiments.json}"
export PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
export DEVICE="${DEVICE:-cuda}"
export CELLS_DIR="${CELLS_DIR:-results/runtime_benchmark_cells}"
mkdir -p "${CELLS_DIR}"

echo "===================================================================="
echo "ITCAS runtime benchmark (per-problem) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "problem=${PROBLEM} n_concurrent=${N_CONCURRENT} device=${DEVICE} seed=0"
echo "methods=${METHODS}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[rtb_problem] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

CPUS="${SLURM_CPUS_PER_TASK:-8}"
PER_CELL_THREADS=$(( CPUS / N_CONCURRENT ))
(( PER_CELL_THREADS < 1 )) && PER_CELL_THREADS=1
export OMP_NUM_THREADS="${PER_CELL_THREADS}" MKL_NUM_THREADS="${PER_CELL_THREADS}" \
       OPENBLAS_NUM_THREADS="${PER_CELL_THREADS}" NUMEXPR_NUM_THREADS="${PER_CELL_THREADS}"
export MPLBACKEND="Agg"

# --- Drain methods list, N_CONCURRENT in flight at a time (wait -n pool) ---
# Mirrors run_seedset.sh's concurrency pattern. runtime_benchmark_cell.sh
# self-skips any (method, problem) already recorded in CELLS_DIR.
read -r -a METHODS_ARR <<< "${METHODS}"
inflight=0
for m in "${METHODS_ARR[@]}"; do
    bash "${REPO_ROOT}/scripts/runtime_benchmark_cell.sh" "${m}" "${PROBLEM}" &
    inflight=$(( inflight + 1 ))
    if (( inflight >= N_CONCURRENT )); then
        wait -n || true
        inflight=$(( inflight - 1 ))
    fi
done
wait

# --- Final accounting: this problem is done iff every method's CSV exists --
MISSING=0
for m in "${METHODS_ARR[@]}"; do
    [[ -f "${CELLS_DIR}/${PROBLEM}__${m}.csv" ]] || MISSING=$(( MISSING + 1 ))
done

echo "===================================================================="
if (( MISSING == 0 )); then
    echo "[rtb_problem] COMPLETED problem=${PROBLEM} at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "===================================================================="
    exit 0
else
    echo "[rtb_problem] INCOMPLETE problem=${PROBLEM}: ${MISSING}/${#METHODS_ARR[@]} cell(s) missing at $(date '+%Y-%m-%d %H:%M:%S')" >&2
    echo "===================================================================="
    exit 1
fi
