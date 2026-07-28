#!/usr/bin/env bash
#SBATCH --job-name=itcas_rtb_short
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/itcas_rtb_short-%j.out
#SBATCH --error=slurm_logs/itcas_rtb_short-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=FAIL
# ---------------------------------------------------------------------------
# runtime_benchmark_short_lane.sbatch — high-priority, time-boxed worker.
#
# The 'gpu' partition is 100% GPU-allocated cluster-wide right now (139
# pending jobs queued, batch-short/batch-long backfill estimate ~2 days out
# for this account's current fairshare/priority). The 'short' QOS has much
# higher scheduling priority (15000 vs 2000) and backfills into small gaps
# fast, but caps at gres/gpu=2 and cpu=10 PER USER and --time<=02:00:00 --
# so this job drains whatever (method,problem) cells are still missing
# (checked fresh each segment -- shared with the batch-short per-problem
# jobs and any other lane, so nothing is double-recorded once persisted),
# 2 concurrent cells packed onto this one GPU (same rationale as
# runtime_benchmark_problem.sbatch: ~9.2/32GB VRAM, ~6% util per cell), and
# stops launching new cells with SAFETY_BUFFER left on the wall clock so
# in-flight cells aren't killed mid-run by the QOS time limit. The caller
# chains segments via --dependency=afterany so a lane keeps making progress
# across many 2h windows without needing to be resubmitted by hand.
#
# Required env: none (reads config defaults). Optional: N_CONCURRENT (2),
# SAFETY_BUFFER_SEC (900), CONDA_ENV, CONDA_MODULE, EXPERIMENTS_CONFIG,
# PROBLEMS_CONFIG, DEVICE, CELLS_DIR.
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

N_CONCURRENT="${N_CONCURRENT:-2}"
SAFETY_BUFFER_SEC="${SAFETY_BUFFER_SEC:-900}"   # stop launching new cells 15min before the 2h cap
CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
export EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG:-configs/experiments.json}"
export PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
export DEVICE="${DEVICE:-cuda}"
export CELLS_DIR="${CELLS_DIR:-results/runtime_benchmark_cells}"
mkdir -p "${CELLS_DIR}"

DEADLINE=$(( $(date +%s) + 2*3600 - SAFETY_BUFFER_SEC ))

echo "===================================================================="
echo "ITCAS runtime benchmark (short-QOS lane) | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "n_concurrent=${N_CONCURRENT} device=${DEVICE} seed=0 deadline=$(date -d "@${DEADLINE}" '+%H:%M:%S' 2>/dev/null || echo "${DEADLINE}")"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true

if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[rtb_short] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

CPUS="${SLURM_CPUS_PER_TASK:-5}"
PER_CELL_THREADS=$(( CPUS / N_CONCURRENT ))
(( PER_CELL_THREADS < 1 )) && PER_CELL_THREADS=1
export OMP_NUM_THREADS="${PER_CELL_THREADS}" MKL_NUM_THREADS="${PER_CELL_THREADS}" \
       OPENBLAS_NUM_THREADS="${PER_CELL_THREADS}" NUMEXPR_NUM_THREADS="${PER_CELL_THREADS}"
export MPLBACKEND="Agg"

DEFAULT_METHODS="itcas_ndig itcas_seq_ndig random straddle_then_sample_lse10 straddle_then_sample_lse10_batch bes_then_sample_lse10 bes_then_sample_lse10_batch cas_eci cas_eci_batch moc_cas_hard moc_cas_hard_batch"

# --- Build the current global remaining-cell list (re-checked fresh here, --
# shared safely with any other lane/job since a cell is only "claimed" by
# actually writing its CSV; worst case on a race is a duplicate re-run of the
# same cell, not a correctness problem). ---------------------------------
PROBLEMS=$(python - <<PYEOF
from itcas.reporting.summary import _synthetic_problems
print("\n".join(_synthetic_problems("${PROBLEMS_CONFIG}")))
PYEOF
)

REMAINING=()
while IFS= read -r PROBLEM; do
    [[ -z "${PROBLEM}" ]] && continue
    for m in ${DEFAULT_METHODS}; do
        [[ -f "${CELLS_DIR}/${PROBLEM}__${m}.csv" ]] && continue
        REMAINING+=("${m} ${PROBLEM}")
    done
done <<< "${PROBLEMS}"

echo "[rtb_short] ${#REMAINING[@]} cell(s) still remaining at segment start."

inflight=0
launched=0
for pair in "${REMAINING[@]}"; do
    read -r m p <<< "${pair}"
    # Re-check right before launch (another lane may have finished it since we listed).
    [[ -f "${CELLS_DIR}/${p}__${m}.csv" ]] && continue
    if (( $(date +%s) >= DEADLINE )); then
        echo "[rtb_short] safety deadline reached, not launching more new cells this segment."
        break
    fi
    bash "${REPO_ROOT}/scripts/runtime_benchmark_cell.sh" "${m}" "${p}" &
    inflight=$(( inflight + 1 ))
    launched=$(( launched + 1 ))
    if (( inflight >= N_CONCURRENT )); then
        wait -n || true
        inflight=$(( inflight - 1 ))
    fi
done
wait

echo "===================================================================="
echo "[rtb_short] segment finished at $(date '+%Y-%m-%d %H:%M:%S'), launched ${launched} cell(s) this segment"
echo "===================================================================="
