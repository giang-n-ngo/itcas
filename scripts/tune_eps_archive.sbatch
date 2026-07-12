#!/usr/bin/env bash
#SBATCH --job-name=itcas_tune_eps_archive
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/itcas_tune_eps_archive-%j.out
#SBATCH --error=slurm_logs/itcas_tune_eps_archive-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# tune_eps_archive.sbatch — single CPU-only job that runs the offline
# eps_archive calibration (itcas/reporting/tune_eps_archive.py) over the full
# results/sweep tree and patches configs/experiments.json in place.
#
# Wraps the Scientific Coder's entry point (python -m
# itcas.reporting.tune_eps_archive); no science inside.
#
# Why a single job (no array fan-out like summarize.sbatch):
#   calibrate_all() already processes one (problem, difficulty) group at a
#   time and discards each group's loaded runs before moving to the next, so
#   peak RSS is bounded to ~1-2 GB per group regardless of sweep size. With
#   ~110 (problem, difficulty) groups total and ~10-15s/group observed in
#   login-node timing tests on the larger groups, the whole calibration is
#   expected to finish serially in well under an hour -- not worth the
#   overhead/complexity of a job array + dependent aggregate step.
#
# Resource rationale:
#   - partition=cpu, qos=cpu : CPU-only workload (pdist over pooled objective
#     vectors + summary/jsonl parsing), no GPU work.
#   - cpus-per-task=4        : light parallelism headroom for BLAS calls
#     inside pdist/percentile; the workload is not itself multi-threaded.
#   - mem=16G                : generous margin above the ~1-2 GB/group figure
#     from small-scale login-node testing (subsample cap --max-pool-size
#     bounds pdist's O(n^2) memory regardless of how many points a group
#     pools before subsampling).
#   - time=02:00:00           : ~110 groups x ~15s/group is well under an
#     hour; doubled again for margin since the full sweep hasn't been run
#     end-to-end through this script yet.
#
# This job WRITES for real (no --dry-run): it patches only the eps_archive
# numeric field per (problem, difficulty) block in configs/experiments.json,
# leaving everything else in that file (including any other pending
# hand-edits already in the working tree) untouched.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                          # conda env to activate
#   CONDA_MODULE=Anaconda3                   # module providing conda
#   SWEEP_ROOT=results/sweep                 # sweep root to scan
#   CONFIG=configs/experiments.json          # config file to patch
#
# Submit:
#   sbatch scripts/tune_eps_archive.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs 2>/dev/null || \
    echo "[tune_eps] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
SWEEP_ROOT="${SWEEP_ROOT:-results/sweep}"
CONFIG="${CONFIG:-configs/experiments.json}"

echo "===================================================================="
echo "ITCAS eps_archive calibration | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "sweep_root=${SWEEP_ROOT} config=${CONFIG} conda_env=${CONDA_ENV}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[tune_eps] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"

# --- Run calibration (writes configs/experiments.json for real) -----------
srun --kill-on-bad-exit=1 python -m itcas.reporting.tune_eps_archive \
    --sweep-root "${SWEEP_ROOT}" \
    --config "${CONFIG}"
RC=$?

echo "===================================================================="
echo "[tune_eps] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[tune_eps] config patched: ${CONFIG}"
echo "===================================================================="
exit "${RC}"
