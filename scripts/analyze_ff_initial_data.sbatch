#!/usr/bin/env bash
#SBATCH --job-name=ff_init_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/ff_initial_data/%x-%j.out
#SBATCH --error=slurm_logs/ff_initial_data/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu

# ---------------------------------------------------------------------------
# analyze_ff_initial_data.sbatch
#
# Operator wrapper for scripts/analyze_ff_initial_data.py.
# Computes:
#   - Set A: points within current FF design+context bounds
#   - Set B: points in A that satisfy filter thresholds
#   - Per-constraint and all-constraints counts over B
#
# Environment overrides (via --export=ALL,...):
#   CONDA_ENV=itcas
#   CONDA_MODULE=Anaconda3
#   SMARTSAT_ROOT=/home/<user>/SmartSat
#   RMSE_FILTER_M=1000        historical filter threshold (m)
#   FUEL_FILTER_KG=1.0        historical filter threshold (kg)
#   OUT_JSON=<path>           output file (default: Python constant-derived name)
#   SCAN=1                    sweep all threshold combinations (default: 0)
#   RMSE_THRESHOLDS=50,100,150,200    constraint scan grid (m)
#   FUEL_THRESHOLDS_G=50,100,150,200  constraint scan grid (g)
#   THRUST_THRESHOLDS=1,2,5,10        constraint scan grid (N)
#
# Submit:
#   sbatch scripts/analyze_ff_initial_data.sbatch
#   sbatch --export=ALL,SCAN=1 scripts/analyze_ff_initial_data.sbatch
#   sbatch --export=ALL,SMARTSAT_ROOT=/home/giangn/SmartSat,SCAN=1 scripts/analyze_ff_initial_data.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

# Slurm executes a staged copy of this script; use submit dir as repo root.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p slurm_logs/ff_initial_data

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
SMARTSAT_ROOT="${SMARTSAT_ROOT:-}"
RMSE_FILTER_M="${RMSE_FILTER_M:-1000}"
FUEL_FILTER_KG="${FUEL_FILTER_KG:-1.0}"
SCAN="${SCAN:-0}"
RMSE_THRESHOLDS="${RMSE_THRESHOLDS:-50,100,150,200}"
FUEL_THRESHOLDS_G="${FUEL_THRESHOLDS_G:-50,100,150,200}"
THRUST_THRESHOLDS="${THRUST_THRESHOLDS:-1,2,5,10}"
OUT_JSON="${OUT_JSON:-}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[ff_init_data] WARNING: could not activate ${CONDA_ENV}; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi

export PYTHONNOUSERSITE=1
THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"

# Resolve the output path from Python's constant if not set externally, so
# the log and mkdir below reflect the actual file Python will write.
if [[ -z "${OUT_JSON}" ]]; then
    if [[ "${SCAN}" == "1" ]]; then
        _PY_DEFAULT="from scripts.analyze_ff_initial_data import _DEFAULT_SCAN_OUT_JSON; print(_DEFAULT_SCAN_OUT_JSON)"
    else
        _PY_DEFAULT="from scripts.analyze_ff_initial_data import _DEFAULT_OUT_JSON; print(_DEFAULT_OUT_JSON)"
    fi
    OUT_JSON="$(python -c "${_PY_DEFAULT}" 2>/dev/null)" \
        || OUT_JSON="results/ff_initial_data/ff_initial_data.json"
fi
mkdir -p "$(dirname "${OUT_JSON}")"

echo "===================================================================="
echo "FF initial-data analysis | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "repo_root=${REPO_ROOT}"
echo "smartsat_root=${SMARTSAT_ROOT:-<auto>}"
echo "rmse_filter_m=${RMSE_FILTER_M} fuel_filter_kg=${FUEL_FILTER_KG}"
echo "scan=${SCAN}"
[[ "${SCAN}" == "1" ]] && echo "rmse_thresholds=${RMSE_THRESHOLDS} fuel_thresholds_g=${FUEL_THRESHOLDS_G} thrust_thresholds=${THRUST_THRESHOLDS}"
echo "out_json=${OUT_JSON}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

ARGS=(
    --rmse-filter-m "${RMSE_FILTER_M}"
    --fuel-filter-kg "${FUEL_FILTER_KG}"
    --out-json "${OUT_JSON}"
)
if [[ "${SCAN}" == "1" ]]; then
    ARGS+=(
        --scan
        --rmse-thresholds "${RMSE_THRESHOLDS}"
        --fuel-thresholds-g "${FUEL_THRESHOLDS_G}"
        --thrust-thresholds "${THRUST_THRESHOLDS}"
    )
fi
if [[ -n "${SMARTSAT_ROOT}" ]]; then
    ARGS+=(--smartsat-root "${SMARTSAT_ROOT}")
fi

srun --kill-on-bad-exit=1 python scripts/analyze_ff_initial_data.py "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[ff_init_data] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ff_init_data] JSON written to: ${OUT_JSON}"
echo "===================================================================="
exit "${RC}"
