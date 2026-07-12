#!/usr/bin/env bash
#SBATCH --job-name=itcas_calib
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_calib-%j.out
#SBATCH --error=slurm_logs/itcas_calib-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# calibrate.sbatch — run threshold (tau) calibration for ONE problem across
# ALL difficulty levels. Wraps the Scientific Coder's entry point
# (python -m itcas.calibrate); no science inside.
#
# The only required input is the PROBLEM name. The calibrator sweeps every
# difficulty level (target joint-feasible fractions) and writes the results to
# the shared thresholds store consumed by the experiments
# (configs/thresholds.json by default, keyed by [problem][percentage]).
#
# Why CPU (no GPU):
#   Calibration is Monte-Carlo sampling + analytic objective evaluation on CPU
#   (no GP fitting), so it runs on the cpu partition — GPUs would sit idle.
#
# Resource rationale:
#   - partition=cpu, qos=cpu : light, CPU-only workload.
#   - cpus=4, mem=4G, time=00:30:00 : 200k-sample sweeps over d<=20 finish in
#     well under a minute; this is a generous upper bound.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                         # conda env to activate
#   CONDA_MODULE=Anaconda3                  # module providing conda
#   PERCENTAGES="0.2 0.1 0.05 0.01"         # difficulty levels (default: all)
#   N_SAMPLES=200000                        # MC samples per calibration
#   SEED=0                                  # RNG seed (reproducibility)
#   OUT=configs/thresholds.json             # thresholds store to update
#
# Submit (problem is the single positional argument):
#   sbatch scripts/calibrate.sbatch sphere2_6d
#   sbatch --export=ALL,CONDA_ENV=itcas scripts/calibrate.sbatch dtlz2_6d
#
# Valid problems: sphere2_6d, rosenbrock_sphere_6d, multimodal_trap_20d,
#   dtlz2_6d, two_circles_2d, contextual_circles_3d.
# ---------------------------------------------------------------------------
set -euo pipefail

PROBLEM="${1:-${PROBLEM:-}}"
if [[ -z "${PROBLEM}" ]]; then
    echo "ERROR: no PROBLEM given. Usage: sbatch scripts/calibrate.sbatch <problem>" >&2
    exit 2
fi

# Slurm copies the batch script to a spool dir, so BASH_SOURCE does NOT point
# at the repo. Use SLURM_SUBMIT_DIR (the dir sbatch was invoked from) as the
# repo root; fall back to BASH_SOURCE for direct (non-sbatch) execution.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs 2>/dev/null || \
    echo "[calibrate] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
N_SAMPLES="${N_SAMPLES:-200000}"
SEED="${SEED:-0}"
OUT="${OUT:-configs/thresholds.json}"
# Default to ALL difficulty levels; override with PERCENTAGES="...".
read -r -a PCTS <<< "${PERCENTAGES:-0.2 0.1 0.05 0.01}"

echo "===================================================================="
echo "ITCAS threshold calibration | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "problem=${PROBLEM} percentages=${PCTS[*]} n_samples=${N_SAMPLES} seed=${SEED}"
echo "out=${OUT} conda_env=${CONDA_ENV}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment (cluster pattern; falls back to system Python) ----
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[calibrate] WARNING: could not 'conda activate ${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
# Isolate from ~/.local user-site so the conda env's packages are used.
export PYTHONNOUSERSITE=1

# Keep BLAS threads within our CPU allocation.
THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"

# --- Run calibration for the single problem, all difficulty levels ---------
srun --kill-on-bad-exit=1 python -m itcas.calibrate \
    --problems    "${PROBLEM}" \
    --percentages "${PCTS[@]}" \
    --n-samples   "${N_SAMPLES}" \
    --seed        "${SEED}" \
    --out         "${OUT}"
RC=$?

echo "===================================================================="
echo "[calibrate] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[calibrate] thresholds store: ${OUT}"
echo "===================================================================="
exit "${RC}"
