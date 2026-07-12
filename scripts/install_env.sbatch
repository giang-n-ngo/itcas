#!/usr/bin/env bash
#SBATCH --job-name=itcas_install
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/itcas_install-%j.out
#SBATCH --error=slurm_logs/itcas_install-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# install_env.sbatch — build (or update) the ITCAS conda environment on the
# cluster, then verify it. One-shot batch job (not an array).
#
# What it does (non-interactive, idempotent):
#   1. module load Anaconda3
#   2. create conda env CONDA_ENV (default 'itcas', Python 3.10) if missing
#   3. (optional) install a CUDA-specific torch wheel from TORCH_INDEX_URL
#   4. pip install -r requirements.txt
#   5. verify imports + report torch / CUDA availability + run smoke tests
#
# Why a GPU is requested:
#   We allocate 1 GPU (--gpus=1) so the verification step confirms the *exact*
#   environment the experiments will use can see CUDA (torch.cuda.is_available()).
#   The pip install itself does not need a GPU; set GPUS='' to skip it (override
#   via submit-time --gpus or by editing this directive) if GPU queue wait is long.
#
# Resource rationale:
#   - partition=gpu, qos=batch-short : matches the experiment jobs' QoS.
#   - cpus=4, mem=8G, time=01:30:00 : ample for downloading/compiling wheels
#     (torch is the large one); installs typically finish in minutes.
#
# Environment overrides (export before sbatch, or use --export):
#   CONDA_ENV=itcas                  # env name to create/update
#   PYVER=3.10                       # Python version for a fresh env
#   CONDA_MODULE=Anaconda3           # module providing conda
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#                                    # if set, install torch from this index 1st
#   REQ_FILE=requirements.txt        # requirements file to install
#
# Submit:
#   sbatch scripts/install_env.sbatch
#   # with a specific CUDA torch build:
#   sbatch --export=ALL,CONDA_ENV=itcas,TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#          scripts/install_env.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

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
    echo "[install] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-itcas}"
PYVER="${PYVER:-3.10}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
# The cluster's NVIDIA driver is CUDA 12.7; the default PyPI torch wheel ships
# CUDA 13.0 and will NOT see the GPU. Default to the cu124 build (compatible
# with driver >= 12.4, so fine on a 12.7 driver). Override with TORCH_INDEX_URL.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "===================================================================="
echo "ITCAS env install | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "conda_env=${CONDA_ENV} python=${PYVER} req=${REQ_FILE}"
echo "torch_index_url=${TORCH_INDEX_URL}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-NA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

[[ -f "${REQ_FILE}" ]] || { echo "ERROR: '${REQ_FILE}' not found in ${REPO_ROOT}." >&2; exit 2; }

# --- 1. Load conda ---------------------------------------------------------
module purge 2>/dev/null || true
module load "${CONDA_MODULE}" 2>/dev/null || true
command -v conda >/dev/null 2>&1 || { echo "ERROR: 'conda' not on PATH after 'module load ${CONDA_MODULE}'." >&2; exit 3; }
echo "[install] using conda: $(command -v conda)  ($(conda --version 2>/dev/null))"

# --- 2. Create env if missing ---------------------------------------------
if conda env list | awk '{print $1}' | grep -qxF "${CONDA_ENV}"; then
    echo "[install] conda env '${CONDA_ENV}' already exists -> updating in place."
else
    echo "[install] creating conda env '${CONDA_ENV}' (python=${PYVER})..."
    conda create -y -n "${CONDA_ENV}" "python=${PYVER}" pip
fi

# Activate (Anaconda3-style: 'source activate' enables 'conda activate').
source activate 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV}"
# Isolate the env from ~/.local user site-packages. Without this, a fresh env
# sees user-site packages (e.g. a CUDA-13 torch) on sys.path, pip treats the
# requirement as already satisfied, and nothing lands in the env -> the wrong
# (driver-incompatible) torch gets imported at runtime.
export PYTHONNOUSERSITE=1
export PIP_USER=0
echo "[install] active python: $(command -v python)  ($(python --version 2>&1))"
echo "[install] site dirs:"; python -c "import site,sys; print('  prefix=',sys.prefix); print('  ENABLE_USER_SITE=',site.ENABLE_USER_SITE)"

python -m pip install --upgrade pip

# --- 3. Install the CUDA-matched torch wheel FIRST -------------------------
# Done before requirements so the env holds a driver-compatible torch; the
# subsequent `-r ${REQ_FILE}` then sees torch satisfied and won't override it.
echo "[install] installing torch from ${TORCH_INDEX_URL} ..."
python -m pip install --force-reinstall torch --index-url "${TORCH_INDEX_URL}"

# --- 4. Install project requirements --------------------------------------
echo "[install] pip install -r ${REQ_FILE} ..."
python -m pip install -r "${REQ_FILE}"

# --- 5. Verify -------------------------------------------------------------
echo "[install] verifying environment..."
python - <<'PY'
import importlib, os, sys
mods = ["torch", "gpytorch", "botorch", "numpy", "matplotlib", "yaml"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"  {m:11s} {getattr(mod, '__version__', '?')}")
import torch
# Confirm torch is imported from the conda env, not ~/.local user-site.
print(f"  torch loaded from        = {os.path.dirname(torch.__file__)}")
if torch.__file__.startswith(os.path.expanduser('~/.local')):
    sys.exit("ERROR: torch is imported from ~/.local, not the conda env. "
             "User-site leak — aborting.")
avail = torch.cuda.is_available()
print(f"  torch.cuda.is_available() = {avail}")
if avail:
    print(f"  CUDA device              = {torch.cuda.get_device_name(0)}")
    print(f"  torch CUDA build         = {torch.version.cuda}")
# If Slurm allocated a GPU but torch can't use it, fail so we don't silently
# 'complete' with a broken (CPU-only / driver-mismatched) environment.
gpu_alloc = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
if gpu_alloc and not avail:
    sys.exit("ERROR: a GPU was allocated but torch.cuda.is_available() is False "
             "(driver/CUDA-build mismatch). Try a different TORCH_INDEX_URL.")
PY

echo "[install] running smoke unit tests..."
python -m tests.test_smoke

echo "===================================================================="
echo "[install] DONE: conda env '${CONDA_ENV}' ready."
echo "[install] use it in sweeps with:  CONDA_ENV=${CONDA_ENV} scripts/submit.sh scripts/jobs.json"
echo "finish=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
