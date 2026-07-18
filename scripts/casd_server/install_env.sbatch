#!/usr/bin/env bash
#SBATCH --job-name=casd_server_install
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/casd_server/install-%j.out
#SBATCH --error=slurm_logs/casd_server/install-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# install_env.sbatch (casd_server) — build (or update) the CASD evaluator
# server's conda environment on the cluster, then verify it. One-shot batch
# job (not an array). Mirrors scripts/install_env.sbatch's structure/doc
# style, but targets requirements-casd-server.txt into a NEW, separate
# env (default 'casd-server') instead of the 'itcas' env -- see
# requirements-casd-server.txt's header for why the two envs must stay
# separate (conflicting torch/transformers pinning between the BoTorch/
# GPyTorch stack and vLLM).
#
# What it does (non-interactive, idempotent):
#   1. module load Anaconda3
#   2. ALWAYS recreate conda env CONDA_ENV from scratch (default 'casd-server',
#      Python 3.10) -- remove it first if it already exists, rather than
#      updating in place. See the "Always recreate the env from scratch" note
#      at step 2 below for the incident (orphaned flashinfer/tvm_ffi/
#      torch_c_dlpack_ext packages surviving a prior broken install and
#      crashing real-mode server launch job 113218) that motivated this.
#   3. pip install torch==2.6.0/torchvision==0.21.0/torchaudio==2.6.0 from
#      the cu124 PyTorch index, THEN pip install -r requirements-casd-server.txt
#      (mirrors scripts/install_env.sbatch's two-step pattern for the itcas
#      env). torch is pinned -- indirectly via vllm==0.8.5.post1's own exact
#      pin, and directly here -- because an earlier version of this file left
#      both vllm and torch unpinned, which resolved to a CUDA 13.0 torch
#      build this cluster's driver (CUDA 12.7) cannot run; see
#      requirements-casd-server.txt's header ("GPU note") for the full
#      incident writeup and the researched fix.
#   4. verify imports (vllm, fastapi, uvicorn, transformers, datasets,
#      torch) + report torch / CUDA availability
#
# What it deliberately does NOT do:
#   - It does NOT load any real model weights (Qwen2.5-7B-Instruct,
#     toxic-bert, the reward model) or the RealToxicityPrompts dataset --
#     that happens once, at server startup, in launch_server.sbatch. This
#     keeps the install job fast/cheap and independent of HF Hub
#     availability/rate limits.
#
# Why a GPU is requested (mirrors scripts/install_env.sbatch's rationale):
#   The pip install itself does not strictly need a GPU, but this job
#   allocates 1 GPU on gpu-large anyway so the verification step confirms
#   the *exact* environment the server will run in can see an H100/H200
#   CUDA device (torch.cuda.is_available()) via the vLLM-selected torch
#   build, catching a driver/CUDA-build mismatch here (cheap, ~1-2h) rather
#   than after a slow real-mode server launch (tens of minutes of model
#   downloads). Only 1 GPU is requested (not a whole node) since this is a
#   verification step, not the actual serving workload.
#
# Resource rationale:
#   - partition=gpu-large, gres=gpu:h100:1 : H100/H200 GPUs only (the
#     default 'gpu*' partition is A100-only -- NOT what the CASD server
#     targets; see scripts/m_submit_jobs.sbatch's comment on this same
#     partition split). Only 1 GPU, matching the server's actual footprint
#     (a 7B model in bf16 needs ~16-20GB, far under a full H100/H200).
#   - qos=batch-short : the 'hpc' account's assoc only grants
#     {batch-long, batch-short, interactive, short}; batch-short (5-day
#     MaxWall) comfortably covers this <2h job and matches
#     scripts/install_env.sbatch's convention for install jobs.
#   - cpus=8, mem=32G, time=02:00:00 : vLLM's install pulls a large CUDA
#     torch wheel + compiled CUDA extensions; give it more headroom than
#     the itcas install job (which uses cpus=4/mem=8G/time=1.5h) since the
#     vLLM wheel + its transitive deps are noticeably larger/slower to
#     resolve and unpack.
#
# Environment overrides (export before sbatch, or use --export):
#   CONDA_ENV=casd-server             # env name to create/update
#   PYVER=3.10                        # Python version for a fresh env
#   CONDA_MODULE=Anaconda3            # module providing conda
#   REQ_FILE=requirements-casd-server.txt   # requirements file to install
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#                                      # index to pre-install torch/vision/audio
#                                      # from (must match vllm's pinned torch
#                                      # version -- see requirements-casd-server.txt)
#
# Submit:
#   sbatch scripts/casd_server/install_env.sbatch
#   # with a different env name:
#   sbatch --export=ALL,CONDA_ENV=casd-server-dev scripts/casd_server/install_env.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

# Slurm copies the batch script to a spool dir, so BASH_SOURCE does NOT point
# at the repo. Use SLURM_SUBMIT_DIR (the dir sbatch was invoked from) as the
# repo root; fall back to BASH_SOURCE for direct (non-sbatch) execution.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs/casd_server 2>/dev/null || \
    echo "[casd-install] WARNING: could not create '${REPO_ROOT}/slurm_logs/casd_server' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-casd-server}"
PYVER="${PYVER:-3.10}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
REQ_FILE="${REQ_FILE:-requirements-casd-server.txt}"
# The cluster's NVIDIA driver reports CUDA 12.7. vllm==0.8.5.post1 (pinned in
# REQ_FILE) hard-pins torch==2.6.0, the last torch release with a cu124 wheel
# (2.7.0+ only ships cu126/cu128/cu129/cu130). Default to that index; see
# requirements-casd-server.txt's header for the full rationale/incident.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "===================================================================="
echo "CASD server env install | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "conda_env=${CONDA_ENV} python=${PYVER} req=${REQ_FILE}"
echo "torch_index_url=${TORCH_INDEX_URL}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-NA} partition=${SLURM_JOB_PARTITION:-NA}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

[[ -f "${REQ_FILE}" ]] || { echo "ERROR: '${REQ_FILE}' not found in ${REPO_ROOT}." >&2; exit 2; }

# --- 1. Load conda ---------------------------------------------------------
module purge 2>/dev/null || true
module load "${CONDA_MODULE}" 2>/dev/null || true
command -v conda >/dev/null 2>&1 || { echo "ERROR: 'conda' not on PATH after 'module load ${CONDA_MODULE}'." >&2; exit 3; }
echo "[casd-install] using conda: $(command -v conda)  ($(conda --version 2>/dev/null))"

# --- 2. Always recreate the env from scratch --------------------------------
# CORRECTED 2026-07-17: this used to update an existing env "in place" (skip
# `conda create` if the env already existed, then just re-run pip install).
# That silently broke a real launch: this env's very first build (job 113176)
# resolved unpinned vllm>=0.6.0 to vllm==0.25.1, which pulled in flashinfer /
# tvm_ffi / torch_c_dlpack_ext -- optional packages vLLM auto-imports if
# present, with a compiled .so (torch_c_dlpack_ext) linked against that run's
# torch==2.11.0+cu130 ABI. Once requirements-casd-server.txt was fixed to pin
# vllm==0.8.5.post1/torch==2.6.0 (see that file's header), re-running this
# script with "update in place" correctly reinstalled torch/vllm/transformers
# -- but vllm==0.8.5.post1 does NOT declare flashinfer/tvm_ffi/
# torch_c_dlpack_ext as hard dependencies, so pip had no reason to touch them;
# they were silently left behind from job 113176, still linked against the
# old torch 2.11 ABI. The result: the install job (113195, 113217) verified
# clean, but the real-mode server launch (113218) crashed at engine-init with
# `OSError: ... torch_c_dlpack_ext...: undefined symbol:
# _ZNK3c106Device3strB5cxx11Ev` the moment vLLM's sampler auto-imported the
# stale flashinfer build. Confirmed via file mtimes: flashinfer/tvm_ffi/
# torch_c_dlpack_ext all date to job 113176's original (pre-fix) install, not
# either subsequent "fixed" install.
#
# Fix: always remove and recreate the env from scratch here, so every run of
# this script produces an environment containing EXACTLY (and only) what
# requirements-casd-server.txt currently resolves to -- no cross-run leftover
# packages from a since-changed dependency set can survive. This costs a
# fresh conda-create + full pip install every run (a few minutes) in exchange
# for reproducibility; given this job already targets <2h and installs are
# the exception rather than the common case, that trade is worth it.
if conda env list | awk '{print $1}' | grep -qxF "${CONDA_ENV}"; then
    echo "[casd-install] conda env '${CONDA_ENV}' already exists -> removing it first (see 'Always recreate' note above: incremental updates can leave orphaned packages linked against a stale torch ABI)."
    conda env remove -y -n "${CONDA_ENV}"
fi
echo "[casd-install] creating conda env '${CONDA_ENV}' (python=${PYVER})..."
conda create -y -n "${CONDA_ENV}" "python=${PYVER}" pip

# Activate (Anaconda3-style: 'source activate' enables 'conda activate').
source activate 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV}"
# Isolate the env from ~/.local user site-packages (same rationale as
# scripts/install_env.sbatch): without this, a fresh env can see user-site
# packages on sys.path, pip treats the requirement as already satisfied, and
# the wrong (possibly driver-incompatible) torch build gets imported.
export PYTHONNOUSERSITE=1
export PIP_USER=0
echo "[casd-install] active python: $(command -v python)  ($(python --version 2>&1))"

python -m pip install --upgrade pip

# --- 3. Install the CUDA-matched torch wheel FIRST, then requirements -------
# Mirrors scripts/install_env.sbatch's two-step pattern for the itcas env.
# Pre-installing torch/vision/audio from TORCH_INDEX_URL ensures pip resolves
# the +cu124-tagged build (PEP 440 local-version precedence prefers it once
# present) before `pip install -r ${REQ_FILE}` pulls in vllm==0.8.5.post1,
# which itself hard-pins torch==2.6.0/torchvision==0.21.0/torchaudio==2.6.0.
# See requirements-casd-server.txt's header ("GPU note") for why torch/vllm
# must be pinned this way -- a previous unpinned `pip install -r ${REQ_FILE}`
# alone resolved to a CUDA 13.0 torch build incompatible with this cluster's
# driver (CUDA 12.7).
echo "[casd-install] installing torch==2.6.0/torchvision==0.21.0/torchaudio==2.6.0 from ${TORCH_INDEX_URL} ..."
python -m pip install --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url "${TORCH_INDEX_URL}"

echo "[casd-install] pip install -r ${REQ_FILE} ..."
python -m pip install -r "${REQ_FILE}"

# --- 4. Verify ---------------------------------------------------------------
echo "[casd-install] verifying environment..."
python - <<'PY'
import importlib, os, sys
mods = ["torch", "vllm", "fastapi", "uvicorn", "transformers", "datasets", "accelerate"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"  {m:13s} {getattr(mod, '__version__', '?')}")
import torch
print(f"  torch loaded from        = {os.path.dirname(torch.__file__)}")
if torch.__file__.startswith(os.path.expanduser('~/.local')):
    sys.exit("ERROR: torch is imported from ~/.local, not the conda env. "
             "User-site leak -- aborting.")
avail = torch.cuda.is_available()
print(f"  torch.cuda.is_available() = {avail}")
if avail:
    print(f"  CUDA device              = {torch.cuda.get_device_name(0)}")
    print(f"  torch CUDA build         = {torch.version.cuda}")
gpu_alloc = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
if gpu_alloc and not avail:
    sys.exit("ERROR: a GPU was allocated but torch.cuda.is_available() is False "
             "(driver/CUDA-build mismatch). vLLM's resolved torch build may be "
             "incompatible with this cluster's driver -- report to the "
             "Scientific Coder, do not patch requirements-casd-server.txt here.")
PY

echo "===================================================================="
echo "[casd-install] DONE: conda env '${CONDA_ENV}' ready."
echo "[casd-install] launch the server with:"
echo "    CONDA_ENV=${CONDA_ENV} sbatch scripts/casd_server/launch_server.sbatch"
echo "finish=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
