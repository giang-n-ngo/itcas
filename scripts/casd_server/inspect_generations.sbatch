#!/usr/bin/env bash
#SBATCH --job-name=casd_inspect_gen
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/casd_server/inspect_gen-%j.out
#SBATCH --error=slurm_logs/casd_server/inspect_gen-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# inspect_generations.sbatch — one-shot diagnostic job.
#
# Runs scripts/casd_server/inspect_generations.py DIRECTLY, in-process, on a
# GPU node. This is NOT the persistent HTTP server
# (scripts/casd_server/launch_server.sbatch) -- inspect_generations.py
# imports server.py as a module and drives server.load_state(...) /
# server.STATE.llm.generate(...) / server._score_toxicity_batch(...)
# directly, without ever calling uvicorn.run(). No FastAPI process, no
# address.txt, no /health polling. The job runs the script to completion and
# exits -- no `wait`-on-a-background-server loop like launch_server.sbatch.
#
# What it does: regenerates completions for 25 rows from
# results/casd_llm/calibration/samples.csv (10 lowest-f1 + 15
# high-toxicity/high-f1) using the real vLLM engine + toxic-bert judge + the
# same 750-prompt pool (DEFAULT_POOL_LOAD_SEED=0, deterministic) the
# calibration run used, and writes prompt+generation+scores side by side to
# results/casd_llm/calibration/generation_inspection.md. See the script's
# own module docstring for the full "why".
#
# Resource rationale (same as launch_server.sbatch, since this loads the
# same models):
#   - partition=gpu-large, gres=gpu:h100:1 : H100 GPU, matches the server job
#     (Qwen2.5-7B-Instruct in bf16 needs ~16-20GB of an 80GB H100).
#   - qos=batch-short : 5-day MaxWall, comfortably covers this <1h job.
#   - cpus=8, mem=64G : same headroom rationale as launch_server.sbatch
#     (HF tokenizers/dataset in host RAM, judge model CPU pre/post-processing,
#     vLLM CPU-side scheduling/tokenization threads).
#   - time=01:00:00 : model loading (vLLM engine + toxic-bert) took a few
#     minutes in prior server-launch runs; the 25 generations themselves are
#     single-item (not batched) generate() calls and should be fast. 1 hour
#     is generous headroom for queue/loading variance; expected actual
#     runtime is well under that.
#
# Environment overrides (export before sbatch, or use --export=ALL,VAR=val):
#   CONDA_ENV=casd-server                  # env to activate
#   CASD_TARGET_MODEL=Qwen/Qwen2.5-7B-Instruct
#   CASD_N_PROMPTS=750
#   INSPECT_ARGS=""                        # extra CLI args appended to the
#                                            # python invocation, e.g.
#                                            # "--n-lowest-f1 5"
#
# Submit:
#   sbatch scripts/casd_server/inspect_generations.sbatch
# ---------------------------------------------------------------------------
set -u -o pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
LOG_DIR="${REPO_ROOT}/slurm_logs/casd_server"
mkdir -p "${LOG_DIR}" 2>/dev/null || \
    echo "[inspect-gen] WARNING: could not create '${LOG_DIR}' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-casd-server}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
HOST_NAME="$(hostname)"

echo "===================================================================="
echo "CASD inspect_generations | job ${SLURM_JOB_ID:-local} | host=${HOST_NAME} partition=${SLURM_JOB_PARTITION:-NA}"
echo "conda_env=${CONDA_ENV} gpus=${CUDA_VISIBLE_DEVICES:-NA}"
echo "target_model=${CASD_TARGET_MODEL:-<default in server.py>}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- 1. Activate env ---------------------------------------------------------
module purge 2>/dev/null || true
module load "${CONDA_MODULE}" 2>/dev/null || true
command -v conda >/dev/null 2>&1 || { echo "ERROR: 'conda' not on PATH after 'module load ${CONDA_MODULE}'." >&2; exit 3; }
source activate 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV}" || { echo "ERROR: could not activate conda env '${CONDA_ENV}'. Run scripts/casd_server/install_env.sbatch first." >&2; exit 4; }
echo "[inspect-gen] active python: $(command -v python)  ($(python --version 2>&1))"

# --- 2. Run the diagnostic script directly (no server, no uvicorn) ---------
INSPECT_ARGS="${INSPECT_ARGS:-}"
echo "[inspect-gen] launching: python scripts/casd_server/inspect_generations.py ${INSPECT_ARGS}"
# shellcheck disable=SC2086
python scripts/casd_server/inspect_generations.py ${INSPECT_ARGS}
RC=$?
echo "[inspect-gen] script exited rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
exit "${RC}"
