#!/usr/bin/env bash
#SBATCH --job-name=casd_ablate_scoring
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/casd_server/ablate_scoring-%j.out
#SBATCH --error=slurm_logs/casd_server/ablate_scoring-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# ablate_scoring.sbatch — one-shot diagnostic job.
#
# Runs scripts/casd_server/ablate_scoring.py DIRECTLY, in-process, on a GPU
# node. Same pattern as inspect_generations.sbatch: ablate_scoring.py
# imports server.py as a module and drives server.load_state(...) /
# server.STATE.llm.generate(...) / scoring directly, without ever calling
# uvicorn.run(). No FastAPI process, no address.txt, no /health polling.
# The job runs the script to completion and exits -- no `wait`-on-a-
# background-server loop like launch_server.sbatch.
#
# What it does: follow-up to inspect_generations.py's findings. Runs 5
# scoring treatments (baseline, chat_template, longer_generation,
# multilabel_score, worst_of_5) over 150 selected rows from
# results/casd_llm/calibration/samples.csv (real vLLM engine + toxic-bert
# judge, same 750-prompt pool as the calibration run), and writes:
#   - results/casd_llm/calibration/ablation_scores.csv
#   - results/casd_llm/calibration/ablation_summary.csv
#   - results/casd_llm/calibration/ablation_histograms.png
# See the script's own module docstring for the full "why".
#
# Resource rationale (same as inspect_generations.sbatch, since this loads
# the same models):
#   - partition=gpu-large, gres=gpu:h100:1 : H100 GPU, matches the server job
#     (Qwen2.5-7B-Instruct in bf16 needs ~16-20GB of an 80GB H100).
#   - qos=batch-short : 5-day MaxWall, comfortably covers this job.
#   - cpus=8, mem=64G : same headroom rationale as inspect_generations.sbatch
#     (HF tokenizers/dataset in host RAM, judge model CPU pre/post-processing,
#     vLLM CPU-side scheduling/tokenization threads).
#   - time=01:30:00 : bumped from inspect_generations.sbatch's 1h because
#     this job does more work (150 rows x 5 treatments, one of which --
#     worst_of_5 -- generates 5 completions/row, so ~1200 total generations
#     plus judge-model scoring passes, vs. inspect_generations.py's 25
#     single-shot generations). Per the calibration run's measured
#     throughput (1000 generations in ~23s once the model was loaded), the
#     generation+scoring work itself should still be fast; the dominant
#     cost historically has been queue wait + model load (~1-2 min), not
#     generation. 90 minutes is generous headroom for queue/loading
#     variance; expected actual runtime is well under that.
#
# Environment overrides (export before sbatch, or use --export=ALL,VAR=val):
#   CONDA_ENV=casd-server                  # env to activate
#   CASD_TARGET_MODEL=Qwen/Qwen2.5-7B-Instruct
#   CASD_N_PROMPTS=750
#   ABLATE_ARGS=""                         # extra CLI args appended to the
#                                            # python invocation, e.g.
#                                            # "--n-random 100"
#
# Submit:
#   sbatch scripts/casd_server/ablate_scoring.sbatch
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
    echo "[ablate-scoring] WARNING: could not create '${LOG_DIR}' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-casd-server}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
HOST_NAME="$(hostname)"

echo "===================================================================="
echo "CASD ablate_scoring | job ${SLURM_JOB_ID:-local} | host=${HOST_NAME} partition=${SLURM_JOB_PARTITION:-NA}"
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
echo "[ablate-scoring] active python: $(command -v python)  ($(python --version 2>&1))"

# --- 2. Run the diagnostic script directly (no server, no uvicorn) ---------
ABLATE_ARGS="${ABLATE_ARGS:-}"
echo "[ablate-scoring] launching: python scripts/casd_server/ablate_scoring.py ${ABLATE_ARGS}"
# shellcheck disable=SC2086
python scripts/casd_server/ablate_scoring.py ${ABLATE_ARGS}
RC=$?
echo "[ablate-scoring] script exited rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
exit "${RC}"
