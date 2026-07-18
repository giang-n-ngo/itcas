#!/usr/bin/env bash
#SBATCH --job-name=casd_server
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/casd_server/server-%j.out
#SBATCH --error=slurm_logs/casd_server/server-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# launch_server.sbatch — run the CASD evaluator server
# (scripts/casd_server/server.py) as a PERSISTENT Slurm job on gpu-large.
#
# IMPORTANT: this is a long-running service job, not a batch job that does
# work and exits. The sbatch job process IS the running uvicorn server; the
# allocation is held for the entire --time window (or until the server
# process dies / you `scancel` it). This is a dev/validation server, NOT a
# forever-on production service:
#   - default --time=12:00:00. Extend per-submission with
#     `sbatch --time=DD-HH:MM:SS ...` (up to qos=batch-short's 5-day
#     MaxWall; use `--qos=batch-long` for up to 10 days if you truly need
#     longer).
#   - when the job's time limit is hit (or it's cancelled/crashes), the
#     server disappears and any itcas job pointed at it will start failing
#     health checks / evaluate() calls. To keep the service up longer than
#     one allocation, just resubmit this same script -- it will overwrite
#     slurm_logs/casd_server/address.txt with the new node:port once the
#     new server is healthy, so clients that re-read that file (or that
#     re-`export CASD_SERVER_URL=...` per run) pick up the new location
#     automatically. There is no automatic requeue/restart wired up here on
#     purpose, to avoid silently resurrecting a server nobody's watching.
#
# What it does:
#   1. cd to repo root, module load Anaconda3, conda activate CONDA_ENV
#      (default 'casd-server' -- see scripts/casd_server/install_env.sbatch)
#   2. launches `python scripts/casd_server/server.py --host 0.0.0.0
#      --port PORT` in the background within this job
#   3. polls http://127.0.0.1:PORT/health from ON the compute node
#      (localhost, so this works regardless of whether the login node can
#      reach the compute node's port -- see the Slurm Operator's job report
#      for the separately-tested login->compute reachability result) until
#      it responds healthy or HEALTH_WAIT_SECS elapses
#   4. once healthy, writes "<hostname>:<port>" to
#      slurm_logs/casd_server/address.txt -- this is how OTHER jobs
#      discover this dynamically-scheduled node/port:
#          export CASD_SERVER_URL=http://$(cat slurm_logs/casd_server/address.txt)
#      A pre-existing address.txt (e.g. left over from a crashed prior run)
#      is renamed aside (.stale-<old_jobid>) rather than blindly overwritten
#      in place, so a mid-write crash can't leave half-written garbage; the
#      file is only ever replaced by a single atomic `mv` after the new
#      server is confirmed healthy.
#   5. `wait`s on the server process in the foreground so the Slurm job
#      stays RUNNING for exactly as long as the server is alive; on normal
#      job termination (TERM from scancel/timeout) a trap removes
#      address.txt so stale-but-still-present address files aren't trusted
#      by new client jobs, and stops the server process.
#
# Real vs mock mode:
#   - Real mode (default): loads the actual vLLM engine + judge models +
#     RealToxicityPrompts dataset. Startup can take several minutes
#     (weight downloads + engine init) -- HEALTH_WAIT_SECS defaults
#     accordingly (see below).
#   - Mock mode: set CASD_MOCK=1 (env, exported before sbatch or via
#     --export) to skip all model/dataset loading and serve cheap synthetic
#     scores. Use this FIRST to validate the sbatch/launch mechanics (job
#     stays RUNNING, address file written correctly, /health responds)
#     before spending real GPU time on model loading. GPU is still
#     requested in mock mode so the allocation mechanics themselves are
#     exercised end to end.
#
# Resource rationale:
#   - partition=gpu-large, gres=gpu:h100:1 : H100/H200 GPUs only (the
#     default 'gpu*' partition is A100-only). Only 1 GPU requested --
#     Qwen2.5-7B-Instruct in bf16 needs ~16-20GB, a small fraction of an
#     H100(80GB)/H200(141GB); do not hog a whole 4- or 8-GPU node for a 7B
#     model. Override the GPU generation with, e.g.,
#     `sbatch --gres=gpu:h200:1 scripts/casd_server/launch_server.sbatch`
#     (CLI --gres overrides this script's #SBATCH default).
#   - qos=batch-short : the 'hpc' account's assoc only grants
#     {batch-long, batch-short, interactive, short}; batch-short's 5-day
#     MaxWall comfortably covers any reasonable dev-server window.
#   - cpus=8, mem=64G : headroom for the HF tokenizers/dataset (loaded into
#     host RAM), the two judge models' CPU-side pre/post-processing, and
#     vLLM's own CPU-side scheduling/tokenization threads -- KV cache and
#     model weights themselves live in GPU memory, not host RAM, so this is
#     comfortably generous rather than tightly sized.
#
# Environment overrides (export before sbatch, or use --export=ALL,VAR=val):
#   CONDA_ENV=casd-server              # env to activate (see install_env.sbatch)
#   PORT=8008                          # server port
#   CASD_MOCK=1                        # mock mode (unset/0 = real mode)
#   CASD_TARGET_MODEL=Qwen/Qwen2.5-7B-Instruct
#   CASD_N_PROMPTS=750
#   HEALTH_WAIT_SECS=                  # override the health-poll timeout;
#                                       # defaults to 120s in mock mode,
#                                       # 2400s (40min) in real mode
#
# Submit (mock-mode mechanics validation -- do this FIRST):
#   sbatch --export=ALL,CASD_MOCK=1 scripts/casd_server/launch_server.sbatch
#
# Submit (real mode, once mock-mode validation passes):
#   sbatch scripts/casd_server/launch_server.sbatch
#
# Once RUNNING and healthy, other jobs discover it via:
#   export CASD_SERVER_URL=http://$(cat slurm_logs/casd_server/address.txt)
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
    echo "[casd-server] WARNING: could not create '${LOG_DIR}' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-casd-server}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
PORT="${PORT:-8008}"
ADDR_FILE="${LOG_DIR}/address.txt"
MOCK_MODE="${CASD_MOCK:-0}"
HOST_NAME="$(hostname)"

echo "===================================================================="
echo "CASD server launch | job ${SLURM_JOB_ID:-local} | host=${HOST_NAME} partition=${SLURM_JOB_PARTITION:-NA}"
echo "conda_env=${CONDA_ENV} port=${PORT} mock=${MOCK_MODE} gpus=${CUDA_VISIBLE_DEVICES:-NA}"
echo "target_model=${CASD_TARGET_MODEL:-<default in server.py>}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- 0. Guard against a stale address file from a crashed prior run --------
# Never blindly overwrite in place; rename any pre-existing file aside so a
# concurrent reader never sees a half-written new file, and so operators can
# see evidence of a prior run if something goes wrong.
if [[ -f "${ADDR_FILE}" ]]; then
    OLD_CONTENTS="$(cat "${ADDR_FILE}" 2>/dev/null || echo unknown)"
    STALE_FILE="${ADDR_FILE}.stale-$(date +%s)"
    mv "${ADDR_FILE}" "${STALE_FILE}"
    echo "[casd-server] found pre-existing address.txt (contents: '${OLD_CONTENTS}'); moved aside to ${STALE_FILE}"
fi

# --- 1. Activate env ---------------------------------------------------------
module purge 2>/dev/null || true
module load "${CONDA_MODULE}" 2>/dev/null || true
command -v conda >/dev/null 2>&1 || { echo "ERROR: 'conda' not on PATH after 'module load ${CONDA_MODULE}'." >&2; exit 3; }
source activate 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV}" || { echo "ERROR: could not activate conda env '${CONDA_ENV}'. Run scripts/casd_server/install_env.sbatch first." >&2; exit 4; }
echo "[casd-server] active python: $(command -v python)  ($(python --version 2>&1))"

# --- 2. Launch the server in the background ---------------------------------
SERVER_ARGS=(--host 0.0.0.0 --port "${PORT}")
if [[ "${MOCK_MODE}" == "1" || "${MOCK_MODE,,}" == "true" ]]; then
    SERVER_ARGS+=(--mock)
    export CASD_MOCK=1
fi

echo "[casd-server] launching: python scripts/casd_server/server.py ${SERVER_ARGS[*]}"
python scripts/casd_server/server.py "${SERVER_ARGS[@]}" &
SERVER_PID=$!
echo "[casd-server] server_pid=${SERVER_PID}"

# --- 3. Cleanup trap: remove address file + stop server on job termination -
cleanup() {
    echo "[casd-server] cleanup: job ending at $(date '+%Y-%m-%d %H:%M:%S'), removing ${ADDR_FILE} (if owned by this job) and stopping server_pid=${SERVER_PID}"
    if [[ -f "${ADDR_FILE}" ]] && grep -q "^${HOST_NAME}:${PORT}$" "${ADDR_FILE}" 2>/dev/null; then
        rm -f "${ADDR_FILE}"
    fi
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# --- 4. Poll localhost health until the server is up (or we time out) ------
if [[ -n "${HEALTH_WAIT_SECS:-}" ]]; then
    WAIT_SECS="${HEALTH_WAIT_SECS}"
elif [[ "${MOCK_MODE}" == "1" || "${MOCK_MODE,,}" == "true" ]]; then
    WAIT_SECS=120
else
    WAIT_SECS=2400   # real mode: vLLM engine init + ~15GB model + 2 judge models + dataset
fi

echo "[casd-server] polling http://127.0.0.1:${PORT}/health (timeout=${WAIT_SECS}s)..."
ELAPSED=0
HEALTHY=0
while (( ELAPSED < WAIT_SECS )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server process (pid=${SERVER_PID}) exited before becoming healthy. Check ${LOG_DIR}/server-${SLURM_JOB_ID:-NA}.err" >&2
        exit 5
    fi
    if curl -s -f -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    sleep 10
    ELAPSED=$(( ELAPSED + 10 ))
    if (( ELAPSED % 60 == 0 )); then
        echo "[casd-server] still waiting for /health ... (${ELAPSED}s / ${WAIT_SECS}s elapsed)"
    fi
done

if [[ "${HEALTHY}" != "1" ]]; then
    echo "ERROR: server did not become healthy within ${WAIT_SECS}s. Leaving process running for inspection; check ${LOG_DIR}/server-${SLURM_JOB_ID:-NA}.err. NOT writing address.txt." >&2
    wait "${SERVER_PID}"
    exit 6
fi

HEALTH_BODY="$(curl -s -m 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo '<unavailable>')"
echo "[casd-server] healthy: ${HEALTH_BODY}"

# --- 5. Write address file atomically (write to tmp, then mv) --------------
TMP_ADDR_FILE="${ADDR_FILE}.tmp-${SLURM_JOB_ID:-$$}"
printf '%s:%s\n' "${HOST_NAME}" "${PORT}" > "${TMP_ADDR_FILE}"
mv "${TMP_ADDR_FILE}" "${ADDR_FILE}"
echo "[casd-server] wrote address file: ${ADDR_FILE} -> $(cat "${ADDR_FILE}")"
echo "[casd-server] other jobs can now: export CASD_SERVER_URL=http://\$(cat ${ADDR_FILE})"

# --- 6. Stay alive for as long as the server process runs -------------------
echo "[casd-server] server RUNNING (job ${SLURM_JOB_ID:-local}). Waiting on server_pid=${SERVER_PID}..."
wait "${SERVER_PID}"
RC=$?
echo "[casd-server] server process exited rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
exit "${RC}"
