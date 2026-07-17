#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start_broker.sh -- launch the FF-sim dispatch broker as a background
# process on the LOGIN node (NOT a Slurm job -- see broker.py docstring for
# why: it only shells out to sbatch/squeue/scancel and does file I/O, so
# running it under Slurm would waste one of the 5 backfill slots this whole
# mechanism exists to relieve).
#
# The broker is pure stdlib Python (no torch/gpytorch/conda env needed), so
# any python3 >= 3.9 on the login node works; no environment activation
# required.
#
# Usage:
#   scripts/ff_sim_broker/start_broker.sh                 # use default queue dir
#   FF_SIM_BROKER_QUEUE_DIR=/path/to/test/queue \
#     scripts/ff_sim_broker/start_broker.sh                # point at a test queue
#
# Env vars (all optional, see broker.py docstring for defaults):
#   SMARTSAT_ROOT, FF_SIM_BROKER_QUEUE_DIR, FF_SIM_MAX_BATCH,
#   FF_SIM_BROKER_POLL_WINDOW, FF_SIM_BROKER_SCAN_INTERVAL,
#   FF_SIM_BROKER_JOB_TIMEOUT, FF_SIM_CONDA_ENV
#
# Stop with scripts/ff_sim_broker/stop_broker.sh.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SMARTSAT_ROOT="${SMARTSAT_ROOT:-$(cd "${REPO_ROOT}/../SmartSat" && pwd)}"
export SMARTSAT_ROOT
QUEUE_DIR="${FF_SIM_BROKER_QUEUE_DIR:-${SMARTSAT_ROOT}/ff_sim_work/broker}"
export FF_SIM_BROKER_QUEUE_DIR="${QUEUE_DIR}"

mkdir -p "${QUEUE_DIR}"
PID_FILE="${QUEUE_DIR}/broker.pid"
LOG_FILE="${QUEUE_DIR}/broker.out.log"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[start_broker] already running: pid=$(cat "${PID_FILE}") queue_dir=${QUEUE_DIR}"
    exit 0
fi

PYTHON_BIN="${FF_SIM_BROKER_PYTHON:-python3}"
nohup "${PYTHON_BIN}" "${SCRIPT_DIR}/broker.py" >> "${LOG_FILE}" 2>&1 &
BROKER_PID=$!
echo "${BROKER_PID}" > "${PID_FILE}"
disown "${BROKER_PID}" 2>/dev/null || true

echo "[start_broker] started pid=${BROKER_PID} queue_dir=${QUEUE_DIR} log=${LOG_FILE}"
