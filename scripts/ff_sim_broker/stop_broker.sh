#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stop_broker.sh -- stop the FF-sim dispatch broker started by
# start_broker.sh. Sends SIGTERM to the PID recorded in broker.pid; the
# daemon's main loop will finish its current scan pass and exit (in-flight
# merged Slurm jobs are left running and will still resolve their requests'
# result_NNNN.json / DONE files the next time the broker is started, since
# nothing about a claimed-but-not-yet-finished batch depends on the daemon
# staying alive except writing the DONE marker back -- restart the broker
# before those callers' own client-side timeouts expire).
#
# Usage:
#   scripts/ff_sim_broker/stop_broker.sh
#   FF_SIM_BROKER_QUEUE_DIR=/path/to/test/queue scripts/ff_sim_broker/stop_broker.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMARTSAT_ROOT="${SMARTSAT_ROOT:-$(cd "${REPO_ROOT}/../SmartSat" && pwd)}"
QUEUE_DIR="${FF_SIM_BROKER_QUEUE_DIR:-${SMARTSAT_ROOT}/ff_sim_work/broker}"
PID_FILE="${QUEUE_DIR}/broker.pid"

if [[ ! -f "${PID_FILE}" ]]; then
    echo "[stop_broker] no pid file at ${PID_FILE}; nothing to stop."
    exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}"
    echo "[stop_broker] sent SIGTERM to pid=${PID}"
else
    echo "[stop_broker] pid=${PID} not running."
fi
rm -f "${PID_FILE}"
