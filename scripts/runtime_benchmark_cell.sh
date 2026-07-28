#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# runtime_benchmark_cell.sh — time ONE (method, problem) cell, seed=0 only,
# and persist the result to a dedicated CSV file (NOT the Slurm log).
#
# Writes results/runtime_benchmark_cells/<problem>__<method>.csv containing
# a single line "method,problem,seconds" on success. Skips (exit 0
# immediately) if that file already exists, so callers can freely re-launch
# without redoing work -- this is what makes runtime_benchmark_problem.sbatch
# safe to requeue/retry.
#
# Usage: runtime_benchmark_cell.sh <method> <problem>
# Env:   CELLS_DIR (default results/runtime_benchmark_cells)
#        EXPERIMENTS_CONFIG, PROBLEMS_CONFIG, DEVICE (default cuda)
# ---------------------------------------------------------------------------
set -u -o pipefail

METHOD="${1:?usage: runtime_benchmark_cell.sh <method> <problem>}"
PROBLEM="${2:?usage: runtime_benchmark_cell.sh <method> <problem>}"

CELLS_DIR="${CELLS_DIR:-results/runtime_benchmark_cells}"
EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG:-configs/experiments.json}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
DEVICE="${DEVICE:-cuda}"
SEED=0   # runtime benchmark only ever needs ONE seed to record wall-clock cost

mkdir -p "${CELLS_DIR}"
OUT_CSV="${CELLS_DIR}/${PROBLEM}__${METHOD}.csv"

if [[ -f "${OUT_CSV}" ]]; then
    echo "[cell] SKIP ${METHOD} x ${PROBLEM} (already recorded: ${OUT_CSV})"
    exit 0
fi

TMP_MD="$(mktemp)"
STDOUT_LOG="$(mktemp)"
python -m itcas.reporting.runtime_benchmark \
    --methods "${METHOD}" \
    --problems "${PROBLEM}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --experiments-config "${EXPERIMENTS_CONFIG}" \
    --problems-config "${PROBLEMS_CONFIG}" \
    --output "${TMP_MD}" \
    > "${STDOUT_LOG}" 2>&1
RC=$?

if [[ ${RC} -ne 0 ]]; then
    echo "[cell] FAILED ${METHOD} x ${PROBLEM} rc=${RC}; tail of output:" >&2
    tail -n 20 "${STDOUT_LOG}" >&2
    rm -f "${TMP_MD}" "${STDOUT_LOG}"
    exit "${RC}"
fi

# Parse "[1/1] <method> x <problem>: X.XXs" out of stdout -- this is only
# used to extract the number; the durable record is OUT_CSV, not this log.
SECS=$(grep -oE '\[1/1\][^:]+: [0-9.]+s' "${STDOUT_LOG}" | grep -oE '[0-9.]+s$' | tr -d 's')
rm -f "${TMP_MD}" "${STDOUT_LOG}"

if [[ -z "${SECS}" ]]; then
    echo "[cell] FAILED ${METHOD} x ${PROBLEM}: could not parse elapsed seconds from output" >&2
    exit 1
fi

echo "${METHOD},${PROBLEM},${SECS}" > "${OUT_CSV}"
echo "[cell] DONE ${METHOD} x ${PROBLEM}: ${SECS}s -> ${OUT_CSV}"
