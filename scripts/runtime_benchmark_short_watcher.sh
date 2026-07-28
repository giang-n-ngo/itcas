#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# runtime_benchmark_short_watcher.sh — keeps exactly one
# runtime_benchmark_short_lane.sbatch job alive under the 'short' QOS.
#
# The 'short' QOS only allows this account ONE submitted job at a time
# (MaxJobsPerUser=1, MaxSubmitJobsPerUser=1 -- confirmed via
# `sacctmgr show qos short`), so segments cannot be pre-chained with
# --dependency (a dependent job still counts as "submitted" immediately and
# hits the same limit). This script runs as a plain background process on
# the login node (NOT a Slurm job, so it isn't itself subject to that cap):
# it submits one segment, polls until that job leaves the queue, checks
# whether any runtime-benchmark cells are still missing, and if so submits
# the next segment -- repeating until all 165 cells are recorded or
# MAX_ITERS is hit.
#
# Usage: nohup bash scripts/runtime_benchmark_short_watcher.sh \
#          > slurm_logs/rtb_short_watcher.log 2>&1 &
#        disown
# Check progress: tail -f slurm_logs/rtb_short_watcher.log
# Stop early:      kill <pid>   (see slurm_logs/rtb_short_watcher.pid)
# ---------------------------------------------------------------------------
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CELLS_DIR="${CELLS_DIR:-results/runtime_benchmark_cells}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
MAX_ITERS="${MAX_ITERS:-30}"
POLL_SEC="${POLL_SEC:-60}"

echo $$ > slurm_logs/rtb_short_watcher.pid
echo "[watcher] started pid=$$ at $(date '+%Y-%m-%d %H:%M:%S')"

DEFAULT_METHODS="itcas_ndig itcas_seq_ndig random straddle_then_sample_lse10 straddle_then_sample_lse10_batch bes_then_sample_lse10 bes_then_sample_lse10_batch cas_eci cas_eci_batch moc_cas_hard moc_cas_hard_batch"

count_missing() {
    python - <<PYEOF
from itcas.reporting.summary import _synthetic_problems
problems = _synthetic_problems("${PROBLEMS_CONFIG}")
methods = "${DEFAULT_METHODS}".split()
import os
missing = sum(
    1 for p in problems for m in methods
    if not os.path.exists(f"${CELLS_DIR}/{p}__{m}.csv")
)
print(missing)
PYEOF
}

for i in $(seq 1 "${MAX_ITERS}"); do
    missing=$(count_missing)
    echo "[watcher] iter ${i}/${MAX_ITERS}: ${missing} cell(s) still missing at $(date '+%Y-%m-%d %H:%M:%S')"
    if [[ "${missing}" -eq 0 ]]; then
        echo "[watcher] all cells recorded -- submitting aggregate report job."
        AGG_JID=$(sbatch --parsable scripts/runtime_benchmark_aggregate.sbatch 2>>slurm_logs/rtb_short_watcher.log)
        echo "[watcher] aggregate job: ${AGG_JID:-FAILED TO SUBMIT}"
        break
    fi

    JID=""
    while [[ -z "${JID}" ]]; do
        JID=$(sbatch --parsable scripts/runtime_benchmark_short_lane.sbatch 2>>slurm_logs/rtb_short_watcher.log)
        if [[ -z "${JID}" ]]; then
            echo "[watcher] sbatch submit failed (QOS limit still held by a stale job?), retrying in ${POLL_SEC}s"
            sleep "${POLL_SEC}"
        fi
    done
    echo "[watcher] submitted segment job ${JID}"

    # Poll until it leaves the queue entirely.
    while squeue -j "${JID}" -h -o "%T" 2>/dev/null | grep -q .; do
        sleep "${POLL_SEC}"
    done
    STATE=$(sacct -j "${JID}" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    echo "[watcher] segment ${JID} left queue, final state=${STATE:-unknown}"
done

echo "[watcher] exiting at $(date '+%Y-%m-%d %H:%M:%S')"
