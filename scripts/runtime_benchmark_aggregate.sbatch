#!/usr/bin/env bash
#SBATCH --job-name=itcas_rtb_aggregate
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/itcas_rtb_aggregate-%j.out
#SBATCH --error=slurm_logs/itcas_rtb_aggregate-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# runtime_benchmark_aggregate.sbatch — builds the final results/runtime_benchmark.md
# from the per-cell CSVs under CELLS_DIR (results/runtime_benchmark_cells/
# <problem>__<method>.csv, one "method,problem,seconds" line each -- written
# by runtime_benchmark_cell.sh, NEVER read from any Slurm .out log). Reuses
# the existing build_report()/build_markdown_table() from
# itcas.reporting.runtime_benchmark (imported, not modified) so the final
# table is byte-for-byte the same format the original monolithic job would
# have produced.
#
# Submitted (with --dependency=afterok:<per-problem job ids>) by
# runtime_benchmark_dispatch.sbatch.
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
CELLS_DIR="${CELLS_DIR:-results/runtime_benchmark_cells}"
OUTPUT="${OUTPUT:-results/runtime_benchmark.md}"
EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG:-configs/experiments.json}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"

echo "===================================================================="
echo "ITCAS runtime benchmark aggregate | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "cells_dir=${CELLS_DIR} output=${OUTPUT}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[rtb_aggregate] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
fi
export PYTHONNOUSERSITE=1

CELLS_DIR="${CELLS_DIR}" OUTPUT="${OUTPUT}" \
EXPERIMENTS_CONFIG="${EXPERIMENTS_CONFIG}" PROBLEMS_CONFIG="${PROBLEMS_CONFIG}" \
python - <<'PYEOF'
import csv, glob, os, sys
from pathlib import Path

from itcas.reporting.runtime_benchmark import (
    DEFAULT_METHODS, build_report,
)
from itcas.reporting.summary import _synthetic_problems

cells_dir = Path(os.environ["CELLS_DIR"])
output = Path(os.environ["OUTPUT"])
problems_config = os.environ["PROBLEMS_CONFIG"]
experiments_config = os.environ["EXPERIMENTS_CONFIG"]

methods = list(DEFAULT_METHODS)
problems = _synthetic_problems(problems_config)

timings: dict[str, dict[str, float]] = {m: {} for m in methods}
missing = []
for problem in problems:
    for method in methods:
        f = cells_dir / f"{problem}__{method}.csv"
        if not f.exists():
            missing.append(f.name)
            continue
        with f.open() as fh:
            row = next(csv.reader(fh))
        m, p, secs = row[0], row[1], float(row[2])
        assert m == method and p == problem, f"mismatched cell file {f}: {row}"
        timings[method][problem] = secs

if missing:
    print(f"[rtb_aggregate] ERROR: {len(missing)} cell(s) missing, refusing to write a partial report:", file=sys.stderr)
    for f in missing:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)

report = build_report(timings, methods, problems, experiments_config)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(report + "\n")
print(f"[rtb_aggregate] wrote {output} ({len(problems)} problems x {len(methods)} methods, {len(problems) * len(methods)} cells)")
PYEOF
RC=$?

echo "===================================================================="
echo "[rtb_aggregate] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
exit "${RC}"
