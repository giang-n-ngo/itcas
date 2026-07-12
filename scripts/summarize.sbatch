#!/usr/bin/env bash
#SBATCH --job-name=itcas_summary_dispatch
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_summary_dispatch-%j.out
#SBATCH --error=slurm_logs/itcas_summary_dispatch-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# summarize.sbatch — orchestrator: fans out one Slurm job per problem, then
# submits a final aggregate job that depends on all per-problem jobs.
#
# Workflow
# --------
#   1. This job (the orchestrator) runs Python to discover all problem names
#      found under INPUT_DIR.
#   2. For each problem it submits summarize_problem.sbatch, which:
#        - renders per-problem summary PDFs
#        - writes <METRICS_DIR>/<problem>_metrics.json
#   3. After all per-problem jobs succeed, summarize_aggregate.sbatch runs
#      to produce the combined hypervolume figure and global stats reports
#      by loading the metrics JSONs (no re-reading of run logs).
#
# Output structure
# ----------------
#   OUTPUT_DIR/
#     <problem>_summary_vs_evaluations.pdf   (one per problem)
#     <problem>_summary_vs_steps.pdf         (one per problem)
#     <problem>_<variant>_stats_report.json  (one per problem × variant)
#     <problem>_<variant>_stats_report.md
#     metrics/
#       <problem>_metrics.json               (intermediate, used by aggregate)
#     all_problems_hypervolume_vs_evaluations.pdf
#     all_problems_hypervolume_vs_steps.pdf
#     all_problems_<variant>_stats_report.json
#     all_problems_<variant>_stats_report.md
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                          # conda env to activate
#   CONDA_MODULE=Anaconda3                   # module providing conda
#   INPUT_DIR=results/sweep                  # sweep root to scan
#   OUTPUT_DIR=results/sweep/_summary        # where to write all outputs
#   METRICS_DIR=<OUTPUT_DIR>/metrics         # where per-problem metrics go
#   ALPHA=0.05                               # stats significance level
#
# Submit:
#   sbatch scripts/summarize.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs 2>/dev/null || \
    echo "[dispatch] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
INPUT_DIR="${INPUT_DIR:-results/sweep}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/_summary}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
ALPHA="${ALPHA:-0.05}"
export INPUT_DIR OUTPUT_DIR METRICS_DIR ALPHA CONDA_ENV CONDA_MODULE

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "===================================================================="
echo "ITCAS summary dispatch | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} output_dir=${OUTPUT_DIR} metrics_dir=${METRICS_DIR}"
echo "conda_env=${CONDA_ENV}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[dispatch] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Discover problems -------------------------------------------------------
# Read only the small *.summary.json files — never the large *.jsonl logs.
echo "[dispatch] Scanning ${INPUT_DIR} for problems..."
PROBLEMS=$(python - <<'PYEOF'
import json, os, sys
from pathlib import Path
problems = set()
for f in Path(os.environ["INPUT_DIR"]).rglob("*.summary.json"):
    try:
        d = json.loads(f.read_text())
        p = d.get("problem") or (d.get("config") or {}).get("problem")
        if p and p != "unknown":
            problems.add(p)
    except Exception:
        pass
if not problems:
    sys.exit("[dispatch] ERROR: no summary files found under INPUT_DIR")
print("\n".join(sorted(problems)))
PYEOF
)

if [[ -z "${PROBLEMS}" ]]; then
    echo "[dispatch] No problems found under ${INPUT_DIR}. Nothing to do."
    exit 0
fi

echo "[dispatch] Problems found:"
echo "${PROBLEMS}" | sed 's/^/  /'

# --- Submit one per-problem job per problem ----------------------------------
JOB_IDS=""
while IFS= read -r PROBLEM; do
    [[ -z "${PROBLEM}" ]] && continue
    JID=$(sbatch --parsable \
        --export=ALL,BENCHMARK="${PROBLEM}",INPUT_DIR="${INPUT_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",METRICS_DIR="${METRICS_DIR}",ALPHA="${ALPHA}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
        "${REPO_ROOT}/scripts/summarize_problem.sbatch")
    echo "[dispatch] Submitted per-problem job ${JID} for '${PROBLEM}'"
    JOB_IDS="${JOB_IDS}:${JID}"
done <<< "${PROBLEMS}"

# Strip leading ":"
JOB_IDS="${JOB_IDS#:}"

if [[ -z "${JOB_IDS}" ]]; then
    echo "[dispatch] No per-problem jobs were submitted. Exiting."
    exit 1
fi

# --- Submit aggregate job (runs after all per-problem jobs succeed) ----------
AGG_JID=$(sbatch --parsable \
    --dependency="afterok:${JOB_IDS}" \
    --export=ALL,METRICS_DIR="${METRICS_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",ALPHA="${ALPHA}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
    "${REPO_ROOT}/scripts/summarize_aggregate.sbatch")
echo "[dispatch] Submitted aggregate job ${AGG_JID} (depends on: ${JOB_IDS})"

echo "===================================================================="
echo "[dispatch] Done. Per-problem jobs: ${JOB_IDS}"
echo "[dispatch] Aggregate job:          ${AGG_JID}"
echo "[dispatch] Monitor with: squeue -j ${JOB_IDS//:/,},${AGG_JID}"
echo "[dispatch] finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
