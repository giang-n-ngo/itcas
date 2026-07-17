#!/usr/bin/env bash
#SBATCH --job-name=itcas_synth_dispatch
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_synth_dispatch-%j.out
#SBATCH --error=slurm_logs/itcas_synth_dispatch-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# synthetic_comparison_dispatch.sbatch — orchestrator for the memory-bounded,
# split synthetic-comparison pipeline. Mirrors the existing
# summarize.sbatch -> summarize_problem.sbatch -> summarize_aggregate.sbatch
# fan-out/fan-in pattern, but for `itcas.summarize --synthetic-comparison-*`
# (see itcas.reporting.summary.summarize_synthetic_comparison_problem /
# summarize_synthetic_comparison_aggregate) instead of --benchmark/
# --aggregate-from.
#
# THIS is the script to submit for full-scale synthetic-comparison runs.
# scripts/synthetic_comparison.sbatch (the older monolithic single-job
# version, `--synthetic-comparison`, everything in memory at once) OOM-killed
# at 16G because it loaded every synthetic problem's full run history
# simultaneously (job 3427762, 15.99/16.00 GB, died ~11 min in). It is kept
# unchanged only for quick local/small-scale testing (e.g. a handful of
# problems on a workstation) -- do NOT submit it for the full ~15-problem
# sweep. Use this dispatch script instead.
#
# Workflow
# --------
#   1. This job (the orchestrator) runs Python to get the authoritative list
#      of synthetic problem names via
#      itcas.reporting.summary._synthetic_problems(PROBLEMS_CONFIG), i.e. the
#      "problems" list in PROBLEMS_CONFIG (configs/final_problems.json) minus
#      spacecraft_formation_flying_a1 (the one Basilisk-simulation problem,
#      excluded because it isn't a closed-form synthetic objective). This is
#      the exact same helper the Python pipeline itself uses, so the job list
#      here can never silently diverge from what the aggregate step expects.
#   2. For each problem it submits synthetic_comparison_problem.sbatch, which:
#        - loads only that problem's own runs (never any other problem's
#          .jsonl logs)
#        - renders that problem's own per-difficulty PDF(s)
#        - writes <METRICS_DIR>/<problem>_synthetic_metrics.json (a small
#          intermediate summary, not raw run data)
#   3. After all per-problem jobs succeed, synthetic_comparison_aggregate.sbatch
#      runs to produce the combined avg-rank PDF, relative-AUC PDF, and
#      Friedman/Wilcoxon stats report per difficulty, by loading only the
#      *_synthetic_metrics.json files under METRICS_DIR (no .jsonl/raw run
#      logs touched at all).
#
# Output structure
# ----------------
#   OUTPUT_DIR/
#     <difficulty>/synthetic_comparison_<problem>_vs_evaluations.pdf  (per problem)
#     <difficulty>/synthetic_comparison_avg_rank_vs_evaluations.pdf   (aggregate)
#     <difficulty>/synthetic_comparison_relative_auc_vs_evaluations.pdf (aggregate)
#     <difficulty>/synthetic_comparison_stats_report.{json,md}        (aggregate)
#   METRICS_DIR/
#     <problem>_synthetic_metrics.json   (intermediate, used by aggregate)
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                              # conda env to activate
#   CONDA_MODULE=Anaconda3                       # module providing conda
#   INPUT_DIR=results/sweep                      # sweep root to read
#   PROBLEMS_CONFIG=configs/final_problems.json  # problem suite config
#   OUTPUT_DIR=results/synthetic_comparison       # where <difficulty>/ subfolders go
#   METRICS_DIR=<OUTPUT_DIR>/metrics              # where per-problem metrics go
#   ALPHA=0.05                                    # Friedman gate / Wilcoxon significance level
#
# Submit:
#   sbatch scripts/synthetic_comparison_dispatch.sbatch
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
    echo "[synth_dispatch] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
INPUT_DIR="${INPUT_DIR:-results/sweep}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/synthetic_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
ALPHA="${ALPHA:-0.05}"
export INPUT_DIR PROBLEMS_CONFIG OUTPUT_DIR METRICS_DIR ALPHA CONDA_ENV CONDA_MODULE

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "===================================================================="
echo "ITCAS synthetic comparison dispatch | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} problems_config=${PROBLEMS_CONFIG}"
echo "output_dir=${OUTPUT_DIR} metrics_dir=${METRICS_DIR}"
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
        echo "[synth_dispatch] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Discover synthetic problems ---------------------------------------------
# Delegate to the same helper the Python pipeline itself uses
# (itcas.reporting.summary._synthetic_problems) so this list can never
# silently diverge from what summarize_synthetic_comparison_aggregate
# expects. This reads only PROBLEMS_CONFIG (a small JSON config), never any
# run logs.
echo "[synth_dispatch] Discovering synthetic problems from ${PROBLEMS_CONFIG}..."
PROBLEMS=$(python - <<'PYEOF'
import os, sys
from itcas.reporting.summary import _synthetic_problems
problems = _synthetic_problems(os.environ["PROBLEMS_CONFIG"])
if not problems:
    sys.exit("[synth_dispatch] ERROR: no synthetic problems found in PROBLEMS_CONFIG")
print("\n".join(problems))
PYEOF
)

if [[ -z "${PROBLEMS}" ]]; then
    echo "[synth_dispatch] No synthetic problems found. Nothing to do."
    exit 0
fi

echo "[synth_dispatch] Synthetic problems found:"
echo "${PROBLEMS}" | sed 's/^/  /'

# --- Submit one per-problem job per problem ----------------------------------
JOB_IDS=""
while IFS= read -r PROBLEM; do
    [[ -z "${PROBLEM}" ]] && continue
    JID=$(sbatch --parsable \
        --export=ALL,PROBLEM="${PROBLEM}",INPUT_DIR="${INPUT_DIR}",PROBLEMS_CONFIG="${PROBLEMS_CONFIG}",OUTPUT_DIR="${OUTPUT_DIR}",METRICS_DIR="${METRICS_DIR}",ALPHA="${ALPHA}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
        "${REPO_ROOT}/scripts/synthetic_comparison_problem.sbatch")
    echo "[synth_dispatch] Submitted per-problem job ${JID} for '${PROBLEM}'"
    JOB_IDS="${JOB_IDS}:${JID}"
done <<< "${PROBLEMS}"

# Strip leading ":"
JOB_IDS="${JOB_IDS#:}"

if [[ -z "${JOB_IDS}" ]]; then
    echo "[synth_dispatch] No per-problem jobs were submitted. Exiting."
    exit 1
fi

# --- Submit aggregate job (runs after all per-problem jobs succeed) ----------
AGG_JID=$(sbatch --parsable \
    --dependency="afterok:${JOB_IDS}" \
    --export=ALL,METRICS_DIR="${METRICS_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",ALPHA="${ALPHA}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
    "${REPO_ROOT}/scripts/synthetic_comparison_aggregate.sbatch")
echo "[synth_dispatch] Submitted aggregate job ${AGG_JID} (depends on: ${JOB_IDS})"

echo "===================================================================="
echo "[synth_dispatch] Done. Per-problem jobs: ${JOB_IDS}"
echo "[synth_dispatch] Aggregate job:          ${AGG_JID}"
echo "[synth_dispatch] Monitor with: squeue -j ${JOB_IDS//:/,},${AGG_JID}"
echo "[synth_dispatch] finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
