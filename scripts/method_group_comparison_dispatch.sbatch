#!/usr/bin/env bash
#SBATCH --job-name=itcas_mgc_dispatch
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_mgc_dispatch-%j.out
#SBATCH --error=slurm_logs/itcas_mgc_dispatch-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# method_group_comparison_dispatch.sbatch — orchestrator for the memory-
# bounded, split method-group-comparison pipeline. Mirrors
# scripts/synthetic_comparison_dispatch.sbatch's fan-out/fan-in pattern
# (that script's header explains why the split form exists at all: a
# monolithic single-job version OOM-killed at 16G loading every synthetic
# problem's run history at once, job 3427762), but for
# `itcas.summarize --method-group-comparison-*` (see
# itcas.reporting.method_group_comparison.summarize_method_group_comparison_problem
# / summarize_method_group_comparison_aggregate) instead of
# `--synthetic-comparison-*`.
#
# This pipeline runs TWO independent comparisons, selected by the required
# METHOD_GROUP env var:
#   METHOD_GROUP=batch       -> the 5 batch methods against each other
#   METHOD_GROUP=sequential  -> the 5 sequential methods against each other
# (`random` is excluded from both -- see method_group_comparison.py's module
# docstring.) There is no monolithic (non-split) form of this pipeline at
# all -- unlike synthetic_comparison.sbatch, which is kept only for small
# local testing, this report was written after that OOM lesson was already
# learned, so only the split form is offered. This ONE dispatch script (plus
# its two per-stage siblings) serves BOTH groups -- submit it twice, once per
# METHOD_GROUP value, rather than duplicating files.
#
# Workflow
# --------
#   1. This job (the orchestrator) validates METHOD_GROUP is exactly "batch"
#      or "sequential" (fails fast, before submitting anything, if not), then
#      runs Python to get the authoritative list of synthetic problem names
#      via itcas.reporting.summary._synthetic_problems(PROBLEMS_CONFIG) -- the
#      exact same helper synthetic_comparison_dispatch.sbatch uses, since this
#      pipeline covers the identical synthetic problem suite (just fewer
#      methods per problem, see method_group_comparison.py's module
#      docstring). This is the same helper the Python pipeline itself uses,
#      so the job list here can never silently diverge from what the
#      aggregate step expects.
#   2. For each problem it submits method_group_comparison_problem.sbatch,
#      which loads only that problem's runs for METHOD_GROUP's 5 methods and
#      writes <METRICS_DIR>/<problem>_<METHOD_GROUP>_comparison_metrics.json
#      (a small intermediate summary, not raw run data; no per-problem PDF is
#      rendered by this pipeline at all -- see module docstring).
#   3. After all per-problem jobs succeed, method_group_comparison_aggregate.sbatch
#      runs to produce the combined relative-AUC and normalized-curve figures
#      for METHOD_GROUP, by loading only the
#      *_<METHOD_GROUP>_comparison_metrics.json files under METRICS_DIR (no
#      .jsonl/raw run logs touched at all). There is no avg-rank plot and no
#      Friedman/Wilcoxon stats report in this pipeline (unlike the synthetic
#      one) -- see method_group_comparison.py's module docstring.
#
# Job naming / log files for children
# ------------------------------------
# #SBATCH --job-name/--output/--error directives are parsed by sbatch before
# any --export env vars (including METHOD_GROUP) are available, so they
# cannot reference ${METHOD_GROUP} inside method_group_comparison_problem.sbatch
# / method_group_comparison_aggregate.sbatch's own headers. Instead, THIS
# script bakes METHOD_GROUP into the *submission* by passing --job-name/-o/-e
# explicitly on each child's `sbatch` command line (command-line options
# override a script's #SBATCH directives) -- exactly the same way it already
# passes dynamic values via --export=... -- so batch/sequential runs stay
# distinguishable in `squeue`/log files and can run concurrently without
# clobbering each other's logs.
#
# Output structure
# ----------------
#   OUTPUT_DIR/
#     <difficulty>/<group>_comparison_relative_auc_vs_evaluations.pdf  (aggregate)
#     <group>_comparison_relative_auc_by_difficulty.pdf                (aggregate)
#     <group>_comparison_normalized_avg_curve_vs_pct_budget.pdf        (aggregate)
#   METRICS_DIR/
#     <problem>_<group>_comparison_metrics.json   (intermediate, used by aggregate)
#
# Environment overrides (export before sbatch, or via --export):
#   METHOD_GROUP=batch|sequential                 # REQUIRED, no default -- which
#                                                  # comparison to run
#   CONDA_ENV=itcas                               # conda env to activate
#   CONDA_MODULE=Anaconda3                        # module providing conda
#   INPUT_DIR=results/sweep                       # sweep root to read
#   PROBLEMS_CONFIG=configs/final_problems.json   # problem suite config
#   OUTPUT_DIR=results/<METHOD_GROUP>_comparison   # where aggregate PDFs go
#                                                  # (results/batch_comparison or
#                                                  # results/sequential_comparison)
#   METRICS_DIR=<OUTPUT_DIR>/metrics               # where per-problem metrics go
#
# Submit (one call per group -- both groups must be submitted separately):
#   sbatch --job-name=itcas_mgc_batch_dispatch \
#          --output=slurm_logs/itcas_mgc_batch_dispatch-%j.out \
#          --error=slurm_logs/itcas_mgc_batch_dispatch-%j.err \
#          --export=ALL,METHOD_GROUP=batch \
#          scripts/method_group_comparison_dispatch.sbatch
#
#   sbatch --job-name=itcas_mgc_sequential_dispatch \
#          --output=slurm_logs/itcas_mgc_sequential_dispatch-%j.out \
#          --error=slurm_logs/itcas_mgc_sequential_dispatch-%j.err \
#          --export=ALL,METHOD_GROUP=sequential \
#          scripts/method_group_comparison_dispatch.sbatch
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
    echo "[mgc_dispatch] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

# --- Validate METHOD_GROUP before submitting anything -----------------------
METHOD_GROUP="${METHOD_GROUP:-}"
if [[ "${METHOD_GROUP}" != "batch" && "${METHOD_GROUP}" != "sequential" ]]; then
    echo "[mgc_dispatch] ERROR: METHOD_GROUP must be exactly 'batch' or 'sequential'," \
         "got '${METHOD_GROUP}'. Set it via, e.g.," \
         "'sbatch --export=ALL,METHOD_GROUP=batch scripts/method_group_comparison_dispatch.sbatch'." >&2
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
INPUT_DIR="${INPUT_DIR:-results/sweep}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${METHOD_GROUP}_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
export INPUT_DIR PROBLEMS_CONFIG OUTPUT_DIR METRICS_DIR METHOD_GROUP CONDA_ENV CONDA_MODULE

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "===================================================================="
echo "ITCAS method-group comparison dispatch | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "method_group=${METHOD_GROUP}"
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
        echo "[mgc_dispatch] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
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
# silently diverge from what summarize_method_group_comparison_aggregate
# expects, and stays identical to synthetic_comparison_dispatch.sbatch's own
# job list (this pipeline uses the same problem suite, see module docstring).
# This reads only PROBLEMS_CONFIG (a small JSON config), never any run logs.
echo "[mgc_dispatch] Discovering synthetic problems from ${PROBLEMS_CONFIG}..."
PROBLEMS=$(python - <<'PYEOF'
import os, sys
from itcas.reporting.summary import _synthetic_problems
problems = _synthetic_problems(os.environ["PROBLEMS_CONFIG"])
if not problems:
    sys.exit("[mgc_dispatch] ERROR: no synthetic problems found in PROBLEMS_CONFIG")
print("\n".join(problems))
PYEOF
)

if [[ -z "${PROBLEMS}" ]]; then
    echo "[mgc_dispatch] No synthetic problems found. Nothing to do."
    exit 0
fi

echo "[mgc_dispatch] Synthetic problems found:"
echo "${PROBLEMS}" | sed 's/^/  /'

# --- Submit one per-problem job per problem ----------------------------------
JOB_IDS=""
while IFS= read -r PROBLEM; do
    [[ -z "${PROBLEM}" ]] && continue
    JID=$(sbatch --parsable \
        --job-name="itcas_mgc_${METHOD_GROUP}_problem" \
        --output="slurm_logs/itcas_mgc_${METHOD_GROUP}_problem-%j.out" \
        --error="slurm_logs/itcas_mgc_${METHOD_GROUP}_problem-%j.err" \
        --export=ALL,METHOD_GROUP="${METHOD_GROUP}",PROBLEM="${PROBLEM}",INPUT_DIR="${INPUT_DIR}",PROBLEMS_CONFIG="${PROBLEMS_CONFIG}",OUTPUT_DIR="${OUTPUT_DIR}",METRICS_DIR="${METRICS_DIR}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
        "${REPO_ROOT}/scripts/method_group_comparison_problem.sbatch")
    echo "[mgc_dispatch] Submitted per-problem job ${JID} for '${PROBLEM}' (group=${METHOD_GROUP})"
    JOB_IDS="${JOB_IDS}:${JID}"
done <<< "${PROBLEMS}"

# Strip leading ":"
JOB_IDS="${JOB_IDS#:}"

if [[ -z "${JOB_IDS}" ]]; then
    echo "[mgc_dispatch] No per-problem jobs were submitted. Exiting."
    exit 1
fi

# --- Submit aggregate job (runs after all per-problem jobs succeed) ----------
AGG_JID=$(sbatch --parsable \
    --job-name="itcas_mgc_${METHOD_GROUP}_aggregate" \
    --output="slurm_logs/itcas_mgc_${METHOD_GROUP}_aggregate-%j.out" \
    --error="slurm_logs/itcas_mgc_${METHOD_GROUP}_aggregate-%j.err" \
    --dependency="afterok:${JOB_IDS}" \
    --export=ALL,METHOD_GROUP="${METHOD_GROUP}",METRICS_DIR="${METRICS_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
    "${REPO_ROOT}/scripts/method_group_comparison_aggregate.sbatch")
echo "[mgc_dispatch] Submitted aggregate job ${AGG_JID} (group=${METHOD_GROUP}, depends on: ${JOB_IDS})"

echo "===================================================================="
echo "[mgc_dispatch] Done. method_group=${METHOD_GROUP}"
echo "[mgc_dispatch] Per-problem jobs: ${JOB_IDS}"
echo "[mgc_dispatch] Aggregate job:    ${AGG_JID}"
echo "[mgc_dispatch] Monitor with: squeue -j ${JOB_IDS//:/,},${AGG_JID}"
echo "[mgc_dispatch] finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
