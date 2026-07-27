#!/usr/bin/env bash
#SBATCH --job-name=itcas_nka_dispatch
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/itcas_nka_dispatch-%j.out
#SBATCH --error=slurm_logs/itcas_nka_dispatch-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# ndig_kernel_ablation_comparison_dispatch.sbatch — orchestrator for the
# memory-bounded, split NDIG QD-DPP diversity-kernel-ablation comparison
# pipeline. Mirrors scripts/method_group_comparison_dispatch.sbatch's fan-out
# /fan-in pattern (itself mirroring synthetic_comparison_dispatch.sbatch's,
# whose header explains why the split form exists at all: a monolithic
# single-job version OOM-killed at 16G loading every synthetic problem's run
# history at once, job 3427762), but for
# `itcas.summarize --ndig-kernel-ablation-*` (see
# itcas.reporting.ndig_kernel_ablation_comparison.summarize_ndig_kernel_ablation_problem
# / summarize_ndig_kernel_ablation_aggregate) instead of
# `--method-group-comparison-*`.
#
# This pipeline compares the proposed method's full batch NDIG acquisition
# (itcas_ndig, i.e. method="itcas", quality="ndig") against its two QD-DPP
# diversity-kernel-ablated siblings (ndig_no_kobj_batch: drops k_obj from the
# L-ensemble; ndig_no_kctx_batch: drops k_ctx) -- see
# ndig_kernel_ablation_comparison.py's module docstring and
# itcas/pipeline/loop.py's NDIG_KERNEL_ABLATION. Unlike
# method_group_comparison_dispatch.sbatch, there is no group selector here --
# the method set is fixed, so this ONE dispatch submits ONE fan-out/fan-in,
# not two. Every per-problem job additionally produces a "_with_seq" variant
# of both output figures that includes itcas_seq_ndig (the forced-sequential
# sibling, no QD-DPP batch diversity at all) alongside the 3 batch methods --
# see module docstring's "Two output pairs" section.
#
# Workflow
# --------
#   1. This job (the orchestrator) runs Python to get the authoritative list
#      of synthetic problem names via
#      itcas.reporting.summary._synthetic_problems(PROBLEMS_CONFIG) -- the
#      same helper synthetic_comparison_dispatch.sbatch /
#      method_group_comparison_dispatch.sbatch use, since this pipeline
#      covers the identical synthetic problem suite (just 4 methods per
#      problem instead of 5/6). This is the same helper the Python pipeline
#      itself uses, so the job list here can never silently diverge from
#      what the aggregate step expects.
#   2. For each problem it submits ndig_kernel_ablation_comparison_problem.sbatch,
#      which loads only that problem's runs for the 4 methods (itcas_ndig,
#      ndig_no_kobj_batch, ndig_no_kctx_batch, itcas_seq_ndig) and writes
#      <METRICS_DIR>/<problem>_ndig_kernel_ablation_metrics.json (a small
#      intermediate summary, not raw run data, holding both the base and
#      "_with_seq" variants' data; no per-problem PDF is rendered by this
#      pipeline at all -- see module docstring).
#   3. After all per-problem jobs succeed,
#      ndig_kernel_ablation_comparison_aggregate.sbatch runs to produce the
#      combined relative-AUC-by-difficulty and normalized-curve figures, each
#      in both the base (3-method) and "_with_seq" (4-method) variant (each
#      a 2x2 grid of the 4 metrics), by loading only the
#      *_ndig_kernel_ablation_metrics.json files under METRICS_DIR (no
#      .jsonl/raw run logs touched at all). There is no avg-rank plot and no
#      Friedman/Wilcoxon stats report in this pipeline.
#
# Output structure
# ----------------
#   OUTPUT_DIR/
#     ndig_kernel_ablation_relative_auc_by_difficulty.pdf                    (aggregate, 2x2 grid, 3 methods)
#     ndig_kernel_ablation_normalized_avg_curve_vs_pct_budget.pdf             (aggregate, 2x2 grid, 3 methods)
#     ndig_kernel_ablation_relative_auc_by_difficulty_with_seq.pdf           (aggregate, 2x2 grid, 3 methods + itcas_seq_ndig)
#     ndig_kernel_ablation_normalized_avg_curve_vs_pct_budget_with_seq.pdf    (aggregate, 2x2 grid, 3 methods + itcas_seq_ndig)
#   METRICS_DIR/
#     <problem>_ndig_kernel_ablation_metrics.json   (intermediate, used by aggregate)
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                                        # conda env to activate
#   CONDA_MODULE=Anaconda3                                 # module providing conda
#   INPUT_DIR=results/sweep                                # sweep root to read
#   PROBLEMS_CONFIG=configs/final_problems.json            # problem suite config
#   OUTPUT_DIR=results/ndig_kernel_ablation_comparison     # where aggregate PDFs go
#   METRICS_DIR=<OUTPUT_DIR>/metrics                       # where per-problem metrics go
#
# Submit:
#   sbatch scripts/ndig_kernel_ablation_comparison_dispatch.sbatch
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
    echo "[nka_dispatch] WARNING: could not create '${REPO_ROOT}/slurm_logs' (continuing)." >&2

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
INPUT_DIR="${INPUT_DIR:-results/sweep}"
PROBLEMS_CONFIG="${PROBLEMS_CONFIG:-configs/final_problems.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ndig_kernel_ablation_comparison}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
export INPUT_DIR PROBLEMS_CONFIG OUTPUT_DIR METRICS_DIR CONDA_ENV CONDA_MODULE

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "===================================================================="
echo "ITCAS NDIG kernel-ablation comparison dispatch | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
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
        echo "[nka_dispatch] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
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
# silently diverge from what summarize_ndig_kernel_ablation_aggregate
# expects, and stays identical to synthetic_comparison_dispatch.sbatch's /
# method_group_comparison_dispatch.sbatch's own job list (this pipeline uses
# the same problem suite, see module docstring). This reads only
# PROBLEMS_CONFIG (a small JSON config), never any run logs.
echo "[nka_dispatch] Discovering synthetic problems from ${PROBLEMS_CONFIG}..."
PROBLEMS=$(python - <<'PYEOF'
import os, sys
from itcas.reporting.summary import _synthetic_problems
problems = _synthetic_problems(os.environ["PROBLEMS_CONFIG"])
if not problems:
    sys.exit("[nka_dispatch] ERROR: no synthetic problems found in PROBLEMS_CONFIG")
print("\n".join(problems))
PYEOF
)

if [[ -z "${PROBLEMS}" ]]; then
    echo "[nka_dispatch] No synthetic problems found. Nothing to do."
    exit 0
fi

echo "[nka_dispatch] Synthetic problems found:"
echo "${PROBLEMS}" | sed 's/^/  /'

# --- Submit one per-problem job per problem ----------------------------------
JOB_IDS=""
while IFS= read -r PROBLEM; do
    [[ -z "${PROBLEM}" ]] && continue
    JID=$(sbatch --parsable \
        --export=ALL,PROBLEM="${PROBLEM}",INPUT_DIR="${INPUT_DIR}",PROBLEMS_CONFIG="${PROBLEMS_CONFIG}",OUTPUT_DIR="${OUTPUT_DIR}",METRICS_DIR="${METRICS_DIR}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
        "${REPO_ROOT}/scripts/ndig_kernel_ablation_comparison_problem.sbatch")
    echo "[nka_dispatch] Submitted per-problem job ${JID} for '${PROBLEM}'"
    JOB_IDS="${JOB_IDS}:${JID}"
done <<< "${PROBLEMS}"

# Strip leading ":"
JOB_IDS="${JOB_IDS#:}"

if [[ -z "${JOB_IDS}" ]]; then
    echo "[nka_dispatch] No per-problem jobs were submitted. Exiting."
    exit 1
fi

# --- Submit aggregate job (runs after all per-problem jobs succeed) ----------
AGG_JID=$(sbatch --parsable \
    --dependency="afterok:${JOB_IDS}" \
    --export=ALL,METRICS_DIR="${METRICS_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",CONDA_ENV="${CONDA_ENV}",CONDA_MODULE="${CONDA_MODULE}" \
    "${REPO_ROOT}/scripts/ndig_kernel_ablation_comparison_aggregate.sbatch")
echo "[nka_dispatch] Submitted aggregate job ${AGG_JID} (depends on: ${JOB_IDS})"

echo "===================================================================="
echo "[nka_dispatch] Done."
echo "[nka_dispatch] Per-problem jobs: ${JOB_IDS}"
echo "[nka_dispatch] Aggregate job:    ${AGG_JID}"
echo "[nka_dispatch] Monitor with: squeue -j ${JOB_IDS//:/,},${AGG_JID}"
echo "[nka_dispatch] finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
