#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# regenerate_normalized_curve_reports.sh — convenience orchestrator, NOT a
# Slurm job itself. Just submits the 3 existing per-family report pipelines
# (each already produces the new "normalized average metric curve vs. % of
# evaluation budget" PDF as a side effect of its normal run -- no script
# content changes needed for any of them):
#
#   1. scripts/casd_comparison.sbatch              -> results/casd_comparison/casd_comparison_normalized_avg_curve_vs_pct_budget.pdf
#   2. scripts/ff_comparison.sbatch                -> results/ff_comparison/ff_comparison_normalized_avg_curve_vs_pct_budget.pdf
#   3. scripts/synthetic_comparison_dispatch.sbatch -> results/synthetic_comparison/<difficulty>/synthetic_comparison_normalized_avg_curve_vs_pct_budget.pdf
#
# Item 3 MUST be the full dispatch pipeline (per-problem fan-out ->
# `--dependency=afterok` -> aggregate fan-in), not just the aggregate step:
# the new figure's data (normalized_curve_lists) is computed in the
# per-problem stage and stored in <METRICS_DIR>/<problem>_synthetic_metrics.json.
# Any such JSON predating this change lacks that key, so re-running only the
# aggregate step would silently omit those problems from the new figure.
# synthetic_comparison_dispatch.sbatch already does the right thing here (see
# its own header comment) -- this wrapper just calls it.
#
# All three jobs are cheap (~15-30 min CPU-only per their own sbatch header
# comments), read-only against results/sweep, and idempotent (they overwrite
# their own output dirs in place) -- safe to submit together, no coordination
# needed between them.
#
# This script does no computation itself -- it only calls `sbatch` 3 times --
# so it is a plain bash script run directly on the login/submit node, NOT
# submitted via sbatch itself (that would waste a queue slot on ~instant
# work).
#
# Usage:
#   bash scripts/regenerate_normalized_curve_reports.sh
#
# Environment overrides (forwarded verbatim to each underlying sbatch script
# via --export=ALL; leave unset to use each script's own defaults, which is
# almost always what you want for a straight report regeneration):
#   INPUT_DIR, OUTPUT_DIR, ALPHA                         (casd_comparison.sbatch)
#   INPUT_DIR, OUTPUT_DIR, ALPHA                         (ff_comparison.sbatch)
#   INPUT_DIR, PROBLEMS_CONFIG, OUTPUT_DIR, METRICS_DIR, ALPHA
#                                                         (synthetic_comparison_dispatch.sbatch)
#   NOTE: the three families use SEPARATE OUTPUT_DIR defaults
#   (results/casd_comparison, results/ff_comparison, results/synthetic_comparison)
#   -- if you override OUTPUT_DIR here it applies to ALL THREE jobs
#   identically, which is almost never what you want across families. Prefer
#   leaving OUTPUT_DIR unset and letting each script use its own default, or
#   submit the underlying .sbatch scripts individually with per-family
#   overrides instead of using this wrapper.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p slurm_logs

TS="$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
echo "[regen_normalized_curves] Submitting 3 report-regeneration jobs (repo=${REPO_ROOT})"
echo "[regen_normalized_curves] start=${TS}"
echo "===================================================================="

CASD_JID=$(sbatch --parsable "${REPO_ROOT}/scripts/casd_comparison.sbatch")
echo "[regen_normalized_curves] CASD comparison        -> job ${CASD_JID} (scripts/casd_comparison.sbatch)"

FF_JID=$(sbatch --parsable "${REPO_ROOT}/scripts/ff_comparison.sbatch")
echo "[regen_normalized_curves] FF comparison          -> job ${FF_JID} (scripts/ff_comparison.sbatch)"

SYN_DISPATCH_JID=$(sbatch --parsable "${REPO_ROOT}/scripts/synthetic_comparison_dispatch.sbatch")
echo "[regen_normalized_curves] Synthetic dispatch     -> job ${SYN_DISPATCH_JID} (scripts/synthetic_comparison_dispatch.sbatch)"
echo "[regen_normalized_curves]   (this dispatcher will itself submit per-problem jobs + an"
echo "[regen_normalized_curves]    aggregate job with --dependency=afterok; see its own stdout"
echo "[regen_normalized_curves]    in slurm_logs/itcas_synth_dispatch-${SYN_DISPATCH_JID}.out for those child job IDs)"

echo "===================================================================="
echo "[regen_normalized_curves] Submitted job IDs: casd=${CASD_JID} ff=${FF_JID} synth_dispatch=${SYN_DISPATCH_JID}"
echo "[regen_normalized_curves] Monitor with: squeue -j ${CASD_JID},${FF_JID},${SYN_DISPATCH_JID}"
echo "[regen_normalized_curves] Full synthetic pipeline once dispatch's children appear: squeue -u \$USER -n itcas_synth_problem,itcas_synth_aggregate"
echo "[regen_normalized_curves] Logs:"
echo "[regen_normalized_curves]   slurm_logs/itcas_casd_comparison-${CASD_JID}.{out,err}"
echo "[regen_normalized_curves]   slurm_logs/itcas_ff_comparison-${FF_JID}.{out,err}"
echo "[regen_normalized_curves]   slurm_logs/itcas_synth_dispatch-${SYN_DISPATCH_JID}.{out,err}"
echo "[regen_normalized_curves] finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
