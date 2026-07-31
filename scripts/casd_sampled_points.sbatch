#!/usr/bin/env bash
#SBATCH --job-name=itcas_casd_sampled_points
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:08:00
#SBATCH --output=slurm_logs/itcas_casd_sampled_points-%j.out
#SBATCH --error=slurm_logs/itcas_casd_sampled_points-%j.err
#SBATCH --mail-user=g.ngo@deakin.edu.au
#SBATCH --mail-type=END,FAIL
# ---------------------------------------------------------------------------
# casd_sampled_points.sbatch — single lightweight CPU job (no array, no
# dependency), structurally identical to casd_comparison.sbatch (same
# problem, same --input-dir tree, same CPU-only profile), just a different
# report.
#
# Runs itcas/reporting/casd_sampled_points.py, which reads run logs under
# --input-dir for the "CASD problem" (casd_llm) and plots the raw points a
# single representative run of each main algorithm actually *sampled* (both
# in objective space -- Safety vs. Utility -- and in context space --
# prompt_toxicity vs. prompt_length), split into two method groups
# (SEQUENTIAL_METHODS: the four forced-sequential baselines; BATCH_METHODS:
# itcas_ndig, the one proposed batch method) x the 4 CASD difficulty levels
# (p1_00..p4_00) = 8 grid PDFs written to --output-dir. Unlike
# casd_comparison.py, this module's CLI takes no --alpha (there is no
# Wilcoxon/Friedman stats battery here -- it's discovery + one matplotlib
# scatter render per figure) and defaults --seed to 0, selecting one
# representative seed per (method, difficulty) column rather than
# aggregating over seeds.
#
# CPU-only (matplotlib PDF rendering only, no GPU, no torch model fitting,
# no CASD server involved -- just reads already-logged run point arrays).
#
# Resource sizing: sized down from casd_comparison.sbatch's 00:15:00/4G/4
# cpus, NOT measured locally (no cluster access at authoring time). This job
# reads a strict subset of what casd_comparison.sbatch reads -- 5 methods x
# 4 levels x 1 seed each (20 runs) vs that job's 6 methods x 4 levels x ~20
# seeds (480 runs), roughly 24x fewer run logs to parse -- and does no
# Friedman-gated Holm-Bonferroni-corrected Wilcoxon battery, which
# casd_comparison.sbatch's header notes dominates its overhead regardless of
# data volume. Since that stats overhead is absent here and discovery I/O is
# the main cost, --time=00:08:00 / --mem=2G / --cpus-per-task=2 is a
# conservative-but-not-razor-thin cut (roughly half the time/mem, half the
# cpus) rather than a proportional 24x cut, to leave headroom for matplotlib
# rendering 8 PDFs (16 scatter panels total) and any filesystem slowness on
# the shared results/sweep tree. Revise once actual run time is observed on
# this cluster.
#
# Environment overrides (export before sbatch, or via --export):
#   CONDA_ENV=itcas                                # conda env to activate
#   CONDA_MODULE=Anaconda3                         # module providing conda
#   INPUT_DIR=results/sweep                        # run-log tree to read
#   OUTPUT_DIR=results/casd_sampled_points          # where the 8 PDFs are written
#   SEED=0                                          # representative seed selected per (method, difficulty) column
#
# Submit:
#   sbatch scripts/casd_sampled_points.sbatch
#
# Override example:
#   INPUT_DIR=results/sweep OUTPUT_DIR=results/casd_sampled_points/rerun SEED=1 \
#     sbatch --export=ALL,INPUT_DIR,OUTPUT_DIR,SEED scripts/casd_sampled_points.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs 2>/dev/null || true

CONDA_ENV="${CONDA_ENV:-itcas}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
INPUT_DIR="${INPUT_DIR:-results/sweep}"
OUTPUT_DIR="${OUTPUT_DIR:-results/casd_sampled_points}"
SEED="${SEED:-0}"

echo "===================================================================="
echo "ITCAS CASD sampled points | job ${SLURM_JOB_ID:-local} | host=$(hostname)"
echo "input_dir=${INPUT_DIR} output_dir=${OUTPUT_DIR} seed=${SEED}"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

# --- Activate environment ---------------------------------------------------
if [[ -n "${CONDA_ENV:-}" ]]; then
    module purge 2>/dev/null || true
    module load "${CONDA_MODULE}" 2>/dev/null || true
    source activate 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        echo "[casd_sampled_points] WARNING: could not activate '${CONDA_ENV}'; using current Python." >&2
elif [[ -n "${ITCAS_VENV:-}" && -f "${ITCAS_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ITCAS_VENV}/bin/activate"
fi
export PYTHONNOUSERSITE=1

THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export MPLBACKEND="Agg"

# --- Produce the CASD sampled-points PDFs -----------------------------------
mkdir -p "${OUTPUT_DIR}"
ARGS=(--input-dir "${INPUT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --seed "${SEED}")

srun --kill-on-bad-exit=1 python -m itcas.reporting.casd_sampled_points "${ARGS[@]}"
RC=$?

echo "===================================================================="
echo "[casd_sampled_points] finished rc=${RC} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[casd_sampled_points] outputs written under: ${OUTPUT_DIR}"
echo "===================================================================="
exit "${RC}"
