#!/usr/bin/env bash
# Delete all result files for a given method (or itcas quality variant) across
# all (or selected) problems and difficulties, then remove stale per-difficulty
# comparison plots and the shared _summary/ directory so they are regenerated on
# the next summarize run.
#
# Default mode is DRY-RUN: pass --execute to actually delete anything.
#
# Usage:
#   bash scripts/delete_method_results.sh <method> [OPTIONS]
#
# Options:
#   --quality <name>     For itcas only: restrict to one quality variant
#                        (efig | edig | ndig | roi_mi). Omit to delete all variants.
#   --problem <name>     Restrict to one problem (default: all problems)
#   --difficulty <name>  Restrict to one difficulty dir, e.g. p0_05 (default: all)
#   --sweep-root <path>  Sweep root directory (default: results/sweep)
#   --keep-summary       Do NOT delete _summary/ even when method dirs are found
#   --execute            Actually delete (omit for a safe dry-run preview)
#
# Examples:
#   # Preview what would be deleted for cas_eci across all problems:
#   bash scripts/delete_method_results.sh cas_eci
#
#   # Delete only the efig variant of itcas everywhere:
#   bash scripts/delete_method_results.sh itcas --quality efig --execute
#
#   # Delete all itcas variants for sphere2_6d only:
#   bash scripts/delete_method_results.sh itcas --problem sphere2_6d --execute
#
#   # Wipe moc_cas_hard everywhere, keep the summary dir intact:
#   bash scripts/delete_method_results.sh moc_cas_hard --keep-summary --execute

set -euo pipefail

# ── argument parsing ────────────────────────────────────────────────────────────

if [[ $# -eq 0 ]]; then
    sed -n '2,/^set /p' "$0" | grep -v '^set ' | sed 's/^# //' | sed 's/^#//'
    exit 1
fi

METHOD="$1"; shift

SWEEP_ROOT="results/sweep"
FILTER_QUALITY=""
FILTER_PROBLEM=""
FILTER_DIFF=""
KEEP_SUMMARY=0
EXECUTE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep-root)   SWEEP_ROOT="$2";       shift 2 ;;
        --quality)      FILTER_QUALITY="$2";   shift 2 ;;
        --problem)      FILTER_PROBLEM="$2";   shift 2 ;;
        --difficulty)   FILTER_DIFF="$2";      shift 2 ;;
        --keep-summary) KEEP_SUMMARY=1;        shift   ;;
        --execute)      EXECUTE=1;             shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$FILTER_QUALITY" && "$METHOD" != "itcas" ]]; then
    echo "ERROR: --quality is only valid for method 'itcas' (got '$METHOD')" >&2
    exit 1
fi

# ── validate ────────────────────────────────────────────────────────────────────

if [[ ! -d "$SWEEP_ROOT" ]]; then
    echo "ERROR: sweep root not found: $SWEEP_ROOT" >&2
    exit 1
fi

# ── collect targets ─────────────────────────────────────────────────────────────

METHOD_DIRS=()
PLOT_FILES=()

for prob_dir in "$SWEEP_ROOT"/*/; do
    prob=$(basename "$prob_dir")
    [[ "$prob" == _* ]] && continue
    [[ -n "$FILTER_PROBLEM" && "$prob" != "$FILTER_PROBLEM" ]] && continue

    for diff_dir in "$prob_dir"/*/; do
        [[ -d "$diff_dir" ]] || continue
        diff=$(basename "$diff_dir")
        [[ -n "$FILTER_DIFF" && "$diff" != "$FILTER_DIFF" ]] && continue

        if [[ -n "$FILTER_QUALITY" ]]; then
            # Target a single itcas quality variant: itcas/<quality>/
            method_dir="${diff_dir}itcas/${FILTER_QUALITY}"
        else
            method_dir="${diff_dir}${METHOD}"
        fi
        if [[ -d "$method_dir" ]]; then
            METHOD_DIRS+=("$method_dir")
        fi

        # Per-difficulty comparison plots are stale once any method is removed.
        while IFS= read -r -d '' f; do
            PLOT_FILES+=("$f")
        done < <(find "$diff_dir" -maxdepth 1 -name "*.pdf" -print0 2>/dev/null)
    done
done

SUMMARY_DIR="${SWEEP_ROOT}/_summary"
COMPLETION_FILE="${SWEEP_ROOT}/_completion.txt"

# ── report ──────────────────────────────────────────────────────────────────────

echo "Method  : $METHOD$( [[ -n "$FILTER_QUALITY" ]] && echo "/$FILTER_QUALITY" )"
echo "Root    : $SWEEP_ROOT"
[[ -n "$FILTER_PROBLEM" ]] && echo "Problem : $FILTER_PROBLEM (filtered)"
[[ -n "$FILTER_DIFF"    ]] && echo "Diff    : $FILTER_DIFF (filtered)"
echo "Mode    : $( [[ $EXECUTE -eq 1 ]] && echo EXECUTE || echo DRY-RUN )"
echo ""

TARGET_LABEL="$METHOD$( [[ -n "$FILTER_QUALITY" ]] && echo "/$FILTER_QUALITY" )"
if [[ ${#METHOD_DIRS[@]} -eq 0 ]]; then
    echo "No directories found for '$TARGET_LABEL' — nothing to delete."
    exit 0
fi

echo "── Method directories (${#METHOD_DIRS[@]}) ──────────────────────────────────"
for d in "${METHOD_DIRS[@]}"; do echo "  $d"; done
echo ""

echo "── Stale per-difficulty plots (${#PLOT_FILES[@]}) ──────────────────────────"
for f in "${PLOT_FILES[@]}"; do echo "  $f"; done
echo ""

if [[ $KEEP_SUMMARY -eq 0 ]]; then
    echo "── Shared summary (will be cleared) ─────────────────────────────────────"
    if [[ -d "$SUMMARY_DIR" ]]; then
        echo "  $SUMMARY_DIR/"
    else
        echo "  (not present)"
    fi
    if [[ -f "$COMPLETION_FILE" ]]; then
        echo "  $COMPLETION_FILE"
    fi
    echo ""
fi

# ── execute ─────────────────────────────────────────────────────────────────────

if [[ $EXECUTE -eq 0 ]]; then
    echo "DRY-RUN: nothing deleted. Re-run with --execute to apply."
    exit 0
fi

echo "Deleting method directories..."
for d in "${METHOD_DIRS[@]}"; do
    rm -rf "$d"
    echo "  removed  $d"
done

echo "Deleting stale per-difficulty plots..."
for f in "${PLOT_FILES[@]}"; do
    rm -f "$f"
    echo "  removed  $f"
done

if [[ $KEEP_SUMMARY -eq 0 ]]; then
    echo "Clearing shared summary..."
    if [[ -d "$SUMMARY_DIR" ]]; then
        rm -rf "$SUMMARY_DIR"
        echo "  removed  $SUMMARY_DIR/"
    fi
    if [[ -f "$COMPLETION_FILE" ]]; then
        rm -f "$COMPLETION_FILE"
        echo "  removed  $COMPLETION_FILE"
    fi
fi

echo ""
echo "Done. Re-run summarize jobs to regenerate plots and _summary/."
