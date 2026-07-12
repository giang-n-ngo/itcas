#!/usr/bin/env bash
# Check sweep seed completion across all problems/difficulties/methods.
# Every (problem, difficulty, method/quality) slot is expected to have N_SEEDS
# completed runs. Missing directories and zero-seed slots both show as !0/N.
# Writes _completion.txt at the top of the sweep folder (underscore sorts first).
#
# The method/quality column list is NOT hand-maintained here: it is queried
# from the actual Python registries (itcas.baselines.REGISTRY,
# pipeline.loop.CONTINUOUS_BASELINE_QUALITY/TWO_STAGE_BASE,
# algorithms.QUALITY_REGISTRY) every run, the same "shell out to Python for
# canonical state" pattern run_seedset.sh already uses for compute_pending().
# This is read-only introspection (no itcas/ source is touched) so the report
# covers every baseline automatically and never goes stale when the Scientific
# Coder adds/removes one. Column headers are short aliases (full names are
# often 20+ chars, e.g. straddle_then_sample_batch) with a legend printed
# above the table; directory lookups still use the real, full method name.
#
# By default only the problems listed in configs/final_problems.json and the
# methods listed in configs/final_methods.json are reported (the curated
# benchmark set and method list actually used for the paper/writeup); pass
# --all to report every problem directory found under SWEEP_ROOT and every
# method/quality discovered in the Python registries instead (useful while
# exploratory/ablation runs or not-yet-finalized methods are still on disk).
#
# Usage: bash scripts/check_completion.sh [results/sweep] [N_SEEDS] [--all] [--problems <path>] [--methods <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SWEEP_ROOT="results/sweep"
N_SEEDS="20"
SHOW_ALL=0
FINAL_PROBLEMS_PATH="${FINAL_PROBLEMS_PATH:-${REPO_ROOT}/configs/final_problems.json}"
FINAL_METHODS_PATH="${FINAL_METHODS_PATH:-${REPO_ROOT}/configs/final_methods.json}"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)      SHOW_ALL=1; shift ;;
        --problems) FINAL_PROBLEMS_PATH="$2"; shift 2 ;;
        --methods)  FINAL_METHODS_PATH="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | grep -v '^set ' | sed 's/^# //;s/^#//'
            exit 0 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
[[ ${#POSITIONAL[@]} -ge 1 ]] && SWEEP_ROOT="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && N_SEEDS="${POSITIONAL[1]}"

OUT="${SWEEP_ROOT}/_completion.txt"

# ── resolve the problem filter ──────────────────────────────────────────────────

declare -A FINAL_PROBLEM_SET=()
if [[ "$SHOW_ALL" -eq 0 ]]; then
    if [[ ! -f "$FINAL_PROBLEMS_PATH" ]]; then
        echo "ERROR: final-problems list not found: $FINAL_PROBLEMS_PATH" >&2
        echo "       (pass --all to report every problem on disk instead, or --problems <path>)" >&2
        exit 2
    fi
    mapfile -t FINAL_PROBLEMS < <(python3 -c "
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['problems']))
" "$FINAL_PROBLEMS_PATH")
    for p in "${FINAL_PROBLEMS[@]}"; do FINAL_PROBLEM_SET["$p"]=1; done
fi

if [[ "$SHOW_ALL" -eq 0 && ! -f "$FINAL_METHODS_PATH" ]]; then
    echo "ERROR: final-methods list not found: $FINAL_METHODS_PATH" >&2
    echo "       (pass --all to report every method on disk instead, or --methods <path>)" >&2
    exit 2
fi

# ── discover columns: (full directory name, short header label) pairs ──────────
#
# Excluded on purpose:
#   - eps_constraint / moo_cluster: raise NotImplementedError, can never
#     produce a completed run.
#   - interior_sampling(_batch): the Family-C Stage-2 helper baseline; it is
#     only ever invoked internally by <base>_then_sample, never run standalone.
#   - c2lse / bes / interior_sampling / cr_ndig out of QUALITY_REGISTRY:
#     registered there only so they can reuse the continuous multistart/
#     QD-DPP machinery (see algorithms/quality.py), not meant as
#     `itcas`/`itcas_seq --quality` choices -- cr_ndig is already its own
#     top-level column via CONTINUOUS_BASELINE_QUALITY below.
#
# The full set above is discovered from the Python registries every run so it
# never goes stale; by default it is then filtered down to (and ordered by)
# the curated list in configs/final_methods.json, the same way problems are
# filtered against configs/final_problems.json. Pass --all to skip this
# filter and report every discovered method/quality instead.
mapfile -t COLUMNS < <(python3 - "$SHOW_ALL" "$FINAL_METHODS_PATH" "$SWEEP_ROOT" <<'PY'
import json, sys
from pathlib import Path

from itcas.baselines import REGISTRY as BASELINES
from itcas.pipeline.loop import CONTINUOUS_BASELINE_QUALITY, TWO_STAGE_BASE, _parse_two_stage
from itcas.algorithms import QUALITY_REGISTRY

show_all = sys.argv[1] == "1"
final_methods_path = sys.argv[2]
sweep_root = sys.argv[3]

EXCLUDE_METHODS = {
    "eps_constraint", "moo_cluster",
    "interior_sampling", "interior_sampling_batch",
}
methods = set(BASELINES) - EXCLUDE_METHODS
for base in CONTINUOUS_BASELINE_QUALITY:
    methods.add(base)
    methods.add(base + "_batch")
# NOTE: Family-C two-stage methods (TWO_STAGE_BASE's keys, e.g.
# "straddle_then_sample") are deliberately NOT added here anymore -- every
# such method now requires an explicit, unbounded `_lseNN` infix (see
# _parse_two_stage), so the bare names are no longer valid columns and the
# actual NN values in use can't be enumerated from the registry alone. They
# are instead discovered below: from configs/final_methods.json in curated
# mode, or from the directories actually on disk under --all.

# Short, stable aliases for the handful of "base" acquisition names; anything
# unrecognized (a brand-new baseline family) still gets a usable 4-char
# fallback instead of breaking, so this never needs to be hand-updated to
# avoid an ugly label -- only to make a NEW label prettier.
BASE_ALIAS = {
    "random": "rnd", "straddle": "strd", "cas_eci": "eci",
    "moc_cas_hard": "mcH", "moc_cas_soft": "mcS", "c2lse": "c2ls",
    "bes": "bes", "one_step": "one1", "ez": "ez", "eisr": "eisr",
}

def short(m: str) -> str:
    base, suf = m, ""
    if base.endswith("_batch"):
        base, suf = base[: -len("_batch")], ".b"
    # Family-C two-stage: strip the mandatory _lseNN infix into the suffix
    # (e.g. "straddle_then_sample_lse10" -> base="straddle_then_sample",
    # suf=".ts10"; with the _batch suffix already stripped above, so
    # "straddle_then_sample_lse10_batch" -> ".ts10b").
    for tsbase in TWO_STAGE_BASE:
        prefix = tsbase + "_lse"
        if base.startswith(prefix) and base[len(prefix):].isdigit():
            pct = base[len(prefix):]
            base, suf = tsbase[: -len("_then_sample")], f".ts{pct}" + (".b" if suf else "")
            break
    return BASE_ALIAS.get(base, base[:4]) + suf

rows = [(m, short(m)) for m in sorted(methods)]

EXCLUDE_QUALITIES = {"c2lse", "bes", "interior_sampling", "cr_ndig"}
for q in sorted(set(QUALITY_REGISTRY) - EXCLUDE_QUALITIES):
    rows.append((f"itcas/{q}", f"i.{q[:3]}"))
    # itcas_seq shares the same cfg.quality machinery as itcas (see
    # pipeline/loop.py's `base in ("itcas", "itcas_seq")` checks) but is
    # always forced to batch_size=1, so every quality variant gets a
    # sequential column too.
    rows.append((f"itcas_seq/{q}", f"is.{q[:3]}"))

def is_valid_two_stage(m: str) -> bool:
    try:
        return _parse_two_stage(m) is not None
    except ValueError:
        return False

if show_all:
    # Discover whichever concrete _lseNN two-stage variants actually exist on
    # disk (NN is unbounded, so this can't come from the registry alone).
    found = set()
    for method_dir in Path(sweep_root).glob("*/*/*"):
        if method_dir.is_dir() and is_valid_two_stage(method_dir.name):
            found.add(method_dir.name)
    rows.extend((m, short(m)) for m in sorted(found))
else:
    with open(final_methods_path) as f:
        wanted = json.load(f)["methods"]
    by_name = dict(rows)
    unknown = []
    curated_rows = []
    for m in wanted:
        if m in by_name:
            curated_rows.append((m, by_name[m]))
        elif is_valid_two_stage(m):
            curated_rows.append((m, short(m)))
        else:
            unknown.append(m)
    if unknown:
        sys.exit(
            f"ERROR: {final_methods_path} lists unknown method(s) not found "
            f"in the Python registries: {unknown}"
        )
    rows = curated_rows

for m, s in rows:
    print(f"{m} {s}")
PY
)

if [[ ${#COLUMNS[@]} -eq 0 ]]; then
    echo "ERROR: failed to discover method/quality columns (see Python error above)" >&2
    exit 2
fi

DIRS=()
LABELS=()
for pair in "${COLUMNS[@]}"; do
    DIRS+=("${pair%% *}")
    LABELS+=("${pair##* }")
done
N_COLS=${#DIRS[@]}

PW=32   # problem name column width
DW=8    # difficulty column width

# Cell/legend widths are sized off the actual discovered labels/names rather
# than hardcoded, since bash's `printf "%-Ns"` only pads short strings -- it
# never truncates long ones -- so any label longer than a fixed guess (e.g. a
# two-stage alias like "strd.ts100b", or a full name like
# "straddle_then_sample_lse100_batch") would silently shift every column
# after it out of alignment instead of just looking a bit wide.
CW=8    # each method/quality cell width (labels are short aliases now)
for lbl in "${LABELS[@]}"; do
    (( ${#lbl} + 1 > CW )) && CW=$(( ${#lbl} + 1 ))
done

LEGEND_LW=8   # legend "short label" column width
LEGEND_NW=28  # legend "full name" column width
for lbl in "${LABELS[@]}"; do
    (( ${#lbl} + 2 > LEGEND_LW )) && LEGEND_LW=$(( ${#lbl} + 2 ))
done
for d in "${DIRS[@]}"; do
    (( ${#d} + 2 > LEGEND_NW )) && LEGEND_NW=$(( ${#d} + 2 ))
done

SEP_LEN=$(( PW + 1 + DW + 1 + (CW + 1) * N_COLS ))

# ── helpers ────────────────────────────────────────────────────────────────────

count_summaries() {
    local dir="$1"
    [[ -d "$dir" ]] || { echo 0; return; }
    find "$dir" -maxdepth 1 -name "*.summary.json" 2>/dev/null | wc -l
}

# Format one cell. Every slot is expected so 0 is always flagged.
#   DONE all done   ~N/N partial   !0/N missing/not started
#
# Deliberately ASCII-only: bash's builtin `printf "%-Ns"` pads by BYTE count,
# not display width, so a multi-byte UTF-8 glyph (e.g. a checkmark) is
# miscounted as several columns wide and every such cell ends up short-padded,
# breaking table alignment. Plain ASCII text has no such discrepancy.
fmt_cell() {
    local done="$1"
    local text
    if   [[ "$done" -eq "$N_SEEDS" ]]; then text="DONE"
    elif [[ "$done" -eq 0 ]];          then text="!${done}/${N_SEEDS}"
    else                                    text="~${done}/${N_SEEDS}"
    fi
    printf "%-${CW}s " "${text}"
}

repchar() { printf '%*s' "$1" '' | tr ' ' "$2"; }

print_legend() {
    echo "Columns (short label -> full method/quality, $N_COLS total):"
    local per_line=0
    for i in "${!DIRS[@]}"; do
        printf "  %-${LEGEND_LW}s= %-${LEGEND_NW}s" "${LABELS[$i]}" "${DIRS[$i]}"
        per_line=$(( per_line + 1 ))
        if (( per_line % 3 == 0 )); then echo ""; fi
    done
    if (( per_line % 3 != 0 )); then echo ""; fi
}

# ── report ─────────────────────────────────────────────────────────────────────

generate_report() {
    echo "ITCAS Sweep Completion -- $(date '+%Y-%m-%d %H:%M')"
    echo "Root   : $(cd "$SWEEP_ROOT" && pwd)"
    echo "Seeds  : $N_SEEDS expected per slot"
    if [[ "$SHOW_ALL" -eq 1 ]]; then
        echo "Problems: ALL found under root (--all)"
        echo "Methods : ALL discovered from registries (--all)"
    else
        echo "Problems: ${#FINAL_PROBLEMS[@]} from ${FINAL_PROBLEMS_PATH} (pass --all for every problem on disk)"
        echo "Methods : ${N_COLS} from ${FINAL_METHODS_PATH} (pass --all for every method on disk)"
    fi
    echo "Legend : DONE all done   ~N/N partial   !0/N missing or not started"
    echo ""
    print_legend
    echo ""

    repchar "$SEP_LEN" "-"; echo ""
    printf "%-${PW}s %-${DW}s " "PROBLEM" "DIFF"
    for lbl in "${LABELS[@]}"; do printf "%-${CW}s " "$lbl"; done
    echo ""
    repchar "$SEP_LEN" "-"; echo ""

    local total_slots=0 total_full=0 total_partial=0 total_missing=0

    for prob_dir in "$SWEEP_ROOT"/*/; do
        local prob; prob=$(basename "$prob_dir")
        [[ "$prob" == _* ]] && continue
        [[ -d "$prob_dir" ]] || continue
        if [[ "$SHOW_ALL" -eq 0 && -z "${FINAL_PROBLEM_SET[$prob]:-}" ]]; then
            continue
        fi

        local first=1
        for diff_dir in "$prob_dir"/*/; do
            [[ -d "$diff_dir" ]] || continue
            local diff; diff=$(basename "$diff_dir")

            local plabel=""
            [[ "$first" -eq 1 ]] && { plabel="$prob"; first=0; }
            printf "%-${PW}s %-${DW}s " "$plabel" "$diff"

            for d in "${DIRS[@]}"; do
                local done; done=$(count_summaries "${diff_dir}${d}")
                total_slots=$(( total_slots + 1 ))
                if   [[ "$done" -eq "$N_SEEDS" ]]; then total_full=$(( total_full + 1 ))
                elif [[ "$done" -eq 0 ]];          then total_missing=$(( total_missing + 1 ))
                else                                    total_partial=$(( total_partial + 1 ))
                fi
                fmt_cell "$done"
            done

            echo ""
        done
        echo ""
    done

    repchar "$SEP_LEN" "-"; echo ""
    echo "TOTAL: ${total_full}/${total_slots} slots fully done  |  ${total_partial} partial  |  ${total_missing} missing"
}

generate_report | tee "$OUT"
echo "" >&2
echo "Written: $OUT" >&2
