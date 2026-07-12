#!/usr/bin/env bash
# Check whether every seed in each sweep slot evaluated exactly as many points
# as the budget currently configured for that (problem, difficulty) in
# configs/experiments.json. Sibling of check_completion.sh: same table shape,
# same column-discovery logic, same --all/--problems/--methods/N_SEEDS CLI,
# but each cell reports PASS/FAIL instead of a seed-completion count.
#
# check_completion.sh answers "how many seeds finished?" (a count out of
# N_SEEDS). This script instead answers "did every seed that finished actually
# evaluate the number of points experiments.json says it should have?" -- i.e.
# for each *.summary.json in results/sweep/<problem>/<difficulty>/<method>/, it
# computes evaluated = n_iters * eff_batch_size: n_iters is the number of
# iterations the run actually executed (one row in the run's .jsonl per
# iteration), and eff_batch_size is the TRUE per-run batch size the pipeline
# itself recorded for that run -- NOT guessed from the method/directory name.
# This matters because batch-ness cannot be inferred from naming alone: e.g.
# `itcas` (as opposed to `itcas_seq`) is a batch method by default even though
# its directory name has no "_batch" suffix. Some methods (the `_then_sample*`
# two-stage family) can legitimately evaluate FEWER unique points than
# n_iters * eff_batch_size in a given iteration when few candidates clear the
# interior-sampling PoF threshold -- that per-iteration shortfall is expected
# algorithm behavior, not an incomplete run, so it is intentionally not
# counted against a run here; what's checked is that the run executed the
# planned number of full-width iterations.
#
# The currently configured budget for that (problem, difficulty) is resolved
# via problems[problem][threshold_pct].budget -> problems[problem].defaults.budget
# -> top-level defaults.budget (threshold_pct is read from each run's own
# config.extra.threshold_pct, not guessed from the difficulty directory name).
# PASS requires every seed present in the slot to match that budget exactly.
# Any run that stopped early, was launched with a since-changed (stale)
# budget, or couldn't be parsed makes the whole slot FAIL.
#
# The method/quality column list is NOT hand-maintained here: it is queried
# from the actual Python registries (itcas.baselines.REGISTRY,
# pipeline.loop.CONTINUOUS_BASELINE_QUALITY/TWO_STAGE_BASE,
# algorithms.QUALITY_REGISTRY) every run, same as check_completion.sh. This is
# read-only introspection (no itcas/ source is touched); column headers are
# short aliases with a legend printed above the table, directory lookups still
# use the real, full method name.
#
# By default only the problems listed in configs/final_problems.json and the
# methods listed in configs/final_methods.json are reported (the curated
# benchmark set actually used for the paper/writeup); pass --all to report
# every problem directory found under SWEEP_ROOT and every method/quality
# discovered in the Python registries instead.
#
# Writes results/sweep/_min_evaluated.txt (does NOT touch _completion.txt).
#
# Usage: bash scripts/check_min_evaluated.sh [results/sweep] [N_SEEDS] [--all] [--problems <path>] [--methods <path>] [--experiments <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SWEEP_ROOT="results/sweep"
N_SEEDS="20"
SHOW_ALL=0
FINAL_PROBLEMS_PATH="${FINAL_PROBLEMS_PATH:-${REPO_ROOT}/configs/final_problems.json}"
FINAL_METHODS_PATH="${FINAL_METHODS_PATH:-${REPO_ROOT}/configs/final_methods.json}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${REPO_ROOT}/configs/experiments.json}"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)         SHOW_ALL=1; shift ;;
        --problems)    FINAL_PROBLEMS_PATH="$2"; shift 2 ;;
        --methods)     FINAL_METHODS_PATH="$2"; shift 2 ;;
        --experiments) EXPERIMENTS_PATH="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | grep -v '^set ' | sed 's/^# //;s/^#//'
            exit 0 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
[[ ${#POSITIONAL[@]} -ge 1 ]] && SWEEP_ROOT="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && N_SEEDS="${POSITIONAL[1]}"

OUT="${SWEEP_ROOT}/_min_evaluated.txt"

if [[ ! -f "$EXPERIMENTS_PATH" ]]; then
    echo "ERROR: experiments config not found: $EXPERIMENTS_PATH" >&2
    exit 2
fi

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
# Identical discovery logic to check_completion.sh -- see that file's comments
# for the reasoning behind each exclusion/alias. Duplicated verbatim (not
# imported/shared) so this script has no runtime dependency on
# check_completion.sh and stays a true drop-in sibling.
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

BASE_ALIAS = {
    "random": "rnd", "straddle": "strd", "cas_eci": "eci",
    "moc_cas_hard": "mcH", "moc_cas_soft": "mcS", "c2lse": "c2ls",
    "bes": "bes", "one_step": "one1", "ez": "ez", "eisr": "eisr",
}

def short(m: str) -> str:
    base, suf = m, ""
    if base.endswith("_batch"):
        base, suf = base[: -len("_batch")], ".b"
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
    rows.append((f"itcas_seq/{q}", f"is.{q[:3]}"))

def is_valid_two_stage(m: str) -> bool:
    try:
        return _parse_two_stage(m) is not None
    except ValueError:
        return False

if show_all:
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

# ── precompute PASS/FAIL per (problem, difficulty, method) slot ────────────────
#
# Done as ONE python pass over the whole sweep tree (not one subprocess per
# cell -- with ~4300 cells in --all mode that would mean 4300 python
# start-ups) that walks every *.summary.json once, groups by directory, and
# emits one PASS/FAIL line per slot. Parsing robustness mirrors
# check_budget_consistency.py's classify_run(): a file that fails to load, or
# is missing n_total/n_init/threshold_pct, or has non-numeric n_total/n_init,
# is counted as "unparseable" and makes the whole slot FAIL (we can't confirm
# every seed matched the budget if we couldn't read one of them) rather than
# crashing the whole run or silently ignoring it.
#
# PASS requires ALL of:
#   - at least one seed (summary.json) present in the slot
#   - every summary.json in the slot parsed cleanly
#   - the currently configured budget for (problem, threshold_pct) resolves
#     successfully from experiments.json
#   - every parsed seed's (n_iters * eff_batch_size) equals that budget exactly
# Anything else -> FAIL (missing data, stale/mismatched budget, early-stopped
# run, unparseable file, or unresolvable config).
declare -A STATUS_MAP=()
declare -A REASON_MAP=()
declare -A EXPECTED_MAP=()
declare -A NVALID_MAP=()
declare -A NSKIP_MAP=()

while IFS=$'\t' read -r prob diff method status reason expected nvalid nskip; do
    key="${prob}"$'\x1f'"${diff}"$'\x1f'"${method}"
    STATUS_MAP["$key"]="$status"
    REASON_MAP["$key"]="$reason"
    EXPECTED_MAP["$key"]="$expected"
    NVALID_MAP["$key"]="$nvalid"
    NSKIP_MAP["$key"]="$nskip"
done < <(python3 - "$SWEEP_ROOT" "$EXPERIMENTS_PATH" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

sweep_root = Path(sys.argv[1])
experiments_path = Path(sys.argv[2])
SUMMARY_SUFFIX = ".summary.json"

experiments_cfg = json.loads(experiments_path.read_text())


def resolve_expected_budget(problem: str, difficulty_key: str):
    """problems[problem][difficulty_key].budget -> problems[problem].defaults.budget
    -> top-level defaults.budget. Returns None if unresolvable."""
    problems = experiments_cfg.get("problems", {})
    top_defaults = experiments_cfg.get("defaults", {})

    prob_cfg = problems.get(problem)
    if prob_cfg is None:
        return None

    diff_cfg = prob_cfg.get(difficulty_key)
    if isinstance(diff_cfg, dict) and "budget" in diff_cfg:
        return diff_cfg["budget"]

    prob_defaults = prob_cfg.get("defaults", {})
    if "budget" in prob_defaults:
        return prob_defaults["budget"]

    return top_defaults.get("budget")


# slot key = (problem_dir, difficulty_dir, method)
valid = defaultdict(list)      # evaluated-point counts from files that parsed cleanly
skipped = defaultdict(int)     # count of unparseable files in that slot
thresholds = defaultdict(set)  # distinct threshold_pct strings seen among parsed files

if sweep_root.is_dir():
    for p in sweep_root.rglob(f"*{SUMMARY_SUFFIX}"):
        try:
            rel_parts = p.relative_to(sweep_root).parts
        except ValueError:
            continue
        if len(rel_parts) < 4:
            # not inside <problem>/<difficulty>/<method>/<file> (at least one
            # method-dir level is required; degenerate paths are skipped
            # rather than mistaking the filename itself for a method).
            continue
        problem_dir, difficulty_dir = rel_parts[0], rel_parts[1]
        method = "/".join(rel_parts[2:-1])
        key = (problem_dir, difficulty_dir, method)

        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            skipped[key] += 1
            continue

        n_iters = data.get("n_iters")
        eff_batch_size = data.get("eff_batch_size")
        config = data.get("config") if isinstance(data.get("config"), dict) else {}
        extra = config.get("extra") if isinstance(config.get("extra"), dict) else {}
        threshold_pct = extra.get("threshold_pct")

        if n_iters is None or eff_batch_size is None or threshold_pct is None:
            skipped[key] += 1
            continue
        try:
            evaluated = int(n_iters) * int(eff_batch_size)
        except (TypeError, ValueError):
            skipped[key] += 1
            continue

        valid[key].append(evaluated)
        thresholds[key].add(str(threshold_pct))

all_keys = set(valid) | set(skipped)
for key in all_keys:
    problem_dir, difficulty_dir, method = key
    vals = valid.get(key, [])
    nskip = skipped.get(key, 0)
    tset = thresholds.get(key, set())

    status = "FAIL"
    expected = "NA"
    if not vals:
        reason = "no_data"
    elif nskip > 0:
        reason = "unparseable"
    elif len(tset) != 1:
        # Seeds in the same slot disagree on threshold_pct -- shouldn't
        # happen (same directory = same difficulty by construction), but
        # don't silently pick one if it does.
        reason = "mixed_threshold"
    else:
        threshold_key = next(iter(tset))
        budget = resolve_expected_budget(problem_dir, threshold_key)
        if budget is None:
            reason = "no_config"
        else:
            expected = budget
            if all(v == budget for v in vals):
                status = "PASS"
                reason = "ok"
            else:
                reason = "mismatch"

    print(f"{problem_dir}\t{difficulty_dir}\t{method}\t{status}\t{reason}\t{expected}\t{len(vals)}\t{nskip}")
PY
)

# ── helpers ────────────────────────────────────────────────────────────────────

# Format one cell: "PASS" or "FAIL" (a missing MAP entry -- i.e. zero
# summary.json files found at all for that slot -- defaults to FAIL, since
# there is no data to confirm every seed matched the budget).
#
# Deliberately ASCII-only, matching check_completion.sh's fmt_cell -- bash's
# builtin `printf "%-Ns"` pads by BYTE count, not display width.
fmt_cell() {
    local status="$1"
    printf "%-${CW}s " "${status}"
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
    echo "ITCAS Sweep Budget Pass/Fail -- $(date '+%Y-%m-%d %H:%M')"
    echo "Root        : $(cd "$SWEEP_ROOT" && pwd)"
    echo "Experiments : ${EXPERIMENTS_PATH}"
    echo "Seeds       : $N_SEEDS expected per slot (informational only -- PASS/FAIL"
    echo "              below checks every seed actually present, not a seed count;"
    echo "              see check_completion.sh / _completion.txt for that)"
    if [[ "$SHOW_ALL" -eq 1 ]]; then
        echo "Problems: ALL found under root (--all)"
        echo "Methods : ALL discovered from registries (--all)"
    else
        echo "Problems: ${#FINAL_PROBLEMS[@]} from ${FINAL_PROBLEMS_PATH} (pass --all for every problem on disk)"
        echo "Methods : ${N_COLS} from ${FINAL_METHODS_PATH} (pass --all for every method on disk)"
    fi
    echo "Legend : PASS every seed present ran (n_iters * eff_batch_size) exactly"
    echo "              matching the budget currently configured for"
    echo "              (problem, difficulty) in experiments.json -- eff_batch_size"
    echo "              is each run's own recorded batch size, not guessed from name"
    echo "         FAIL no seeds present, a stale/mismatched budget, an early-stopped"
    echo "              run, an unparseable summary file, or no resolvable budget"
    echo ""
    print_legend
    echo ""

    repchar "$SEP_LEN" "-"; echo ""
    printf "%-${PW}s %-${DW}s " "PROBLEM" "DIFF"
    for lbl in "${LABELS[@]}"; do printf "%-${CW}s " "$lbl"; done
    echo ""
    repchar "$SEP_LEN" "-"; echo ""

    local total_slots=0 total_pass=0 total_fail=0
    local fail_no_data=0 fail_mismatch=0 fail_unparseable=0 fail_no_config=0 fail_mixed=0

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
                local key="${prob}"$'\x1f'"${diff}"$'\x1f'"${d}"
                local status="${STATUS_MAP[$key]:-FAIL}"
                local reason="${REASON_MAP[$key]:-no_data}"

                total_slots=$(( total_slots + 1 ))
                if [[ "$status" == "PASS" ]]; then
                    total_pass=$(( total_pass + 1 ))
                else
                    total_fail=$(( total_fail + 1 ))
                    case "$reason" in
                        no_data)       fail_no_data=$(( fail_no_data + 1 )) ;;
                        mismatch)      fail_mismatch=$(( fail_mismatch + 1 )) ;;
                        unparseable)   fail_unparseable=$(( fail_unparseable + 1 )) ;;
                        no_config)     fail_no_config=$(( fail_no_config + 1 )) ;;
                        mixed_threshold) fail_mixed=$(( fail_mixed + 1 )) ;;
                    esac
                fi
                fmt_cell "$status"
            done

            echo ""
        done
        echo ""
    done

    repchar "$SEP_LEN" "-"; echo ""
    echo "TOTAL: ${total_pass}/${total_slots} slots PASS  |  ${total_fail} FAIL" \
         "(${fail_no_data} no data, ${fail_mismatch} budget mismatch," \
         "${fail_unparseable} unparseable, ${fail_no_config} no resolvable budget," \
         "${fail_mixed} mixed threshold_pct)"
}

generate_report | tee "$OUT"
echo "" >&2
echo "Written: $OUT" >&2
