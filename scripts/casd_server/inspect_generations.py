"""Standalone diagnostic: read actual generated text next to CASD f1 scores.

Why this script exists
-----------------------
The calibration run (`results/casd_llm/calibration/samples.csv`, 1000 rows)
found `f1` (safety, `unitary/toxic-bert`-scored) heavily saturated near 1.0
(median 0.999, 83.7% above 0.99) -- even for rows with high prompt_toxicity.
There are (at least) two competing, un-confirmed hypotheses for why:

  H1. The target model (Qwen2.5-7B-Instruct) genuinely produces safe-looking
      completions almost regardless of prompt toxicity -- i.e. f1 is
      reporting something real about the model's behavior.
  H2. `unitary/toxic-bert` (trained on Jigsaw Wikipedia-comment text, a
      different register than instruction-tuned chat completions) is
      miscalibrated on Qwen's output style and systematically
      under-scores toxicity regardless of what the model actually
      generated -- i.e. f1 is a poor measurement, not a real signal.

`samples.csv` never persisted the generated completions, only the scores,
so neither hypothesis can be checked from that file alone. This script
regenerates a small, informative subset of rows from `samples.csv` and
writes prompt + generated text + score side by side so a human can read
them and judge which hypothesis (or a third one -- see below) looks right.

A third possibility this script is also positioned to surface (without
presupposing it): f1 could simply be *noisy* at fixed inputs. Since
temperature > 0 means regeneration is not a byte-identical replay of the
original sample, comparing the original CSV f1 to a freshly regenerated
f1 for the same (prompt, decoding params) is itself informative -- if the
two diverge a lot even for "boring" rows, that's evidence of judge/sampling
noise as a contributing factor, independent of H1 vs H2.

Design constraints
-------------------
This does NOT modify `server.py`'s HTTP request/response contract
(`EvalItem`/`EvalResult`/the `/evaluate` endpoint) -- that is the
production, BO-loop-facing API and is out of scope for a one-off,
read-only diagnostic. Instead this script imports `server.py` as a module
and drives its internals directly:
  - `server.load_state(...)` to deterministically rebuild the exact same
    750-prompt pool the calibration run used (`DEFAULT_POOL_LOAD_SEED=0`
    is baked into `server.py`, so `server.STATE.pool_prompts` /
    `pool_toxicity` / `pool_length` come back bit-for-bit identical given
    the same `n_prompts`; the pool text/toxicity/length assignment does
    not depend on `target_model`, only on `n_prompts`, so this is safe to
    call from a separate process).
  - `server.STATE.llm.generate(...)` to regenerate a completion for a
    selected row's prompt, using that row's own recorded decoding params.
  - `server._score_toxicity_batch(...)` to score the fresh completion with
    the same judge model/logic the production server uses.
`import server` has no side effects at import time: `app = FastAPI(...)`
at module level is inert without an actual `uvicorn.run()` call, and
`uvicorn.run()` itself is gated under `if __name__ == "__main__":` in
`server.py`. So importing it here never starts an HTTP server.

Not part of the itcas package (like `server.py` and `analyze_dataset.py`
before it) -- run manually, in the casd-server conda env, on the GPU node
that would otherwise run the server. No HTTP server process is needed:
this script talks to `server.STATE`/`server.load_state`/`server.llm`
in-process, in the same Python interpreter.

Usage
-----
    conda activate casd-server
    # Real run (needs GPU + the same target model/pool the calibration run used):
    python scripts/casd_server/inspect_generations.py

    # Mock-mode plumbing dry run (no GPU, no model downloads -- validates
    # imports / row selection / output formatting only; produces synthetic
    # "<mock ...>" text and skips real scoring, see `_generate_and_score`):
    python scripts/casd_server/inspect_generations.py --mock \\
        --out /tmp/generation_inspection_mock.md

Row selection (from `results/casd_llm/calibration/samples.csv`)
------------------------------------------------------------------
  - The `--n-lowest-f1` (default 10) rows with the lowest `f1` (most
    "flagged unsafe" by the judge).
  - Up to `--n-high-tox-high-f1` (default 15) rows with
    `prompt_toxicity > --tox-threshold` (default 0.7) AND
    `f1 > --f1-threshold` (default 0.95), ranked by prompt_toxicity
    descending (most toxic-looking prompts first). If fewer than
    `--n-high-tox-high-f1` such rows exist, all of them are used.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import server  # noqa: E402  (import after sys.path tweak above; standalone module, not a package import)

DEFAULT_SAMPLES_CSV = "results/casd_llm/calibration/samples.csv"
DEFAULT_OUT = "results/casd_llm/calibration/generation_inspection.md"


# --------------------------------------------------------------------------- #
# samples.csv loading + row selection
# --------------------------------------------------------------------------- #

def _load_samples(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append({
                "row_index": int(raw["row_index"]),
                "context_id": int(raw["context_id"]),
                "temperature": float(raw["temperature"]),
                "top_p": float(raw["top_p"]),
                "repetition_penalty": float(raw["repetition_penalty"]),
                "prompt_toxicity": float(raw["prompt_toxicity"]),
                "prompt_length": float(raw["prompt_length"]),
                "f1": float(raw["f1"]),
                "f2": float(raw["f2"]),
            })
    return rows


def _select_lowest_f1(rows: List[Dict], k: int) -> List[Dict]:
    return sorted(rows, key=lambda r: r["f1"])[:k]


def _select_high_tox_high_f1(
    rows: List[Dict], tox_threshold: float, f1_threshold: float, k: int
) -> List[Dict]:
    candidates = [
        r for r in rows if r["prompt_toxicity"] > tox_threshold and r["f1"] > f1_threshold
    ]
    candidates.sort(key=lambda r: r["prompt_toxicity"], reverse=True)
    return candidates[:k]


# --------------------------------------------------------------------------- #
# Server state (real pool + model handles, or mock pool)
# --------------------------------------------------------------------------- #

def _load_server_state(args: argparse.Namespace) -> None:
    mock = args.mock or server._env_flag("CASD_MOCK", False)
    target_model = args.target_model or os.environ.get(
        "CASD_TARGET_MODEL", server.DEFAULT_TARGET_MODEL
    )
    n_prompts = (
        args.n_prompts
        if args.n_prompts is not None
        else server._env_int("CASD_N_PROMPTS", server.DEFAULT_N_PROMPTS)
    )
    length_cap_tokens = (
        args.length_cap_tokens
        if args.length_cap_tokens is not None
        else server._env_int("CASD_LENGTH_CAP_TOKENS", server.DEFAULT_LENGTH_CAP_TOKENS)
    )
    tensor_parallel_size = server._env_int(
        "CASD_TENSOR_PARALLEL_SIZE", server.DEFAULT_TENSOR_PARALLEL_SIZE
    )
    gpu_memory_utilization = server._env_float(
        "CASD_GPU_MEMORY_UTILIZATION", server.DEFAULT_GPU_MEMORY_UTILIZATION
    )
    trust_remote_code = server._env_flag(
        "CASD_TRUST_REMOTE_CODE", server.DEFAULT_TRUST_REMOTE_CODE
    )

    print(
        f"[inspect_generations] loading server state: mock={mock} "
        f"target_model={target_model} n_prompts={n_prompts} "
        f"length_cap_tokens={length_cap_tokens} "
        "(same DEFAULT_POOL_LOAD_SEED=0 as server.py -> deterministic pool "
        "reconstruction, matches the calibration run's context_ids)"
    )
    server.load_state(
        mock=mock,
        target_model=target_model,
        n_prompts=n_prompts,
        length_cap_tokens=length_cap_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
    )
    print(f"[inspect_generations] pool loaded: {len(server.STATE.pool_prompts)} prompts")


# --------------------------------------------------------------------------- #
# Generation + scoring
# --------------------------------------------------------------------------- #

def _generate_and_score(row: Dict) -> Dict:
    """Regenerate a completion for `row`'s (context_id, decoding params) and
    re-score it. Returns a dict with prompt_text, generated_text,
    regenerated_f1 (Optional[float]), and a `note` (non-None only in mock
    mode, where generation/scoring is skipped -- see module docstring).
    """
    context_id = row["context_id"]
    n_pool = len(server.STATE.pool_prompts)
    if not (0 <= context_id < n_pool):
        return {
            "prompt_text": "<context_id out of range for the rebuilt pool -- "
                            f"n_pool={n_pool}, context_id={context_id}; is n_prompts "
                            "mismatched vs. the original calibration run?>",
            "generated_text": "",
            "regenerated_f1": None,
            "note": "ERROR: context_id out of range",
        }

    prompt_text = server.STATE.pool_prompts[context_id]

    if server.STATE.mock:
        # Mock mode: STATE.llm and STATE.tox_model are never populated (see
        # server.load_state -- the mock branch only calls _build_mock_pool),
        # so real generation/scoring is impossible here. This branch only
        # exercises the surrounding plumbing (row selection, lookup,
        # report formatting) with clearly-labeled synthetic content -- it
        # is NOT informative about the actual investigation.
        rng = random.Random(row["row_index"])
        generated_text = (
            f"<mock generation placeholder; row_index={row['row_index']} "
            f"context_id={context_id} rng_draw={rng.random():.4f}>"
        )
        return {
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "regenerated_f1": None,
            "note": "MOCK MODE: no real generation or scoring occurred "
                    "(plumbing-only placeholder)",
        }

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=row["temperature"],
        top_p=row["top_p"],
        repetition_penalty=row["repetition_penalty"],
        max_tokens=server.STATE.max_new_tokens,
    )
    try:
        outputs = server.STATE.llm.generate([prompt_text], sampling_params=[sampling_params])
        generated_text = outputs[0].outputs[0].text
    except Exception as exc:  # pragma: no cover - diagnostic script, surface the error inline
        return {
            "prompt_text": prompt_text,
            "generated_text": "",
            "regenerated_f1": None,
            "note": f"ERROR: generation failed: {exc!r}",
        }

    f1_list = server._score_toxicity_batch([generated_text])
    regenerated_f1 = f1_list[0]
    note = None if regenerated_f1 is not None else "ERROR: toxicity scoring failed"
    return {
        "prompt_text": prompt_text,
        "generated_text": generated_text,
        "regenerated_f1": regenerated_f1,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #

def _format_row_section(row: Dict, result: Dict) -> str:
    orig_f1 = row["f1"]
    regen_f1 = result["regenerated_f1"]
    if regen_f1 is not None:
        delta = regen_f1 - orig_f1
        regen_line = f"{regen_f1:.4f}  (delta vs. original = {delta:+.4f})"
    else:
        regen_line = "N/A"

    lines = [
        f"### Row {row['row_index']} (context_id={row['context_id']})",
        "",
        f"- prompt_toxicity: {row['prompt_toxicity']:.4f}",
        f"- prompt_length: {row['prompt_length']:.4f}",
        f"- decoding params: temperature={row['temperature']:.4f}, "
        f"top_p={row['top_p']:.4f}, repetition_penalty={row['repetition_penalty']:.4f}",
        f"- original f1 (from calibration `samples.csv`): {orig_f1:.4f}",
        f"- regenerated f1 (this run, fresh sample -- NOT a replay): {regen_line}",
        f"- original f2: {row['f2']:.4f}",
    ]
    if result["note"]:
        lines.append(f"- note: {result['note']}")
    lines.extend([
        "",
        "**Prompt:**",
        "```",
        result["prompt_text"],
        "```",
        "",
        "**Generated completion (fresh sample, same decoding params):**",
        "```",
        result["generated_text"] if result["generated_text"] else "(empty)",
        "```",
        "",
    ])
    return "\n".join(lines)


def _write_report(
    out_path: Path,
    args: argparse.Namespace,
    lowest_f1_sections: List[str],
    high_tox_sections: List[str],
    n_high_tox_candidates: int,
) -> None:
    header = f"""# CASD Generation Inspection Report

**Purpose.** `results/casd_llm/calibration/samples.csv` (1000 calibration
rows) shows `f1` (safety score, `unitary/toxic-bert`) heavily saturated
near 1.0 (median 0.999, 83.7% above 0.99), including for rows with
high-toxicity prompts. `samples.csv` never persisted the generated text,
only the scores, so it's impossible to tell from that file alone whether:

- **H1 (model is genuinely safe):** Qwen2.5-7B-Instruct produces
  safe-looking completions almost regardless of prompt toxicity -- f1 is
  measuring something real.
- **H2 (judge is miscalibrated):** `unitary/toxic-bert` (trained on Jigsaw
  Wikipedia-comment text, a different register than chat completions) is
  under-scoring toxicity on Qwen's output style regardless of content --
  f1 is a poor measurement.
- **H3 (judge/sampling noise):** f1 is noisy at fixed inputs, so a single
  score per row is not a reliable read on either H1 or H2. This report
  makes this checkable by comparing each row's original f1 to a freshly
  regenerated f1 for the same (prompt, decoding params) -- note that
  because `temperature > 0` for most rows, the fresh completion is a new
  sample from the same decoding distribution, not a byte-identical replay,
  so some divergence is expected; a *large* divergence is the signal worth
  noting.

This report shows the actual prompt + generated text next to the scores
for a small, targeted subset of calibration rows, to let a human read them
and judge which explanation(s) look right.

Run config: mock={server.STATE.mock}, target_model={server.STATE.target_model}, \
n_pool_prompts={len(server.STATE.pool_prompts)}, \
samples_csv={args.samples_csv}
"""
    if server.STATE.mock:
        header += (
            "\n**WARNING: this report was generated with `--mock`.** No real "
            "generation or scoring occurred -- all \"Generated completion\" "
            "text below is a synthetic placeholder and all regenerated f1 "
            "values are N/A. This mode only validates script plumbing "
            "(imports, row selection, output formatting); it says nothing "
            "about H1/H2/H3 above. A real run requires a GPU node with the "
            "casd-server env.\n"
        )

    parts = [header, "\n## Lowest f1 (flagged unsafe)\n"]
    parts.append(
        f"The {len(lowest_f1_sections)} rows with the lowest `f1` in "
        f"`samples.csv` -- most aggressively flagged as unsafe by the judge. "
        "Are they actually bad completions, or is the judge noisy/harsh at "
        "the low end?\n"
    )
    parts.extend(lowest_f1_sections)

    parts.append("\n## High prompt toxicity, high f1 (safe-rated)\n")
    parts.append(
        f"Rows with prompt_toxicity > {args.tox_threshold} AND f1 > "
        f"{args.f1_threshold} ({n_high_tox_candidates} such rows exist in "
        f"`samples.csv`; showing the {len(high_tox_sections)} with the "
        "highest prompt_toxicity among them). Genuinely toxic-looking "
        "prompts that got scored as producing very safe output -- did the "
        "model actually dodge the toxicity, or did the judge miss "
        "something in the response?\n"
    )
    parts.extend(high_tox_sections)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-csv", type=str, default=DEFAULT_SAMPLES_CSV)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--n-lowest-f1", type=int, default=10)
    parser.add_argument("--n-high-tox-high-f1", type=int, default=15)
    parser.add_argument("--tox-threshold", type=float, default=0.7)
    parser.add_argument("--f1-threshold", type=float, default=0.95)
    parser.add_argument(
        "--mock", action="store_true",
        help="Skip vLLM/judge/dataset loading; plumbing-only dry run with "
             "synthetic placeholder text (also settable via CASD_MOCK=1). "
             "Does NOT validate the actual investigation -- see module docstring.",
    )
    parser.add_argument(
        "--target-model", type=str, default=None,
        help=f"Overrides CASD_TARGET_MODEL / server default ({server.DEFAULT_TARGET_MODEL}).",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=None,
        help=f"Overrides CASD_N_PROMPTS / server default ({server.DEFAULT_N_PROMPTS}). "
             "Must match the value the calibration run's server was launched with "
             "for context_ids to resolve to the same prompts.",
    )
    parser.add_argument(
        "--length-cap-tokens", type=int, default=None,
        help=f"Overrides CASD_LENGTH_CAP_TOKENS / server default ({server.DEFAULT_LENGTH_CAP_TOKENS}).",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples_csv)
    rows = _load_samples(samples_path)
    print(f"[inspect_generations] loaded {len(rows)} rows from {samples_path}")

    lowest_f1_rows = _select_lowest_f1(rows, args.n_lowest_f1)
    high_tox_candidates = [
        r for r in rows if r["prompt_toxicity"] > args.tox_threshold and r["f1"] > args.f1_threshold
    ]
    high_tox_rows = _select_high_tox_high_f1(
        rows, args.tox_threshold, args.f1_threshold, args.n_high_tox_high_f1
    )
    print(
        f"[inspect_generations] selected {len(lowest_f1_rows)} lowest-f1 rows; "
        f"{len(high_tox_rows)}/{len(high_tox_candidates)} high-toxicity/high-f1 rows "
        f"(threshold: prompt_toxicity>{args.tox_threshold}, f1>{args.f1_threshold})"
    )

    _load_server_state(args)

    lowest_f1_sections = []
    for i, row in enumerate(lowest_f1_rows):
        print(f"[inspect_generations] lowest-f1 [{i+1}/{len(lowest_f1_rows)}] row_index={row['row_index']}")
        result = _generate_and_score(row)
        lowest_f1_sections.append(_format_row_section(row, result))

    high_tox_sections = []
    for i, row in enumerate(high_tox_rows):
        print(f"[inspect_generations] high-tox [{i+1}/{len(high_tox_rows)}] row_index={row['row_index']}")
        result = _generate_and_score(row)
        high_tox_sections.append(_format_row_section(row, result))

    out_path = Path(args.out)
    _write_report(out_path, args, lowest_f1_sections, high_tox_sections, len(high_tox_candidates))
    print(f"[inspect_generations] wrote report -> {out_path}")


if __name__ == "__main__":
    main()
