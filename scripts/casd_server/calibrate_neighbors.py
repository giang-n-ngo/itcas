"""Calibrate the CASD neighborhood radius over the FULL RealToxicityPrompts
pool, and build the resulting (prompt -> neighbor indices) lookup table.

Standalone script, run in the `casd-server` conda env (needs `datasets`,
`transformers` for the target model's tokenizer, and `scipy` for the KD-tree
-- all CPU-only, no GPU/vLLM engine needed). Two phases:

1. Radius sweep: load every RealToxicityPrompts row with a non-null
   `prompt.toxicity` (~99k), compute (prompt_toxicity, prompt_length) exactly
   as `scripts/casd_server/server.py::_build_real_pool` does (same target
   tokenizer, same length cap), then for a grid of candidate radii report
   the neighbor-count distribution (mean/median/p90/max, fraction isolated
   i.e. 0 neighbors) so a radius can be picked.

2. Build: given `--radius` (from step 1, or a value you picked by eye from
   its printed table), compute the actual neighbor lookup table -- EVERY
   prompt's full list of neighbor indices within the radius (self excluded,
   no cap) -- and save it to `--out` as a single .npz cache in CSR format
   (`neighbor_flat` + `neighbor_offsets`). This is the "lookup table built
   in advance" the server loads at startup instead of redoing an
   O(n^2)-ish radius query on every boot. The number of neighbors actually
   USED per evaluation is a separate, server-side, runtime choice
   (`CASD_NEIGHBORS_PER_EVAL`, randomly sampled from this full list per
   query) -- not baked into this table, so the radius here can be picked
   for good coverage (few isolated prompts) without worrying about
   per-query evaluation cost.

Usage:
    # Phase 1: sweep to pick a radius
    python scripts/casd_server/calibrate_neighbors.py sweep \\
        --target-model Qwen/Qwen2.5-7B-Instruct --length-cap-tokens 128

    # Phase 2: build the lookup table at the chosen radius
    python scripts/casd_server/calibrate_neighbors.py build \\
        --radius 0.001 \\
        --out results/casd_llm/neighbor_calibration/neighbor_lookup.npz
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


DATASET_NAME = "allenai/real-toxicity-prompts"


def _load_pool_features(target_model: str, length_cap_tokens: int):
    """Returns (toxicity: float32[n], length: float32[n], texts: list[str])
    for every RealToxicityPrompts row with a non-null prompt.toxicity,
    computing `length` via the target model's own tokenizer exactly as
    `server.py::_build_real_pool` does (batched here for speed over ~99k
    rows, vs. that function's one-row-at-a-time loop -- same formula,
    `min(n_tokens / length_cap_tokens, 1.0)`, just faster to compute).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[calibrate_neighbors] loading {DATASET_NAME} ...", flush=True)
    ds = load_dataset(DATASET_NAME, split="train")
    ds = ds.filter(lambda row: row["prompt"]["toxicity"] is not None)
    n = len(ds)
    print(f"[calibrate_neighbors] {n} rows with non-null prompt.toxicity", flush=True)

    texts = [row["text"] for row in ds["prompt"]]
    toxicity = np.asarray([row["toxicity"] for row in ds["prompt"]], dtype=np.float32)

    print(f"[calibrate_neighbors] loading tokenizer for {target_model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(target_model)

    print(f"[calibrate_neighbors] tokenizing {n} prompts (batched) ...", flush=True)
    t0 = time.time()
    enc = tokenizer(texts, add_special_tokens=True)
    n_tokens = np.asarray([len(ids) for ids in enc["input_ids"]], dtype=np.float32)
    length = np.minimum(n_tokens / float(length_cap_tokens), 1.0).astype(np.float32)
    print(f"[calibrate_neighbors] tokenized in {time.time() - t0:.1f}s", flush=True)

    return toxicity, length, texts


def _neighbor_counts(points: np.ndarray, radius: float) -> np.ndarray:
    """Neighbor count per point (self excluded) within `radius`, via a
    KD-tree `query_pairs` bulk call (compiled, not a per-point Python loop)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    counts = np.zeros(len(points), dtype=np.int64)
    if len(pairs) > 0:
        np.add.at(counts, pairs[:, 0], 1)
        np.add.at(counts, pairs[:, 1], 1)
    return counts


def cmd_sweep(args: argparse.Namespace) -> None:
    toxicity, length, _texts = _load_pool_features(args.target_model, args.length_cap_tokens)
    points = np.stack([toxicity, length], axis=1)
    n = len(points)

    radii = [float(r) for r in args.radii.split(",")]
    print(f"\n[calibrate_neighbors] n={n} points in [0,1]^2 (toxicity, length)")
    print(f"{'radius':>10} {'mean':>8} {'median':>8} {'p90':>8} {'max':>8} {'%isolated':>10} {'%>5':>8}")
    rows = []
    for r in radii:
        counts = _neighbor_counts(points, r)
        row = {
            "radius": r,
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "p90": float(np.percentile(counts, 90)),
            "max": int(counts.max()),
            "pct_isolated": float((counts == 0).mean() * 100),
            "pct_over_5": float((counts > 5).mean() * 100),
        }
        rows.append(row)
        print(
            f"{r:>10.4f} {row['mean']:>8.2f} {row['median']:>8.1f} {row['p90']:>8.1f} "
            f"{row['max']:>8d} {row['pct_isolated']:>9.1f}% {row['pct_over_5']:>7.1f}%"
        )

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[calibrate_neighbors] wrote sweep table -> {args.out}")


def cmd_build(args: argparse.Namespace) -> None:
    """Build the FULL (uncapped) neighbor lookup table at `--radius`, stored
    in CSR (compressed sparse row) format: `neighbor_flat` is every prompt's
    neighbor indices concatenated end to end, `neighbor_offsets` (length
    n+1) marks where prompt i's slice starts/ends
    (`neighbor_flat[offsets[i]:offsets[i+1]]`). No neighbor cap is applied
    here -- unlike an earlier version of this table, which stored a fixed
    `max_neighbors`-per-prompt array and evaluated the WHOLE neighborhood at
    query time. That coupled "how many real prompts define a neighborhood"
    to "how many get evaluated per query", forcing a radius small enough to
    keep evaluation cost bounded (r=0.00011, mean 5 neighbors) at the cost of
    23% of prompts being fully isolated. Decoupling the two -- store the
    full neighborhood here, let the SERVER randomly sample a handful of it
    per query (`CASD_NEIGHBORS_PER_EVAL`, see server.py) -- allows a larger,
    more representative radius (e.g. r=0.001, mean ~32 neighbors, only 2.5%
    isolated) without inflating per-query evaluation cost.
    """
    toxicity, length, texts = _load_pool_features(args.target_model, args.length_cap_tokens)
    points = np.stack([toxicity, length], axis=1)
    n = len(points)

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    print(f"[calibrate_neighbors] querying radius={args.radius} neighbor lists for n={n} points ...", flush=True)
    t0 = time.time()
    neighbor_lists = tree.query_ball_point(points, r=args.radius)
    print(f"[calibrate_neighbors] queried in {time.time() - t0:.1f}s", flush=True)

    flat: list = []
    offsets = np.zeros(n + 1, dtype=np.int64)
    n_isolated = 0
    for i, idx_list in enumerate(neighbor_lists):
        idx = [j for j in idx_list if j != i]
        if not idx:
            n_isolated += 1
        flat.extend(idx)
        offsets[i + 1] = len(flat)
    neighbor_flat = np.asarray(flat, dtype=np.int64)
    n_neighbors = np.diff(offsets)

    print(
        f"[calibrate_neighbors] mean neighbors/point={n_neighbors.mean():.2f}, "
        f"median={float(np.median(n_neighbors)):.1f}, max={int(n_neighbors.max())}, "
        f"isolated (0 neighbors)={n_isolated} ({100 * n_isolated / n:.1f}%)"
    )

    np.savez_compressed(
        args.out,
        toxicity=toxicity,
        length=length,
        texts=np.array(texts, dtype=object),
        neighbor_flat=neighbor_flat,
        neighbor_offsets=offsets,
        n_neighbors=n_neighbors,
        radius=np.float64(args.radius),
        target_model=np.array(args.target_model),
        length_cap_tokens=np.int64(args.length_cap_tokens),
    )
    print(f"[calibrate_neighbors] wrote lookup table -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep", help="Sweep candidate radii and report neighbor-count stats.")
    sw.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    sw.add_argument("--length-cap-tokens", type=int, default=128)
    sw.add_argument(
        "--radii", type=str,
        default="0.002,0.005,0.008,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.07,0.1",
    )
    sw.add_argument("--out", type=str, default="results/casd_llm/neighbor_calibration/radius_sweep.json")
    sw.set_defaults(func=cmd_sweep)

    bd = sub.add_parser("build", help="Build and save the FULL (uncapped) neighbor lookup table at a chosen radius.")
    bd.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    bd.add_argument("--length-cap-tokens", type=int, default=128)
    bd.add_argument("--radius", type=float, required=True)
    bd.add_argument("--out", type=str, default="results/casd_llm/neighbor_calibration/neighbor_lookup.npz")
    bd.set_defaults(func=cmd_build)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
