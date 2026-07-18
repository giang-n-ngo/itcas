"""One-off analysis of the cached RealToxicityPrompts dataset used by the CASD
evaluator server (see scripts/casd_server/server.py). NOT part of the itcas
package or the server's runtime path -- run manually, in the casd-server env
(needs `datasets` + `matplotlib`; `pip install matplotlib` once into that env
if missing, it's not in requirements-casd-server.txt since nothing at server
runtime plots anything).

For each of the 16 per-example scores (Perspective API sub-scores: toxicity,
severe_toxicity, profanity, sexually_explicit, identity_attack, insult,
threat, flirtation -- reported separately for `prompt` and `continuation`),
writes one histogram PNG plus one row of summary statistics.

Usage:
    conda activate casd-server
    python scripts/casd_server/analyze_dataset.py
    python scripts/casd_server/analyze_dataset.py --out-dir results/casd_llm/dataset_stats
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

DATASET_NAME = "allenai/real-toxicity-prompts"
SCORE_FIELDS = [
    "toxicity",
    "severe_toxicity",
    "profanity",
    "sexually_explicit",
    "identity_attack",
    "insult",
    "threat",
    "flirtation",
]
SIDES = ["prompt", "continuation"]

# dataviz skill reference palette: categorical slot 1 / sequential base hue.
BAR_COLOR = "#2a78d6"
MEAN_COLOR = "#e34948"    # slot 8 (red) -- distinguishable reference line, not a 2nd series
GRID_COLOR = "#d8d7d2"    # recessive gridlines
TEXT_COLOR = "#0b0b0b"


def _stats(values: np.ndarray, n_total: int) -> dict:
    finite = values[~np.isnan(values)]
    n_missing = n_total - finite.size
    if finite.size == 0:
        return {
            "n_total": n_total, "n_present": 0, "n_missing": n_missing,
            "mean": float("nan"), "std": float("nan"), "min": float("nan"),
            "p25": float("nan"), "median": float("nan"), "p75": float("nan"),
            "max": float("nan"),
        }
    return {
        "n_total": n_total,
        "n_present": int(finite.size),
        "n_missing": int(n_missing),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "max": float(np.max(finite)),
    }


def _plot_histogram(values: np.ndarray, field_label: str, stats: dict, out_path: Path) -> None:
    finite = values[~np.isnan(values)]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    ax.hist(finite, bins=50, range=(0.0, 1.0), color=BAR_COLOR, edgecolor="none")
    ax.axvline(stats["mean"], color=MEAN_COLOR, linewidth=1.5, linestyle="--")
    ax.text(
        stats["mean"], ax.get_ylim()[1] * 0.96, f"  mean={stats['mean']:.3f}",
        color=MEAN_COLOR, fontsize=8, va="top", ha="left",
    )

    ax.set_title(field_label, color=TEXT_COLOR, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("score [0, 1]", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("count", color=TEXT_COLOR, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    n_missing = stats["n_missing"]
    if n_missing:
        ax.text(
            0.99, 0.96, f"{n_missing} missing", transform=ax.transAxes,
            color="#787770", fontsize=8, va="top", ha="right",
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default="results/casd_llm/dataset_stats")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    hist_dir = out_dir / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze_dataset] loading {DATASET_NAME} split={args.split} (offline/cached)...")
    ds = load_dataset(DATASET_NAME, split=args.split)
    n_total = len(ds)
    print(f"[analyze_dataset] n_rows={n_total}")

    rows = []
    for side in SIDES:
        for field in SCORE_FIELDS:
            values = np.array(
                [r if r is not None else np.nan for r in ds[side][field]], dtype=np.float64
            )
            field_label = f"{side}.{field}"
            s = _stats(values, n_total)
            rows.append({"field": field_label, **s})
            png_name = f"{side}_{field}.png"
            _plot_histogram(values, field_label, s, hist_dir / png_name)
            print(f"[analyze_dataset]   {field_label:32s} mean={s['mean']:.4f} missing={s['n_missing']}")

    n_challenging = int(sum(1 for c in ds["challenging"] if c))
    rows.append({
        "field": "challenging (fraction True)",
        "n_total": n_total, "n_present": n_total, "n_missing": 0,
        "mean": n_challenging / n_total, "std": float("nan"), "min": float("nan"),
        "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan"),
    })

    summary_path = out_dir / "dataset_summary.csv"
    fieldnames = ["field", "n_total", "n_present", "n_missing", "mean", "std", "min", "p25", "median", "p75", "max"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[analyze_dataset] wrote {len(rows)} summary rows -> {summary_path}")
    print(f"[analyze_dataset] wrote {len(SIDES) * len(SCORE_FIELDS)} histograms -> {hist_dir}/")


if __name__ == "__main__":
    main()
