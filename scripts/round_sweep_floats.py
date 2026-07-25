"""One-time migration: re-serialize existing results/sweep JSONL/summary.json
files with floats rounded to itcas.io.logger._SIG_FIGS significant figures.

Rewrites files in place (atomically, one at a time via a temp file + os.replace)
so a run interrupted partway through never leaves a half-written file. Safe to
re-run: files already at the target precision round-trip unchanged.

Usage:
    python scripts/round_sweep_floats.py [--root results/sweep] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itcas.io.logger import _to_jsonable  # noqa: E402


def _rewrite_jsonl(path: str, dry_run: bool) -> tuple[int, int]:
    with open(path, "r") as f:
        lines = f.readlines()
    orig_size = sum(len(l.encode("utf-8")) for l in lines)
    out_lines = []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        rec = json.loads(line)
        out_lines.append(json.dumps(_to_jsonable(rec)))
    new_text = "".join(l + "\n" for l in out_lines)
    new_size = len(new_text.encode("utf-8"))
    if not dry_run:
        _atomic_write(path, new_text)
    return orig_size, new_size


def _rewrite_summary(path: str, dry_run: bool) -> tuple[int, int]:
    with open(path, "r") as f:
        text = f.read()
    orig_size = len(text.encode("utf-8"))
    rec = json.loads(text)
    new_text = json.dumps(_to_jsonable(rec), indent=2)
    new_size = len(new_text.encode("utf-8"))
    if not dry_run:
        _atomic_write(path, new_text)
    return orig_size, new_size


def _atomic_write(path: str, text: str) -> None:
    out_dir = os.path.dirname(path)
    orig_mode = os.stat(path).st_mode
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".round.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, orig_mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sweep")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_orig = 0
    total_new = 0
    n_files = 0
    n_errors = 0
    for dirpath, _dirnames, filenames in os.walk(args.root):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            try:
                if fn.endswith(".jsonl"):
                    o, n = _rewrite_jsonl(path, args.dry_run)
                elif fn.endswith(".summary.json"):
                    o, n = _rewrite_summary(path, args.dry_run)
                else:
                    continue
            except Exception as e:
                n_errors += 1
                print(f"ERROR {path}: {e}", file=sys.stderr)
                continue
            total_orig += o
            total_new += n
            n_files += 1
            if n_files % 5000 == 0:
                print(f"...{n_files} files processed", file=sys.stderr)

    pct = 100 * total_new / total_orig if total_orig else 100.0
    mode = "[dry-run] " if args.dry_run else ""
    print(
        f"{mode}{n_files} files, {n_errors} errors: "
        f"{total_orig/1e6:.1f}MB -> {total_new/1e6:.1f}MB ({pct:.1f}%)"
    )


if __name__ == "__main__":
    main()
