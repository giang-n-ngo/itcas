"""Structured logging of per-iteration experiment results.

Writes a JSONL file (one line per iteration) plus a summary JSON when
`finalize` is called. Designed to be cheap to append and easy to parse.

Each ``log_iter`` call is serialised with an advisory file lock and uses
``O_APPEND`` so concurrent writers (e.g. two seeds that mistakenly share an
output path, or a retry that overlaps with the previous attempt) cannot
interleave bytes within a single record. Each record is encoded into a
single ``os.write(fd, payload)`` call which is atomic on POSIX for sizes up
to ``PIPE_BUF`` and serialised by the lock for larger payloads.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from typing import Any, Optional

import torch


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    try:
        return float(x)
    except Exception:
        return str(x)


class RunLogger:
    def __init__(self, out_dir: str, run_name: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.run_name = run_name
        self.jsonl_path = os.path.join(out_dir, f"{run_name}.jsonl")
        self.summary_path = os.path.join(out_dir, f"{run_name}.summary.json")
        # Truncate the JSONL exactly once at construction so a fresh run
        # starts from an empty file; subsequent appends are done via per-call
        # O_APPEND opens guarded by flock (see ``log_iter``). This keeps the
        # legacy "one logger per run" semantics while making the writer
        # robust to overlapping processes that share the same path.
        with open(self.jsonl_path, "w"):
            pass
        self._t0 = time.time()

    def log_iter(self, record: dict) -> None:
        record = {"_t": time.time() - self._t0, **record}
        payload = (json.dumps(_to_jsonable(record)) + "\n").encode("utf-8")
        # O_APPEND positions every write at EOF atomically (POSIX). The
        # advisory flock additionally serialises the write itself so payloads
        # exceeding PIPE_BUF cannot be interleaved either.
        fd = os.open(self.jsonl_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                # os.write may return short on signals; loop until drained.
                view = memoryview(payload)
                while view:
                    n = os.write(fd, view)
                    if n <= 0:
                        break
                    view = view[n:]
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def finalize(self, summary: dict) -> None:
        # Serialise fully in memory first, then write atomically: a plain
        # open(path, "w") + json.dump truncates in place and streams multiple
        # writes, so two overlapping ``finalize`` calls on the same path
        # (e.g. a retry that overlaps with the previous attempt) can
        # interleave and leave a truncated-then-appended file behind. Writing
        # to a temp file in the same directory and ``os.replace``-ing it into
        # place makes the update atomic: any concurrent writer either wins
        # outright or loses outright, but the result is always one complete,
        # valid JSON document.
        payload = json.dumps(_to_jsonable(summary), indent=2)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.out_dir, prefix=f".{self.run_name}.summary.", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.summary_path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def close(self) -> None:
        # Kept for API compatibility; the per-write open/close model means
        # there is no long-lived file handle to release.
        return None
