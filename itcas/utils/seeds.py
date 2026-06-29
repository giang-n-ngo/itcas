"""Seed-sweep helpers: parse seed specs and detect already-completed runs.

These utilities let a single entry point "complete the sequence" of seeds:
given a target set of seeds (e.g. 1-10) and a results directory, we can skip
the seeds whose run already finished and execute only the missing ones.

A run is considered *complete* when its summary JSON
(``<out_dir>/<run_name>.summary.json``) exists and is valid JSON — that file is
only written by ``RunLogger.finalize`` at the very end of ``run_experiment``,
so its presence is a reliable "this seed finished" marker.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

SEED_PLACEHOLDER = "{seed}"


def parse_seed_spec(spec: str | Iterable[int]) -> list[int]:
    """Parse a seed specification into a sorted, de-duplicated list of ints.

    Accepts either an iterable of ints, or a string of comma-separated tokens
    where each token is a single integer or an inclusive ``a-b`` range, e.g.::

        "1-10"          -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        "1,3,6-8"       -> [1, 3, 6, 7, 8]
        "0-2, 5"        -> [0, 1, 2, 5]

    Raises:
        ValueError: on malformed tokens or descending ranges.
    """
    if not isinstance(spec, str):
        return sorted({int(s) for s in spec})

    seeds: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token.lstrip("-"):
            # Range "a-b" (supports negative endpoints like "-2--1" rarely, but
            # primarily non-negative seeds). Split on the last '-' that is not a
            # leading sign.
            lo_str, hi_str = _split_range(token)
            lo, hi = int(lo_str), int(hi_str)
            if hi < lo:
                raise ValueError(f"Descending seed range '{token}' (a-b needs a<=b).")
            seeds.update(range(lo, hi + 1))
        else:
            seeds.add(int(token))
    return sorted(seeds)


def _split_range(token: str) -> tuple[str, str]:
    """Split an ``a-b`` range token, tolerating a leading sign on ``a``."""
    start = 1 if token[0] in "+-" else 0
    idx = token.find("-", start)
    if idx == -1:
        raise ValueError(f"Malformed seed range '{token}'.")
    return token[:idx], token[idx + 1 :]


def run_name_for_seed(template: str, seed: int) -> str:
    """Build a per-seed run name.

    If ``template`` contains the ``{seed}`` placeholder it is substituted;
    otherwise ``_seed<seed>`` is appended. This keeps run names unique per seed
    so their summary files don't collide.
    """
    if SEED_PLACEHOLDER in template:
        return template.replace(SEED_PLACEHOLDER, str(seed))
    return f"{template}_seed{seed}"


def summary_path(out_dir: str, run_name: str) -> str:
    return os.path.join(out_dir, f"{run_name}.summary.json")


def lock_path(out_dir: str, run_name: str) -> str:
    return os.path.join(out_dir, f"{run_name}.lock")


class RunLock:
    """Advisory non-blocking lock to prevent overlapping runs on the same path.

    Used as a context manager; raises ``RunBusyError`` immediately if another
    process already holds the lock. The lockfile is left on disk after release
    (its presence is harmless; the kernel-side flock state is what matters)
    so it does not race with a second attempter that opens the file in
    parallel.
    """

    def __init__(self, out_dir: str, run_name: str):
        os.makedirs(out_dir, exist_ok=True)
        self.path = lock_path(out_dir, run_name)
        self._fd: int | None = None

    def __enter__(self) -> "RunLock":
        import fcntl

        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RunBusyError(
                f"another process holds the run lock for {self.path}"
            )
        self._fd = fd
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import fcntl

        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


class RunBusyError(RuntimeError):
    """Raised when another process is already running this (out_dir, run_name)."""


def is_run_complete(out_dir: str, run_name: str) -> bool:
    """True if ``<out_dir>/<run_name>.summary.json`` exists and is valid JSON."""
    path = summary_path(out_dir, run_name)
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return True


def pending_seeds(
    seeds: Iterable[int],
    out_dir: str,
    run_name_template: str,
    force: bool = False,
) -> list[int]:
    """Return the subset of ``seeds`` whose runs are not yet complete.

    Args:
        seeds: target seeds (the full sequence to complete).
        out_dir: results directory where summary files live.
        run_name_template: base run name (may contain ``{seed}``).
        force: if True, return all seeds (re-run everything).
    """
    out = []
    for s in seeds:
        rn = run_name_for_seed(run_name_template, s)
        if force or not is_run_complete(out_dir, rn):
            out.append(s)
    return out
