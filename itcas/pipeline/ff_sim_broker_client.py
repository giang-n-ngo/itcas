"""Client library for the FF-sim Slurm dispatch broker.

Dispatch-only helper: this module contains NO scientific / scoring logic. It
just implements the *client* side of a simple filesystem-based request queue
protocol so that many independent, concurrently-running processes (e.g. the
``N_CONCURRENT`` sequential-baseline seed processes launched by
``scripts/run_seedset.sh``) can each submit a small number of pending
formation-flying simulations and have them merged, by a separate broker
daemon (``scripts/ff_sim_broker/broker.py``), into fewer/bigger
``sbatch --cpus-per-task=N`` jobs against ``SmartSat/ff_sim_batch.sbatch``.

This exists because ``bf_max_job_user_part`` on the cluster caps concurrently
*running* jobs per user per partition at 5, regardless of how many CPUs each
job asks for. Batching multiple pending single-simulation requests into one
bigger job means the 5-job cap constrains job *count*, not total throughput.

Opt-in only: ``SpacecraftFormationFlyingA1._slurm_batch_fn`` only calls into
this module when ``FF_SIM_USE_BROKER`` is truthy in the environment. When
unset (the default), the existing direct ``sbatch``-per-call path is used
unchanged, so this module has zero effect on the current in-flight sweep.

Queue protocol (a request is one directory under ``<queue_dir>/incoming/``):
    params_NNNN.json   one per requested simulation (written by client)
    READY               written last by the client once all params files are
                         in place; JSON body ``{"n_rows": N, "created_at": t}``.
                         Presence of READY (before the broker claims it) is
                         what makes a request visible to the broker.
    CLAIMED             created by the broker via an atomic ``os.rename`` of
                         READY -> CLAIMED; this is the sole synchronization
                         primitive that lets many concurrent clients share one
                         queue dir without any additional locking, since a
                         rename of a given path can only succeed once.
    JOB_ID              written by the broker once the merged sbatch job is
                         submitted (informational; contains the Slurm job id).
    result_NNNN.json    written by the broker (copied from the merged job's
                         own result files) once available.
    DONE                written last by the broker once every result_NNNN.json
                         for this request has been resolved (present or
                         permanently missing, e.g. after a job-level timeout).
                         Presence of DONE is what tells the client it is safe
                         to read results and clean up.
    ABANDONED           written by the client if IT gives up waiting past its
                         own timeout. Tells the broker (checked right before
                         it would write DONE) that nobody is waiting anymore,
                         so the broker deletes the request dir itself instead
                         of writing into it.

None of this touches how simulations are scored -- ``result_NNNN.json``
contents are opaque to this module and are interpreted exactly as before by
``SpacecraftFormationFlyingA1`` using the existing
``_metrics_to_constraints`` / ``_penalty_constraints`` functions.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

# Queue protocol filenames -- MUST match scripts/ff_sim_broker/broker.py.
READY_NAME = "READY"
CLAIMED_NAME = "CLAIMED"
DONE_NAME = "DONE"
ABANDONED_NAME = "ABANDONED"
JOB_ID_NAME = "JOB_ID"
INCOMING_SUBDIR = "incoming"


def default_queue_dir(smartsat_root: Path) -> Path:
    """Default broker queue directory, alongside the existing ff_sim_work/ dir."""
    return Path(smartsat_root) / "ff_sim_work" / "broker"


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def submit_via_broker(
    params_list: list[dict],
    *,
    queue_dir: Path,
    job_timeout: float,
    poll_interval: float = 5.0,
) -> list[Optional[dict]]:
    """Submit ``len(params_list)`` simulation requests to the broker and block
    until results are available or ``job_timeout`` elapses.

    Returns a list the same length as ``params_list``; entries are the parsed
    ``result_NNNN.json`` dict for rows that succeeded, or ``None`` for rows
    that failed, were lost, or never came back in time (caller applies the
    same penalty-constraint fallback used by the direct-sbatch path).
    """
    incoming = Path(queue_dir) / INCOMING_SUBDIR
    incoming.mkdir(parents=True, exist_ok=True)

    req_dir = Path(tempfile.mkdtemp(prefix="req_", dir=str(incoming)))
    n = len(params_list)

    for i, params in enumerate(params_list):
        with open(req_dir / f"params_{i:04d}.json", "w") as f:
            json.dump(params, f)

    # READY written last (and atomically) so the broker never sees a
    # half-written request.
    _atomic_write_json(req_dir / READY_NAME, {"n_rows": n, "created_at": time.time()})

    deadline = time.monotonic() + job_timeout
    done_path = req_dir / DONE_NAME
    try:
        while not done_path.exists():
            if time.monotonic() > deadline:
                # Give up waiting. Tell the broker (if it is still working on
                # a merged job containing this request) that nobody is
                # listening anymore, so it cleans up instead of writing DONE
                # into an orphaned directory.
                try:
                    (req_dir / ABANDONED_NAME).touch()
                except OSError:
                    pass
                return [None] * n
            time.sleep(poll_interval)

        results: list[Optional[dict]] = []
        for i in range(n):
            rp = req_dir / f"result_{i:04d}.json"
            if rp.exists():
                try:
                    with open(rp) as f:
                        results.append(json.load(f))
                except Exception:
                    results.append(None)
            else:
                results.append(None)
        return results
    finally:
        # Only reachable on the happy (DONE-observed) path; the timeout path
        # returns early above and deliberately leaves the directory for the
        # broker/reaper to remove once it notices ABANDONED.
        if done_path.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def broker_is_available(queue_dir: Path, *, max_staleness: float = 120.0) -> bool:
    """Best-effort liveness check: is a broker daemon actively servicing
    ``queue_dir``? Looks for a heartbeat file the daemon touches every scan
    cycle. Used so callers can fail fast with a clear error instead of
    silently blocking for the full ``job_timeout`` if no broker is running.
    """
    heartbeat = Path(queue_dir) / "broker.heartbeat"
    try:
        age = time.time() - heartbeat.stat().st_mtime
    except OSError:
        return False
    return age <= max_staleness
