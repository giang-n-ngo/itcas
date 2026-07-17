#!/usr/bin/env python3
"""ff_sim_broker/broker.py -- dispatch-only Slurm submission broker for FF sims.

Standalone daemon (pure stdlib, no torch/gpytorch/conda env required). Run it
on a login node (see start_broker.sh); it does NOT need to be, and should NOT
be, submitted as a Slurm job itself -- it only shells out to sbatch/squeue/
scancel and does filesystem I/O, so running it as a Slurm job would just
waste one of the 5 backfill slots this whole mechanism exists to relieve.

Purpose
-------
Many independent, concurrently-running seed processes (see
``scripts/run_seedset.sh``'s ``N_CONCURRENT``) each want to evaluate a small
number (often just 1) of formation-flying simulations at a time, submitted
today via one ``sbatch`` call per process per BO iteration
(``SpacecraftFormationFlyingA1._slurm_batch_fn`` in
``itcas/pipeline/problems.py``). Because the cluster's backfill scheduler
caps concurrently RUNNING jobs at 5 per user per partition
(``bf_max_job_user_part=5``), submitting one job per single simulation wastes
almost all of that headroom on tiny 1-CPU jobs.

This broker sits between those callers and Slurm: callers write their pending
simulation requests into a shared queue directory (via
``itcas/pipeline/ff_sim_broker_client.py``); this daemon periodically scans
that directory, merges up to ``FF_SIM_MAX_BATCH`` pending simulation rows
(possibly from several different callers) into ONE job directory, and submits
ONE ``sbatch --cpus-per-task=N`` job against the existing, unmodified
``SmartSat/ff_sim_batch.sbatch`` (which already fans work out internally via
GNU parallel). Once that job finishes, results are copied back into each
caller's own request directory so each caller sees just its own rows.

Concurrency model
------------------
The daemon can have up to ``FF_SIM_BROKER_MAX_INFLIGHT_JOBS`` merged Slurm
jobs in flight AT ONCE (default 5, matching the backfill concurrency cap this
whole mechanism exists to use). Each main-loop tick (every
``FF_SIM_BROKER_SCAN_INTERVAL`` seconds) does two non-blocking things:
    1. Advances every currently in-flight job by a single check (has it
       produced all its results yet? timed out? vanished from squeue?) --
       never sleeps/blocks waiting on any one job, so many jobs are polled
       and can complete independently, in any order.
    2. If there is spare in-flight capacity, scans the queue and submits (at
       most) one more merged batch.
This is deliberately NOT thread/process-based: a single main loop holding a
plain list of small in-flight-job records is enough to track several
concurrent Slurm jobs and is much simpler to reason about and test than
spawning OS threads per job.

This module performs NO scoring / metric computation -- it moves opaque
params_NNNN.json / result_NNNN.json files around and calls sbatch/squeue/
scancel exactly as ``_slurm_batch_fn`` already did per-call.

Configuration (env vars, all optional):
    FF_SIM_BROKER_QUEUE_DIR     queue directory (default: SMARTSAT_ROOT/ff_sim_work/broker)
    FF_SIM_MAX_BATCH            max simulation rows merged into one job (default: 8)
    FF_SIM_BROKER_MAX_INFLIGHT_JOBS  max merged Slurm jobs in flight at once (default: 5)
    FF_SIM_BROKER_POLL_WINDOW   max seconds to wait accumulating more requests
                                 before submitting a partial (< max batch) job (default: 15)
    FF_SIM_BROKER_SCAN_INTERVAL seconds between queue scans / in-flight-job checks (default: 3)
    FF_SIM_BROKER_JOB_TIMEOUT   max seconds to wait for one merged job's results (default: 1800)
    FF_SIM_BROKER_SQUEUE_GRACE  min seconds after submission before trusting an empty
                                 `squeue --job` as "job finished" rather than "not yet
                                 registered with the scheduler" (default: 20)
    FF_SIM_BROKER_SETTLE_RETRIES  ticks to keep re-checking the result-file count after
                                 squeue reports a job gone, to ride out NFS metadata-
                                 visibility lag before trusting a low/zero count (default: 12)
    FF_SIM_BROKER_SETTLE_INTERVAL  seconds represented by each settle tick, used only to
                                 size the total settle window (retries * interval); actual
                                 pacing follows the main loop's scan_interval (default: 5)
    FF_SIM_BROKER_STALE_SECS    age after which orphaned/finished request dirs are reaped (default: 3600)
    FF_SIM_CONDA_ENV / CONDA_ENV  conda env exported to the child ff_sim_batch.sbatch job (default: scarlet)
    SMARTSAT_ROOT               path to the SmartSat checkout (default: sibling of this repo)

Usage:
    python3 scripts/ff_sim_broker/broker.py [--once]

``--once`` runs a single main-loop tick (advance in-flight jobs + maybe start
one more) and exits; useful for scripted/step-wise testing, but note that
in-flight job state only lives in this process's memory, so chaining several
separate ``--once`` invocations loses track of jobs submitted by an earlier
one. Prefer running the daemon continuously (the default) for anything that
needs to see a merged job through to completion.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

READY_NAME = "READY"
CLAIMED_NAME = "CLAIMED"
DONE_NAME = "DONE"
ABANDONED_NAME = "ABANDONED"
JOB_ID_NAME = "JOB_ID"
INCOMING_SUBDIR = "incoming"


def _log(msg: str) -> None:
    print(f"[ff_sim_broker] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


class BrokerConfig:
    def __init__(self, smartsat_root: Path):
        self.smartsat_root = smartsat_root
        self.queue_dir = Path(
            os.environ.get("FF_SIM_BROKER_QUEUE_DIR")
            or (smartsat_root / "ff_sim_work" / "broker")
        )
        self.max_batch = int(os.environ.get("FF_SIM_MAX_BATCH", "8"))
        self.max_inflight_jobs = int(os.environ.get("FF_SIM_BROKER_MAX_INFLIGHT_JOBS", "5"))
        self.poll_window = float(os.environ.get("FF_SIM_BROKER_POLL_WINDOW", "15"))
        self.scan_interval = float(os.environ.get("FF_SIM_BROKER_SCAN_INTERVAL", "3"))
        self.job_timeout = float(os.environ.get("FF_SIM_BROKER_JOB_TIMEOUT", "1800"))
        self.stale_secs = float(os.environ.get("FF_SIM_BROKER_STALE_SECS", "3600"))
        self.job_squeue_grace = float(os.environ.get("FF_SIM_BROKER_SQUEUE_GRACE", "20"))
        self.job_settle_retries = int(os.environ.get("FF_SIM_BROKER_SETTLE_RETRIES", "12"))
        self.job_settle_interval = float(os.environ.get("FF_SIM_BROKER_SETTLE_INTERVAL", "5"))
        self.conda_env = (
            os.environ.get("FF_SIM_CONDA_ENV")
            or os.environ.get("CONDA_ENV")
            or "scarlet"
        )
        self.sbatch_bin = os.environ.get("FF_SIM_SBATCH_BIN", "sbatch")
        self.squeue_bin = os.environ.get("FF_SIM_SQUEUE_BIN", "squeue")
        self.scancel_bin = os.environ.get("FF_SIM_SCANCEL_BIN", "scancel")
        self.work_root = Path(
            os.environ.get("FF_SIM_WORK_ROOT") or (smartsat_root / "ff_sim_work")
        )
        self.slurm_script = smartsat_root / "ff_sim_batch.sbatch"

    @property
    def settle_window_secs(self) -> float:
        return self.job_settle_retries * self.job_settle_interval


def _acquire_singleton_lock(queue_dir: Path):
    """Ensure only one broker daemon instance runs against a given queue_dir.

    Belt-and-suspenders: request-claiming below is already safe under
    multiple broker instances (atomic rename), but avoiding two daemons
    polling/submitting redundantly is simpler with a single-instance lock.
    """
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock_path = queue_dir / "broker.lock"
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _log(f"ERROR: another broker instance already holds {lock_path}; exiting.")
        sys.exit(1)
    fh.write(str(os.getpid()))
    fh.flush()
    return fh  # keep a reference alive for the lifetime of the process


def _scan_ready_requests(incoming: Path) -> list[tuple[Path, int, float]]:
    """Return (req_dir, n_rows, ready_mtime) for every request awaiting claim,
    oldest first (FIFO), so no request is starved by a stream of newer ones.
    """
    out = []
    if not incoming.is_dir():
        return out
    for d in incoming.iterdir():
        if not d.is_dir():
            continue
        ready = d / READY_NAME
        try:
            st = ready.stat()
        except OSError:
            continue
        try:
            info = json.loads(ready.read_text())
            n_rows = int(info.get("n_rows", 1))
        except Exception:
            n_rows = 1
        out.append((d, n_rows, st.st_mtime))
    out.sort(key=lambda t: t[2])
    return out


def _select_batch(reqs: list[tuple[Path, int, float]], max_batch: int, poll_window: float) -> Optional[list[tuple[Path, int]]]:
    """Decide whether to submit now, and if so, which requests to bundle.

    Waits up to ``poll_window`` seconds (from the oldest pending request) to
    accumulate more work before submitting fewer than ``max_batch`` rows, so
    a lone caller doesn't wait forever, but a burst of concurrent callers
    gets merged into one job.
    """
    if not reqs:
        return None
    total = sum(n for _, n, _ in reqs)
    oldest_wait = time.time() - reqs[0][2]
    if total < max_batch and oldest_wait < poll_window:
        return None  # keep accumulating a bit longer

    selected: list[tuple[Path, int]] = []
    running = 0
    for d, n, _ in reqs:
        if selected and running + n > max_batch:
            break
        selected.append((d, n))
        running += n
    return selected


def _claim_batch(selected: list[tuple[Path, int]]) -> list[Path]:
    """Atomically claim each selected request (READY -> CLAIMED rename).

    A request whose caller already gave up (ABANDONED marker present) is
    cleaned up immediately instead of being handed to a job.
    """
    claimed: list[Path] = []
    for d, _n in selected:
        try:
            os.rename(d / READY_NAME, d / CLAIMED_NAME)
        except FileNotFoundError:
            # Lost the (single-broker-guaranteed-uncontested, but defensive)
            # race to claim -- e.g. a second broker instance, or the client
            # itself cleaned up. Skip; not our request to handle.
            continue
        if (d / ABANDONED_NAME).exists():
            shutil.rmtree(d, ignore_errors=True)
            continue
        claimed.append(d)
    return claimed


class InFlightJob:
    """State for one merged Slurm job the broker is tracking to completion.

    Deliberately a plain mutable record (not the unit of concurrency itself
    -- there's no thread/process per job): the main loop holds a list of
    these and advances each one a little every tick via ``_poll_inflight_job``.
    """

    __slots__ = (
        "job_dir", "job_id", "manifest", "claimed", "n_total",
        "submitted_at", "deadline", "vanished_at", "terminal", "terminal_reason",
    )

    def __init__(self, job_dir: Path, job_id: str, manifest: list[tuple[Path, int, int]],
                 claimed: list[Path], n_total: int, submitted_at: float, deadline: float):
        self.job_dir = job_dir
        self.job_id = job_id
        self.manifest = manifest
        self.claimed = claimed
        self.n_total = n_total
        self.submitted_at = submitted_at
        self.deadline = deadline
        self.vanished_at: Optional[float] = None
        self.terminal = False
        self.terminal_reason = ""


def _count_done(job_dir: Path, n_total: int) -> int:
    return sum(1 for i in range(n_total) if (job_dir / f"result_{i:04d}.json").exists())


def _start_merged_job(cfg: BrokerConfig, claimed: list[Path]) -> InFlightJob:
    """Merge params files from ``claimed`` request dirs into one job dir and
    submit ONE sbatch job. This only does the (fast) merge + submit step --
    it does NOT wait for the job to finish; call ``_poll_inflight_job``
    repeatedly (once per main-loop tick) to advance it, and
    ``_finalize_inflight_job`` once it reports terminal.

    Always returns an ``InFlightJob``, even on failure (empty batch, sbatch
    submission error, or an unexpected exception) -- it is simply marked
    already-terminal in that case, so the caller's normal finalize path
    still runs and resolves every claimed request (with the existing
    penalty-fallback semantics for missing results) instead of hanging.
    """
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="ff_broker_job_", dir=str(cfg.work_root)))
    manifest: list[tuple[Path, int, int]] = []
    now = time.monotonic()
    try:
        global_idx = 0
        for d in claimed:
            for p in sorted(d.glob("params_*.json")):
                try:
                    local_idx = int(p.stem.split("_")[1])
                except ValueError:
                    continue
                shutil.copy(p, job_dir / f"params_{global_idx:04d}.json")
                manifest.append((d, local_idx, global_idx))
                global_idx += 1
        n_total = global_idx
        if n_total == 0:
            _log("WARNING: claimed batch had zero params files; nothing to submit.")
            job = InFlightJob(job_dir, "", manifest, claimed, 0, now, now)
            job.terminal, job.terminal_reason = True, "empty batch"
            return job

        log_pat = str(job_dir / "slurm_%j.log")
        sbatch_cmd = [
            cfg.sbatch_bin,
            "--parsable",
            f"--cpus-per-task={n_total}",
            f"--output={log_pat}",
            f"--error={log_pat}",
            f"--export=ALL,CONDA_ENV={cfg.conda_env}",
            str(cfg.slurm_script),
            str(job_dir),
            str(cfg.smartsat_root),
        ]
        _log(f"submitting merged job: {n_total} sim(s) from {len(claimed)} request(s), cmd={' '.join(sbatch_cmd)}")
        proc = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        submitted_at = time.monotonic()
        if proc.returncode != 0:
            _log(f"ERROR: sbatch failed rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}")
            job = InFlightJob(job_dir, "", manifest, claimed, n_total, submitted_at, submitted_at)
            job.terminal, job.terminal_reason = True, "sbatch submission failed"
            return job
        job_id = proc.stdout.strip().split(";")[0]
        _log(f"submitted job_id={job_id} n_total={n_total}")
        for d in claimed:
            try:
                _atomic_write(d / JOB_ID_NAME, job_id)
            except OSError:
                pass
        return InFlightJob(
            job_dir, job_id, manifest, claimed, n_total,
            submitted_at, submitted_at + cfg.job_timeout,
        )
    except Exception:
        _log("ERROR: exception while starting merged batch:\n" + traceback.format_exc())
        job = InFlightJob(job_dir, "", manifest, claimed, len(manifest), now, now)
        job.terminal, job.terminal_reason = True, "exception during submission"
        return job


def _poll_inflight_job(cfg: BrokerConfig, job: InFlightJob) -> None:
    """Advance one in-flight job by a single non-blocking check.

    Never sleeps. Sets ``job.terminal`` once it's ready to be finalized
    (results copied back + DONE written) and dropped from the in-flight
    list. Called once per main-loop tick for every in-flight job, so many
    jobs make progress independently and in any order -- nothing here waits
    on any single job before the loop can move on to the next one or start
    a new batch.
    """
    if job.terminal or job.n_total == 0:
        job.terminal = True
        return

    n_done = _count_done(job.job_dir, job.n_total)
    if n_done == job.n_total:
        job.terminal, job.terminal_reason = True, "completed"
        return

    now = time.monotonic()
    if now > job.deadline:
        _log(f"job_id={job.job_id} timed out after {cfg.job_timeout}s ({n_done}/{job.n_total} results); scancel + fallback")
        subprocess.run([cfg.scancel_bin, job.job_id], capture_output=True)
        job.terminal, job.terminal_reason = True, "timeout"
        return

    if job.vanished_at is not None:
        # squeue already reported this job gone on an earlier tick; we're
        # riding out a settle window before trusting a low/zero result
        # count, since this broker reads job_dir over NFS from the login
        # node and a file a *different* compute node just finished writing
        # can take a while (empirically sometimes 20-30+s, not just a
        # couple of seconds) to become visible here -- even after the job
        # has already exited squeue's view entirely. n_done is rechecked at
        # the top of this function every tick regardless, so as soon as the
        # filesystem catches up we exit via the check above; if the whole
        # settle window elapses without that happening, give up and finalize
        # with whatever count we have (existing penalty-fallback semantics
        # cover genuinely lost/failed rows).
        if now - job.vanished_at > cfg.settle_window_secs:
            job.terminal, job.terminal_reason = True, "settled"
        return

    job_age = now - job.submitted_at
    if job_age > cfg.job_squeue_grace:
        # Don't trust an empty `squeue --job <id>` as "job finished" until a
        # grace period has passed since submission -- immediately after
        # sbatch returns a job id, the scheduler can have a brief
        # registration delay before that job is visible to squeue at all
        # (distinct from the job actually finishing); checking too early can
        # misread "not registered yet" as "already gone".
        try:
            sq = subprocess.run(
                [cfg.squeue_bin, "--job", job.job_id, "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if not sq.stdout.strip():
                job.vanished_at = now
        except Exception:
            pass
    # else: not yet past the registration-grace period; just keep waiting,
    # re-checked next tick.


def _finalize_inflight_job(job: InFlightJob) -> None:
    """Copy each claimed request's own result rows back and write DONE (or
    clean up if the caller already abandoned it). Missing rows are simply
    left absent -- the client applies the same worst-case penalty fallback
    it always has for a result file that never showed up.
    """
    for d in job.claimed:
        if (d / ABANDONED_NAME).exists():
            shutil.rmtree(d, ignore_errors=True)
            continue
        for req_dir, local_idx, global_idx in job.manifest:
            if req_dir != d:
                continue
            src = job.job_dir / f"result_{global_idx:04d}.json"
            if src.exists():
                try:
                    shutil.copy(src, d / f"result_{local_idx:04d}.json")
                except OSError:
                    pass
        try:
            _atomic_write(d / DONE_NAME, json.dumps({"job_id": job.job_id, "finished_at": time.time()}))
        except OSError:
            pass
    shutil.rmtree(job.job_dir, ignore_errors=True)


def _reap_stale(incoming: Path, stale_secs: float) -> None:
    """Clean up request dirs that will never be picked up by a client again:
    crashed before READY was written, or finished (DONE) long ago and the
    client never came back to read/delete them.
    """
    if not incoming.is_dir():
        return
    now = time.time()
    for d in incoming.iterdir():
        if not d.is_dir():
            continue
        try:
            mtime = d.stat().st_mtime
        except OSError:
            continue
        age = now - mtime
        has_ready_or_claimed = (d / READY_NAME).exists() or (d / CLAIMED_NAME).exists()
        has_done = (d / DONE_NAME).exists()
        if has_done and age > stale_secs:
            shutil.rmtree(d, ignore_errors=True)
        elif not has_ready_or_claimed and not has_done and age > stale_secs:
            # Never became READY (client crashed mid-write) -- orphaned.
            shutil.rmtree(d, ignore_errors=True)


def _resolve_smartsat_root() -> Path:
    env = os.environ.get("SMARTSAT_ROOT")
    if env:
        return Path(env).resolve()
    # scripts/ff_sim_broker/broker.py -> ITCAS repo root -> sibling SmartSat/
    here = Path(__file__).resolve()
    itcas_root = here.parent.parent.parent
    return (itcas_root.parent / "SmartSat").resolve()


def run(once: bool = False) -> None:
    smartsat_root = _resolve_smartsat_root()
    cfg = BrokerConfig(smartsat_root)
    if not cfg.slurm_script.exists():
        _log(f"ERROR: {cfg.slurm_script} not found; refusing to start.")
        sys.exit(2)

    (cfg.queue_dir / INCOMING_SUBDIR).mkdir(parents=True, exist_ok=True)
    lock_fh = _acquire_singleton_lock(cfg.queue_dir)  # noqa: F841 (keep alive)

    _log(
        f"starting: queue_dir={cfg.queue_dir} max_batch={cfg.max_batch} "
        f"max_inflight_jobs={cfg.max_inflight_jobs} poll_window={cfg.poll_window}s "
        f"scan_interval={cfg.scan_interval}s job_timeout={cfg.job_timeout}s "
        f"conda_env={cfg.conda_env}"
    )

    incoming = cfg.queue_dir / INCOMING_SUBDIR
    heartbeat = cfg.queue_dir / "broker.heartbeat"
    last_reap = 0.0
    inflight: list[InFlightJob] = []
    while True:
        try:
            heartbeat.touch()

            # 1. Advance every in-flight job by one non-blocking check; any
            #    that are done, timed out, or settled get finalized and
            #    dropped. Every remaining job gets exactly one check per
            #    tick, regardless of how many are in flight, so N jobs
            #    progress concurrently instead of one at a time.
            still_inflight: list[InFlightJob] = []
            for job in inflight:
                _poll_inflight_job(cfg, job)
                if job.terminal:
                    _finalize_inflight_job(job)
                else:
                    still_inflight.append(job)
            inflight = still_inflight

            # 2. If there's spare concurrency capacity, claim and start one
            #    more merged batch (submission itself is quick -- it does
            #    not block waiting for the job to finish).
            if len(inflight) < cfg.max_inflight_jobs:
                reqs = _scan_ready_requests(incoming)
                selected = _select_batch(reqs, cfg.max_batch, cfg.poll_window)
                if selected:
                    claimed = _claim_batch(selected)
                    if claimed:
                        inflight.append(_start_merged_job(cfg, claimed))

            if time.time() - last_reap > max(cfg.scan_interval * 10, 60):
                _reap_stale(incoming, cfg.stale_secs)
                last_reap = time.time()
        except Exception:
            _log("ERROR in main loop (continuing):\n" + traceback.format_exc())
        if once:
            return
        time.sleep(cfg.scan_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run a single main-loop tick and exit (for testing)")
    args = parser.parse_args()
    run(once=args.once)


if __name__ == "__main__":
    main()
