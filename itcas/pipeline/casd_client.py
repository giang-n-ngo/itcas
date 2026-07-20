"""Client library for the CASD (Context-Aware Safe Decoding) evaluator server.

Mirrors the spirit of ``ff_sim_broker_client.py``: this module contains NO
scientific / scoring logic. It just implements the *client* side of a small
HTTP protocol so the ``itcas`` process (running in the BoTorch/GPyTorch env)
can talk to a separate, persistent ``scripts/casd_server/server.py`` process
(running in its own vLLM/transformers env, typically on a GPU node) that
keeps a vLLM engine + judge models resident across many evaluations.

Unlike the FF-sim broker, there is no Slurm-dispatch queue here (that
protocol exists solely because of the cluster's 5-concurrent-job cap, which
doesn't apply to this benchmark). Instead this is a plain HTTP client
against a long-lived local/network server process, because the point of the
CASD benchmark is sub-0.5s/eval throughput, which requires GPU model
weights to stay loaded across calls rather than being reloaded per query
(as a per-call subprocess or Slurm job would do).

Only the Python standard library (``urllib.request``) is used here --
deliberately NOT ``requests`` -- so that adding this client has zero effect
on ``requirements.txt`` (the itcas/BoTorch env's pinned dependency set).
This module has zero effect on any problem other than
``ContextAwareSafeDecoding`` in ``itcas/pipeline/problems.py``: it's a
plain library, imported (and its functions called) only from there.

Protocol (see scripts/casd_server/server.py for the server-side
implementation and full docstring):
    GET  /health                 -> liveness + pool size + mock flag
    GET  /contexts?n=K&seed=S    -> K deterministic {context_id,
                                     prompt_toxicity, prompt_length} rows
    POST /evaluate                body: JSON list of
                                     {temperature, top_p, repetition_penalty,
                                      context_id}  (or the nearest-neighbor
                                     "continuous c" form with
                                     prompt_toxicity/prompt_length instead of
                                     context_id)
                                   response: JSON list of {f1, f2} or null,
                                     aligned to input order.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import warnings as _warnings
from typing import Optional


def _http_get_json(url: str, timeout: float):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body, timeout: float):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def server_is_available(base_url: str, timeout: float = 5.0) -> bool:
    """Best-effort liveness check against GET /health.

    Used so callers can fail fast with a clear error instead of hanging (or
    silently getting worst-case penalty rows) if no CASD server is running
    at ``base_url``.
    """
    try:
        resp = _http_get_json(f"{base_url.rstrip('/')}/health", timeout=timeout)
    except Exception:
        return False
    return bool(resp) and resp.get("status") == "ok"


def sample_contexts(base_url: str, n: int, seed: int, timeout: float = 30.0) -> list[dict]:
    """GET /contexts?n=K&seed=S -> list of {context_id, prompt_toxicity, prompt_length}.

    Deterministic given (n, seed): repeated calls with the same arguments
    against the same server process return the same rows.
    """
    qs = urllib.parse.urlencode({"n": int(n), "seed": int(seed)})
    url = f"{base_url.rstrip('/')}/contexts?{qs}"
    return _http_get_json(url, timeout=timeout)


def evaluate_batch(
    base_url: str, items: list[dict], timeout: float = 300.0
) -> list[Optional[dict]]:
    """POST /evaluate with a batch of items; returns aligned {f1, f2} or None rows.

    Each item in ``items`` must be a dict with keys ``temperature``,
    ``top_p``, ``repetition_penalty``, and either ``context_id`` (int) or
    both ``prompt_toxicity`` and ``prompt_length`` (floats, triggering
    nearest-neighbor snap-to-real-prompt on the server side).

    On a network/HTTP-level failure (server unreachable, malformed
    response, etc.) the whole batch fails: returns a list of ``None`` the
    same length as ``items`` rather than raising, so callers can apply
    their own worst-case-penalty fallback uniformly regardless of whether
    the failure was per-item (server returned null for that row) or
    whole-batch (server unreachable).
    """
    url = f"{base_url.rstrip('/')}/evaluate"
    try:
        resp = _http_post_json(url, items, timeout=timeout)
    except Exception as e:
        # Logged (not silently swallowed) so a whole-batch failure -- e.g. a
        # non-2xx HTTP response body carrying a server-side traceback, or a
        # timeout -- leaves a trace to diagnose from, instead of looking
        # identical to every other None-row cause once it reaches the
        # worst-case-penalty fallback in ContextAwareSafeDecoding.evaluate_true.
        detail = getattr(e, "read", None)
        body = detail().decode("utf-8", "replace")[:2000] if callable(detail) else ""
        _warnings.warn(
            f"[casd_client] evaluate_batch POST {url} failed for a batch of "
            f"{len(items)} items: {e!r}{(' body=' + body) if body else ''}",
            RuntimeWarning,
            stacklevel=2,
        )
        return [None] * len(items)
    if not isinstance(resp, list) or len(resp) != len(items):
        return [None] * len(items)
    return resp
