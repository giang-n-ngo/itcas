"""Device resolution helpers for CPU/GPU execution.

Experiments are launched on a Slurm cluster where a GPU may or may not be
present. To keep entry points scriptable and free of interactive prompts we
resolve the compute device from a simple string spec:

    "auto"   -> CUDA if available, else CPU
    "cpu"    -> CPU
    "cuda"   -> default CUDA device (errors if unavailable)
    "cuda:N" -> CUDA device N (errors if unavailable)

The heavy lifting (GP fitting, posterior queries, acquisition optimization)
runs on the resolved device; cheap analytic objective evaluations stay on CPU
inside ``Problem`` so the synthetic constants need no device bookkeeping.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import torch


def _cuda_kernels_runnable(device: torch.device) -> tuple[bool, str]:
    """Probe whether the installed PyTorch can actually launch kernels on ``device``.

    On Slurm clusters with mixed GPU generations a node may expose a CUDA device
    whose compute capability is newer than anything the installed PyTorch wheel
    was compiled for (e.g. sm_120 Blackwell against a wheel built for <= sm_90).
    ``torch.cuda.is_available()`` still returns True in that case, but the very
    first kernel launch fails with "no kernel image is available for execution
    on the device". We probe with a trivial op so we can fail fast (or fall
    back) with an actionable message instead of crashing deep inside a run.
    """
    try:
        _ = torch.zeros(1, device=device).add_(1).sum().item()
        return True, ""
    except Exception as e:  # pragma: no cover - hardware dependent
        try:
            major, minor = torch.cuda.get_device_capability(device)
            cap = f"sm_{major}{minor}"
        except Exception:
            cap = "unknown"
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = "unknown"
        try:
            supported = ", ".join(torch.cuda.get_arch_list())
        except Exception:
            supported = "unknown"
        return False, (
            f"CUDA device '{device}' ({name}, capability {cap}) is visible but "
            f"the installed PyTorch build cannot launch kernels on it. "
            f"Supported archs: [{supported}]. Probe error: {e}"
        )


def resolve_device(spec: Optional[Union[str, torch.device]] = "auto") -> torch.device:
    """Resolve a device spec string into a concrete ``torch.device``.

    Args:
        spec: one of {"auto", "cpu", "cuda", "cuda:N"} or a ``torch.device``.
            ``None`` is treated as "auto".

    Returns:
        A ``torch.device``. Falls back to CPU when "auto" is requested and no
        CUDA device is visible.

    Raises:
        RuntimeError: if an explicit CUDA device is requested but unavailable.
    """
    if isinstance(spec, torch.device):
        device = spec
    elif spec is None or spec == "auto":
        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda")
            ok, msg = _cuda_kernels_runnable(cuda_dev)
            if ok:
                return cuda_dev
            warnings.warn(
                f"{msg} Falling back to CPU (device='auto').", RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device("cpu")
    else:
        device = torch.device(spec)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{spec}' but CUDA is not available. "
                f"Use device='cpu' or device='auto'."
            )
        ok, msg = _cuda_kernels_runnable(device)
        if not ok:
            raise RuntimeError(
                f"Requested device '{spec}' but it is not runnable. {msg} "
                f"Either install a PyTorch build supporting this GPU's compute "
                f"capability, exclude the offending Slurm node, or set "
                f"device='auto'/'cpu'."
            )
    return device
