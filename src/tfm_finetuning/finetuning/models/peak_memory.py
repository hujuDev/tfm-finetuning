"""Small CUDA peak-memory probes shared by transfer trainers."""

from __future__ import annotations

from typing import Any

_BYTES_PER_GIB = 1024**3


def reset_peak_gpu_memory(device: str | None) -> None:
    """Start a peak-allocation measurement when the run uses CUDA."""
    resolved = _cuda_device(device)
    if resolved is None:
        return
    import torch

    torch.cuda.synchronize(resolved)
    torch.cuda.reset_peak_memory_stats(resolved)


def peak_gpu_memory_gib(device: str | None) -> float | None:
    """Return peak allocated CUDA memory since the latest reset."""
    resolved = _cuda_device(device)
    if resolved is None:
        return None
    import torch

    torch.cuda.synchronize(resolved)
    return float(torch.cuda.max_memory_allocated(resolved) / _BYTES_PER_GIB)


def _cuda_device(device: str | None) -> Any | None:
    import torch

    if not torch.cuda.is_available():
        return None
    text = str(device or "cuda").strip().lower()
    if text and text != "auto" and not text.startswith("cuda"):
        return None
    return torch.device("cuda" if text in {"", "auto"} else text)


__all__ = ["peak_gpu_memory_gib", "reset_peak_gpu_memory"]
