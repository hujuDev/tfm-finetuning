from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from types import TracebackType


class CudaMemoryTracker:
    """Measure peak PyTorch CUDA allocator memory for a code block.

    Example:
        >>> tracker = CudaMemoryTracker(device="cpu")
        >>> tracker.as_report_fields()
        {}
    """

    def __init__(self, device: Any = None) -> None:
        self.device = device
        self._torch: Any | None = None
        self._device_index: int | None = None
        self._fields: dict[str, Any] = {}
        self._active = False

    def __enter__(self) -> Self:
        """Start a CUDA peak-memory measurement when CUDA is available."""
        torch = _import_torch()
        if torch is None or not _cuda_available(torch):
            return self
        device_index = _resolve_device_index(torch, self.device)
        if device_index is None:
            return self
        self._torch = torch
        self._device_index = device_index
        self._active = True
        self._synchronize()
        start_allocated = _cuda_memory_mib(torch.cuda.memory_allocated(device_index))
        start_reserved = _cuda_memory_mib(torch.cuda.memory_reserved(device_index))
        free_mib, total_mib = _cuda_mem_get_info_mib(torch, device_index)
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats(device_index)
        self._fields.update(
            {
                "gpu_device_index": int(device_index),
                "gpu_device_name": _cuda_device_name(torch, device_index),
                "gpu_memory_start_allocated_mib": start_allocated,
                "gpu_memory_start_reserved_mib": start_reserved,
                "gpu_memory_start_free_mib": free_mib,
                "gpu_memory_total_mib": total_mib,
                "gpu_memory_source": "torch.cuda",
            }
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Stop measuring and keep the collected memory fields."""
        if not self._active or self._torch is None or self._device_index is None:
            return False
        torch = self._torch
        device_index = self._device_index
        self._synchronize()
        start_allocated = _number_or_zero(self._fields.get("gpu_memory_start_allocated_mib"))
        start_reserved = _number_or_zero(self._fields.get("gpu_memory_start_reserved_mib"))
        peak_allocated = _cuda_memory_mib(torch.cuda.max_memory_allocated(device_index))
        peak_reserved = _cuda_memory_mib(torch.cuda.max_memory_reserved(device_index))
        end_allocated = _cuda_memory_mib(torch.cuda.memory_allocated(device_index))
        end_reserved = _cuda_memory_mib(torch.cuda.memory_reserved(device_index))
        free_mib, total_mib = _cuda_mem_get_info_mib(torch, device_index)
        self._fields.update(
            {
                "gpu_memory_peak_allocated_mib": peak_allocated,
                "gpu_memory_peak_reserved_mib": peak_reserved,
                "gpu_memory_peak_allocated_delta_mib": max(0.0, peak_allocated - start_allocated),
                "gpu_memory_peak_reserved_delta_mib": max(0.0, peak_reserved - start_reserved),
                "gpu_memory_end_allocated_mib": end_allocated,
                "gpu_memory_end_reserved_mib": end_reserved,
                "gpu_memory_end_free_mib": free_mib,
                "gpu_memory_total_mib": total_mib,
            }
        )
        return False

    def as_report_fields(self) -> dict[str, Any]:
        """Return JSON-safe report fields for a runtime or evaluation row."""
        return {key: value for key, value in self._fields.items() if value is not None}

    def _synchronize(self) -> None:
        if self._torch is None or self._device_index is None:
            return
        with contextlib.suppress(Exception):
            self._torch.cuda.synchronize(self._device_index)


def _import_torch() -> Any | None:
    with contextlib.suppress(Exception):
        import torch

        return torch
    return None


def _cuda_available(torch: Any) -> bool:
    with contextlib.suppress(Exception):
        return bool(torch.cuda.is_available())
    return False


def _resolve_device_index(torch: Any, device: Any) -> int | None:
    if isinstance(device, int):
        return device
    text = str(device or "").strip().lower()
    if text.startswith("cuda:"):
        with contextlib.suppress(ValueError):
            return int(text.split(":", 1)[1])
    if text and text not in {"auto", "cuda", "gpu"}:
        return None
    with contextlib.suppress(Exception):
        return int(torch.cuda.current_device())
    return 0


def _cuda_device_name(torch: Any, device_index: int) -> str | None:
    with contextlib.suppress(Exception):
        return str(torch.cuda.get_device_name(device_index))
    return None


def _cuda_mem_get_info_mib(torch: Any, device_index: int) -> tuple[float | None, float | None]:
    with contextlib.suppress(Exception):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        return _cuda_memory_mib(free_bytes), _cuda_memory_mib(total_bytes)
    return None, None


def _cuda_memory_mib(value: Any) -> float:
    return float(value) / (1024.0 * 1024.0)


def _number_or_zero(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if numeric >= 0 else 0.0
