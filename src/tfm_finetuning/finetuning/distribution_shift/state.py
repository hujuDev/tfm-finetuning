"""Shared state objects for role-aware distribution interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class TargetTransformState:
    """Prediction inverse transform for original-scale regression scoring.

    Example:
        >>> state = TargetTransformState(mode="robust_log1p", shift=2.0)
        >>> round(float(state.inverse(np.array([0.0]))[0]), 6)
        -1.0
    """

    mode: str
    shift: float = 0.0
    lower: float | None = None
    upper: float | None = None
    report: Mapping[str, Any] = field(default_factory=dict)

    def inverse(self, prediction: Any) -> Any:
        """Return predictions on the original target scale."""
        if self.mode == "robust_log1p":
            return np.expm1(np.asarray(prediction, dtype=float)) - float(self.shift)
        return prediction
