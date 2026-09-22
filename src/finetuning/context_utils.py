from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_context_subsample(
    config: Mapping[str, Any],
) -> tuple[float | int | None, float | int | None, float | int | None, float | int | None]:
    """Resolve context subsample sizes for train/eval/val/test from config defaults."""
    context_subsample_train = config.get("context_subsample_train", config.get("context_subsample"))
    context_subsample_eval = config.get(
        "context_subsample_eval",
        context_subsample_train if context_subsample_train is not None else config.get("context_subsample"),
    )
    context_subsample_val = config.get("context_subsample_val", context_subsample_eval)
    context_subsample_test = config.get("context_subsample_test", context_subsample_eval)
    return context_subsample_train, context_subsample_eval, context_subsample_val, context_subsample_test
