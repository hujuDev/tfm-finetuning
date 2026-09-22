"""Role-aware distribution-shift transforms for unified foundation-model runs.

Example:
    >>> from finetuning.distribution_shift import apply_pre_finite_distribution_shift
    >>> callable(apply_pre_finite_distribution_shift)
    True
"""

from .role_transforms import (
    TargetTransformState,
    apply_post_finite_distribution_shift,
    apply_post_finite_distribution_shift_with_comparison,
    apply_pre_finite_distribution_shift,
    inverse_transform_metric_prediction,
)
from .row_sampling import apply_test_context_majority_downsampling

__all__ = [
    "TargetTransformState",
    "apply_post_finite_distribution_shift",
    "apply_post_finite_distribution_shift_with_comparison",
    "apply_pre_finite_distribution_shift",
    "apply_test_context_majority_downsampling",
    "inverse_transform_metric_prediction",
]
