from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_MISSING_OUTPUT_DIR_WARNING = r"`output_dir` is not set\..*"
_FEATURE_NAME_WARNING = (
    r"X has feature names, but TabPFN(?:Classifier|Regressor) "
    r"was fitted without feature names"
)
_MISSING_RESUME_CHECKPOINT_WARNING = (
    r"Output dir present but no checkpoint file found for train_size=.*"
    r"Starting from default checkpoint and epoch 0"
)


@contextlib.contextmanager
def suppress_tabpfn_user_warning_noise(
    *,
    feature_name_mismatch: bool = False,
    missing_output_dir: bool = False,
    missing_resume_checkpoint: bool = False,
) -> Iterator[None]:
    """Suppress expected upstream warning noise around wrapper-managed TabPFN calls.

    Example:
        with suppress_tabpfn_user_warning_noise(
            feature_name_mismatch=True,
            missing_resume_checkpoint=True,
        ):
            finetuner.fit(X_train, y_train, output_dir=output_dir)
    """
    with warnings.catch_warnings():
        if missing_output_dir:
            warnings.filterwarnings(
                "ignore",
                message=_MISSING_OUTPUT_DIR_WARNING,
                category=UserWarning,
            )
        if feature_name_mismatch:
            warnings.filterwarnings(
                "ignore",
                message=_FEATURE_NAME_WARNING,
                category=UserWarning,
            )
        if missing_resume_checkpoint:
            warnings.filterwarnings(
                "ignore",
                message=_MISSING_RESUME_CHECKPOINT_WARNING,
                category=UserWarning,
            )
        yield
