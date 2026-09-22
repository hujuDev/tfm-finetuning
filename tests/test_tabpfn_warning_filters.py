from __future__ import annotations

import warnings

from src.finetuning.tabpfn_warning_filters import suppress_tabpfn_user_warning_noise


def test_warning_filter_suppresses_missing_resume_checkpoint_warning() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with suppress_tabpfn_user_warning_noise(missing_resume_checkpoint=True):
            warnings.warn(
                "Output dir present but no checkpoint file found for train_size=2666. "
                "Starting from default checkpoint and epoch 0",
                UserWarning,
                stacklevel=1,
            )

    assert recorded == []


def test_warning_filter_keeps_unrelated_user_warnings_visible() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with suppress_tabpfn_user_warning_noise(missing_resume_checkpoint=True):
            warnings.warn("keep me", UserWarning, stacklevel=1)

    assert len(recorded) == 1
    assert str(recorded[0].message) == "keep me"
