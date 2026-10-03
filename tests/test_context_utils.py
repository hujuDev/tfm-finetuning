from __future__ import annotations

from finetuning.datasets.context_utils import resolve_context_subsample


def test_eval_context_defaults_to_train_context() -> None:
    config = {"context_subsample_train": 512}
    train, eval_, val, test = resolve_context_subsample(config)
    assert eval_ == train
    assert val == train
    assert test == train
