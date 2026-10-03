"""Typed access to the official TabPFN classes used by fine-tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tfm_finetuning.finetuning.models.tabpfn.official import import_official_tabpfn
from tfm_finetuning.utils.paths import get_repo_root


@dataclass(frozen=True)
class TabPFNBindings:
    """Official TabPFN classes and functions called by :class:`TabPFNAdapter`."""

    classifier: type[Any]
    regressor: type[Any]
    finetuned_classifier: type[Any]
    finetuned_regressor: type[Any]
    clone_model_for_evaluation: Callable[..., Any]
    save_model: Callable[..., Any]

    def estimator_class(self, *, regression: bool) -> type[Any]:
        """Return the zero-shot classifier or regressor."""
        return self.regressor if regression else self.classifier

    def finetuner_class(self, *, regression: bool) -> type[Any]:
        """Return the fine-tuning classifier or regressor."""
        return self.finetuned_regressor if regression else self.finetuned_classifier


def load_tabpfn_bindings() -> TabPFNBindings:
    """Load the pinned official TabPFN fine-tuning surface."""
    imports = import_official_tabpfn(
        repo_root=get_repo_root(Path(__file__)),
        include_finetuning=True,
    )
    return TabPFNBindings(
        classifier=imports["TabPFNClassifier"],
        regressor=imports["TabPFNRegressor"],
        finetuned_classifier=imports["FinetunedTabPFNClassifier"],
        finetuned_regressor=imports["FinetunedTabPFNRegressor"],
        clone_model_for_evaluation=imports["clone_model_for_evaluation"],
        save_model=imports["save_tabpfn_model"],
    )
