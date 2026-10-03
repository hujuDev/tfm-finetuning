"""Typed access to the installed TabICL classes used by fine-tuning."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetuning.models.registry_sources import ensure_tabicl_on_path
from utils.paths import get_repo_root


@dataclass(frozen=True)
class TabICLBindings:
    classifier: type[Any]
    regressor: type[Any]
    finetuned_classifier: type[Any]
    finetuned_regressor: type[Any]

    def estimator_class(self, *, regression: bool) -> type[Any]:
        """Return the zero-shot classifier or regressor."""
        return self.regressor if regression else self.classifier

    def finetuner_class(self, *, regression: bool) -> type[Any]:
        """Return the fine-tuning classifier or regressor."""
        return self.finetuned_regressor if regression else self.finetuned_classifier


def load_tabicl_bindings() -> TabICLBindings:
    """Load the pinned TabICL sklearn and fine-tuning wrappers."""
    ensure_tabicl_on_path(repo_root=get_repo_root(Path(__file__)))
    tabicl = importlib.import_module("tabicl")
    return TabICLBindings(
        classifier=tabicl.TabICLClassifier,
        regressor=tabicl.TabICLRegressor,
        finetuned_classifier=tabicl.FinetunedTabICLClassifier,
        finetuned_regressor=tabicl.FinetunedTabICLRegressor,
    )
