"""Typed access to AutoGluon's public Mitra sklearn estimators."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tfm_finetuning.finetuning.models.registry_sources import ensure_mitra_on_path
from tfm_finetuning.utils.paths import get_repo_root


@dataclass(frozen=True)
class MitraBindings:
    classifier: type[Any]
    regressor: type[Any]

    def estimator_class(self, *, regression: bool) -> type[Any]:
        """Return the classifier or regressor selected by task kind."""
        return self.regressor if regression else self.classifier


def load_mitra_bindings() -> MitraBindings:
    """Load the pinned AutoGluon Mitra sklearn surface."""
    ensure_mitra_on_path(repo_root=get_repo_root(Path(__file__)))
    module = importlib.import_module("autogluon.tabular.models.mitra.sklearn_interface")
    return MitraBindings(
        classifier=module.MitraClassifier,
        regressor=module.MitraRegressor,
    )
