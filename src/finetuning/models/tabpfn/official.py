"""Load the pinned upstream TabPFN implementation without local compatibility stubs.

Example: ``import_official_tabpfn(include_finetuning=True)`` resolves training classes.
"""
from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any


def ensure_official_tabpfn_on_path(*, repo_root: Path | None = None) -> Path:
    """Locate the installed upstream package and configure its checkpoint cache."""
    root = repo_root or Path(__file__).resolve().parents[4]
    os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', str(root / 'models'))
    spec = find_spec('tabpfn')
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError('TabPFN is not installed; run uv sync --locked')
    return Path(spec.origin).resolve().parent


def import_official_tabpfn(*, repo_root: Path | None = None, include_finetuning: bool = False) -> dict[str, Any]:
    """Import the actual model classes; a missing dependency must be installed."""
    ensure_official_tabpfn_on_path(repo_root=repo_root)
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    payload = {'TabPFNClassifier': TabPFNClassifier, 'TabPFNRegressor': TabPFNRegressor}
    if include_finetuning:
        from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
        from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
        from tabpfn.finetuning.train_util import clone_model_for_evaluation
        from tabpfn.model_loading import save_tabpfn_model
        payload.update(FinetunedTabPFNClassifier=FinetunedTabPFNClassifier,
                       FinetunedTabPFNRegressor=FinetunedTabPFNRegressor,
                       clone_model_for_evaluation=clone_model_for_evaluation,
                       save_tabpfn_model=save_tabpfn_model)
    return payload
