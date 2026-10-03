"""Independent transfer datasets and deterministic nested training row budgets."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tfm_finetuning.experiments.transfer.rows import apply_train_subset_to_split, build_nested_train_subsets
from tfm_finetuning.finetuning.datasets.preprocessing import _configure_adapter_preprocessing_protocol
from tfm_finetuning.finetuning.run.data import _prepare_foundation_run_data
from tfm_finetuning.utils.config import set_nested

_MODEL_DISTRIBUTIONS = {"tabpfn_v2_5": ("tabpfn",), "tabicl_v2": ("tabicl",), "mitra": ("autogluon.tabular",)}


class TransferDatasetLoader:
    """Prepare independently preprocessed dataset-native split objects on demand."""

    def __init__(self, setup: Any) -> None:
        self.setup = setup
        self._cache: dict[str, Any] = {}
        self.runtime_phases: list[dict[str, Any]] = []

    def reconfigured(self, config: Mapping[str, Any]) -> TransferDatasetLoader:
        """Return an independent loader for a reconstructed producer recipe."""
        producer_setup = SimpleNamespace(
            config=json.loads(json.dumps(config)),
            paths=self.setup.paths,
            spec=self.setup.spec,
            preprocessing_protocol=self.setup.preprocessing_protocol,
            resolved_seed=self.setup.resolved_seed,
        )
        return type(self)(producer_setup)

    def __call__(
        self,
        dataset_name: str,
        *,
        row_budget: int | None = None,
        row_budget_policy: str = "cap",
        row_subset_seed: int | None = None,
    ) -> Any:
        """Load one dataset and optionally apply its deterministic nested train prefix."""
        name = _clean_dataset_name(dataset_name)
        if name not in self._cache:
            adapter = self.setup.spec.adapter_cls(self.setup.spec)
            _configure_adapter_preprocessing_protocol(adapter, self.setup.preprocessing_protocol)
            dataset_config = config_for_dataset(
                self.setup.config,
                name,
                repo_root=self.setup.paths.repo_root,
            )
            data, _, comparison = _prepare_foundation_run_data(
                config=dataset_config,
                repo_root=self.setup.paths.repo_root,
                adapter=adapter,
                runtime_phases=self.runtime_phases,
                resolved_seed=int(self.setup.resolved_seed),
                preprocessing_protocol=self.setup.preprocessing_protocol,
                model_id=self.setup.spec.model_id,
            )
            if comparison is not None:
                raise ValueError("Section 3 transfer routes do not support alternate test-view comparisons.")
            self._cache[name] = data
        data = self._cache[name]
        effective_budget = effective_row_budget(
            available_rows=len(data.train_X),
            requested_rows=row_budget,
            policy=row_budget_policy,
        )
        if effective_budget == len(data.train_X):
            return data
        subset = build_nested_train_subsets(
            len(data.train_X),
            [effective_budget],
            seed=(int(self.setup.resolved_seed) if row_subset_seed is None else int(row_subset_seed)),
        )[effective_budget]
        return apply_train_subset_to_split(
            data,
            subset,
            include_validation_in_test_context=bool(data.metadata.get("test_context_includes_validation_data")),
        )


def effective_row_budget(
    *,
    available_rows: int,
    requested_rows: int | None,
    policy: str,
) -> int:
    """Resolve declared caps while preserving E2's exact-budget failure semantics."""
    available = int(available_rows)
    if available <= 0:
        raise ValueError("Transfer datasets require at least one available training row.")
    if requested_rows is None:
        return available
    requested = int(requested_rows)
    if requested <= 0:
        raise ValueError("Transfer row budgets must be positive.")
    if policy == "cap":
        return min(requested, available)
    if policy == "exact":
        if requested > available:
            raise ValueError(f"Exact transfer row budget {requested} exceeds the {available}-row training split.")
        return requested
    raise ValueError(f"Unknown transfer row-budget policy: {policy!r}.")


def transfer_package_versions(model_id: str) -> dict[str, str]:
    """Record library versions used by model-specific stage trainers."""
    distributions = ("torch", "numpy", *_MODEL_DISTRIBUTIONS.get(str(model_id), ()))
    versions = {
        "python": platform.python_version(),
    }
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def config_for_dataset(
    config: Mapping[str, Any],
    dataset_name: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    name = _clean_dataset_name(dataset_name)
    from tfm_finetuning.experiments.datasets import ensure_dataset

    target_path = ensure_dataset(name, copied, repo_root=repo_root)
    set_nested(copied, "data.path", str(target_path))
    return copied


def _clean_dataset_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"Invalid dataset name: {value!r}.")
    return text
