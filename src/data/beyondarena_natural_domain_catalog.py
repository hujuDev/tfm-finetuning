"""Frozen BeyondArena sources and logical same-schema domain panels.

The catalog is deliberately outcome-independent: domain membership is fixed by
the released cohort/site column, and the WiDS hospitals are the three largest
sites satisfying the predeclared class-support rule.

Example:
    >>> panel_for_dataset("heart_disease_cleveland").key
    'heart'
"""

from __future__ import annotations

from dataclasses import dataclass

BEYONDARENA_REPO_ID = "TabArena/BeyondArena"
BEYONDARENA_REVISION = "2ecfe882ccfb814fc27c4de10a64ceefd5d7655c"


@dataclass(frozen=True)
class BeyondArenaContainer:
    """One pinned Data Foundry container in the BeyondArena release."""

    dataset: str
    container_id: str

    @property
    def relative_path(self) -> str:
        """Return the repository-relative container directory."""
        return f"{self.dataset}/{self.container_id}"


@dataclass(frozen=True)
class NaturalDomainPanel:
    """Three compatible logical datasets rotated through the target role."""

    key: str
    datasets: tuple[str, str, str]
    domain_kind: str


CONTAINERS = {
    "heart_disease_cleveland": BeyondArenaContainer("heart_disease_cleveland", "019d736b-4900-714b-8dfc-be588ce802c3"),
    "heart_disease_hungary": BeyondArenaContainer("heart_disease_hungary", "019d736b-5a1a-7c95-92f0-8c2fe5334ad2"),
    "heart_disease_va_long_beach": BeyondArenaContainer(
        "heart_disease_va_long_beach", "019d736b-7c92-71e1-a6d6-27a91ed15d3d"
    ),
    "wids_diabetes_mellitus": BeyondArenaContainer("wids_diabetes_mellitus", "019d7374-50ba-7416-9d10-a526040930ee"),
    "covertype": BeyondArenaContainer("covertype", "019d7391-8e33-757b-b38a-79ec188fa001"),
}

WIDS_HOSPITAL_IDS = (118, 19, 188)
WIDS_DOMAIN_DATASETS = (
    "wids_diabetes_hospital_118",
    "wids_diabetes_hospital_19",
    "wids_diabetes_hospital_188",
)
WILDERNESS_DOMAIN_DATASETS = (
    "covertype_comanche_peak",
    "covertype_neota",
    "covertype_rawah",
)
WILDERNESS_DOMAIN_LABELS = {
    "covertype_comanche_peak": "Comanche Peak Wilderness Area",
    "covertype_neota": "Neota Wilderness Area",
    "covertype_rawah": "Rawah Wilderness Area",
}
GROUPED_COVERTYPE_DATASET = "covertype_grouped_ood"

NATURAL_DOMAIN_PANELS = (
    NaturalDomainPanel(
        key="heart",
        datasets=(
            "heart_disease_cleveland",
            "heart_disease_hungary",
            "heart_disease_va_long_beach",
        ),
        domain_kind="separate_cohort",
    ),
    NaturalDomainPanel(
        key="wids_hospital",
        datasets=WIDS_DOMAIN_DATASETS,
        domain_kind="custom_hospital",
    ),
    NaturalDomainPanel(
        key="covertype_area",
        datasets=WILDERNESS_DOMAIN_DATASETS,
        domain_kind="wilderness_area",
    ),
)


def panel_for_dataset(dataset_name: str) -> NaturalDomainPanel:
    """Return the unique natural-domain panel containing ``dataset_name``."""
    matches = [panel for panel in NATURAL_DOMAIN_PANELS if dataset_name in panel.datasets]
    if len(matches) != 1:
        raise ValueError(f"Dataset {dataset_name!r} is not in exactly one natural-domain panel.")
    return matches[0]


__all__ = [
    "BEYONDARENA_REPO_ID",
    "BEYONDARENA_REVISION",
    "CONTAINERS",
    "GROUPED_COVERTYPE_DATASET",
    "NATURAL_DOMAIN_PANELS",
    "WIDS_HOSPITAL_IDS",
    "WILDERNESS_DOMAIN_LABELS",
    "BeyondArenaContainer",
    "NaturalDomainPanel",
    "panel_for_dataset",
]
