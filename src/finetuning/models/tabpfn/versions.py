"""Data-only catalog for thesis-supported TabPFN fine-tuning versions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TabPFNVersion:
    """Checkpoint and display metadata for one selectable TabPFN model ID."""

    model_id: str
    display_name: str
    classifier_checkpoint: str
    regressor_checkpoint: str


TABPFN_VERSIONS = (
    TabPFNVersion(
        model_id="tabpfn_v2",
        display_name="TabPFN v2",
        classifier_checkpoint="models/tabpfn-v2-classifier.ckpt",
        regressor_checkpoint="models/tabpfn-v2-regressor.ckpt",
    ),
    TabPFNVersion(
        model_id="tabpfn_v2_5",
        display_name="TabPFN v2.5",
        classifier_checkpoint="models/tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
        regressor_checkpoint="models/tabpfn-v2.5-regressor-v2.5_default.ckpt",
    ),
)

TABPFN_VERSION_BY_ID = {version.model_id: version for version in TABPFN_VERSIONS}

# The pinned vendor also contains v2.6 and v3 architectures. They intentionally
# remain outside this catalog until thesis configs, checkpoints, and fine-tuning
# parity tests make them supported experiment targets.
VENDOR_ONLY_TABPFN_ARCHITECTURES = ("tabpfn_v2_6", "tabpfn_v3")


def get_tabpfn_version(model_id: str) -> TabPFNVersion | None:
    """Return supported TabPFN metadata, or ``None`` for another model family."""
    return TABPFN_VERSION_BY_ID.get(str(model_id).strip().lower())
