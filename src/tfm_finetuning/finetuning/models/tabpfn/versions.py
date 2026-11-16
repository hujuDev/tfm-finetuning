"""Data-only catalog for supported TabPFN fine-tuning versions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TabPFNVersion:
    """Checkpoint and display metadata for one selectable TabPFN model ID."""

    model_id: str
    display_name: str
    classifier_checkpoint: str
    regressor_checkpoint: str
    # Value of ``tabpfn.constants.ModelVersion`` passed to the fine-tuning wrapper, which otherwise defaults to the
    # latest model version.
    upstream_model_version: str | None = None


TABPFN_VERSIONS = (
    TabPFNVersion(
        model_id="tabpfn_v2",
        display_name="TabPFN v2",
        classifier_checkpoint="models/tabpfn-v2-classifier.ckpt",
        regressor_checkpoint="models/tabpfn-v2-regressor.ckpt",
        upstream_model_version="v2",
    ),
    TabPFNVersion(
        model_id="tabpfn_v2_5",
        display_name="TabPFN v2.5",
        classifier_checkpoint="models/tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
        regressor_checkpoint="models/tabpfn-v2.5-regressor-v2.5_default.ckpt",
        upstream_model_version="v2.5",
    ),
    TabPFNVersion(
        model_id="tabpfn_v3",
        display_name="TabPFN-3",
        classifier_checkpoint="models/tabpfn-v3-classifier-v3_default.ckpt",
        regressor_checkpoint="models/tabpfn-v3-regressor-v3_default.ckpt",
        upstream_model_version="v3",
    ),
    # TabPFN-3.5 ships one multitask checkpoint that serves classification and regression.
    TabPFNVersion(
        model_id="tabpfn_v3_5",
        display_name="TabPFN-3.5",
        classifier_checkpoint="models/tabpfn-v3.5-20260909.safetensors",
        regressor_checkpoint="models/tabpfn-v3.5-20260909.safetensors",
        upstream_model_version="v3.5",
    ),
    TabPFNVersion(
        model_id="tabpfn_v3_5_fast",
        display_name="TabPFN-3.5-Fast",
        classifier_checkpoint="models/tabpfn-v3.5-fast-20260909.safetensors",
        regressor_checkpoint="models/tabpfn-v3.5-fast-20260909.safetensors",
        upstream_model_version="v3.5-fast",
    ),
)

TABPFN_VERSION_BY_ID = {version.model_id: version for version in TABPFN_VERSIONS}

# The pinned vendor also contains the v2.6 architecture. It intentionally remains outside this catalog until
# configs, checkpoints, and fine-tuning parity tests make it a supported experiment target.
VENDOR_ONLY_TABPFN_ARCHITECTURES = ("tabpfn_v2_6",)


def get_tabpfn_version(model_id: str) -> TabPFNVersion | None:
    """Return supported TabPFN metadata, or ``None`` for another model family."""
    return TABPFN_VERSION_BY_ID.get(str(model_id).strip().lower())
