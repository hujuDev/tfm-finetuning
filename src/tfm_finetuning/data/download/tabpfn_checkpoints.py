"""Fetch the TabPFN weights: ``python -m tfm_finetuning.data.download.tabpfn_checkpoints``."""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from huggingface_hub import hf_hub_download

from tfm_finetuning.utils.paths import get_repo_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointSpec:
    """An immutable upstream file and the SHA-256 of the checkpoint used in the published runs."""

    version: str
    repo_id: str
    revision: str
    filename: str
    sha256: str


# The v2 and v2.5 entries are verified against the files of the published runs and Hugging Face LFS metadata.
# Explicit v2 classifier selection matters: the package default has since changed.
# The v3 and v3.5 entries carry the SHA-256 published in the Hugging Face LFS metadata of the pinned revision.
# v3.5 and v3.5-Fast are single multitask checkpoints. The v3 repo and the v3.5 repo are not gated, but their
# licenses (tabpfn-3-license-v1.0, tabpfn-3-5-license-v1.0) are non-commercial.
CHECKPOINTS = (
    CheckpointSpec(
        "v2",
        "Prior-Labs/TabPFN-v2-clf",
        "a7e9a4696dd0efbd0e64a705645e0d137ca2de72",
        "tabpfn-v2-classifier.ckpt",
        "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f",
    ),
    CheckpointSpec(
        "v2",
        "Prior-Labs/TabPFN-v2-reg",
        "4972a65a1b30806315c6f92499959ffbfc69a673",
        "tabpfn-v2-regressor.ckpt",
        "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
    ),
    CheckpointSpec(
        "v2.5",
        "Prior-Labs/tabpfn_2_5",
        "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
        "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
        "2d0fbd257a5a274f2a219836d5a68e9c27224845936b4685d85dd42e2b8092aa",
    ),
    CheckpointSpec(
        "v2.5",
        "Prior-Labs/tabpfn_2_5",
        "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
        "tabpfn-v2.5-regressor-v2.5_default.ckpt",
        "6e0308b7e5f3a8325b798b96d986b2b282c3608b1cda6e2991425a2a261d40b8",
    ),
    CheckpointSpec(
        "v3",
        "Prior-Labs/tabpfn_3",
        "24a16a89d245878b846555110985634aa2e656d7",
        "tabpfn-v3-classifier-v3_default.ckpt",
        "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
    ),
    CheckpointSpec(
        "v3",
        "Prior-Labs/tabpfn_3",
        "24a16a89d245878b846555110985634aa2e656d7",
        "tabpfn-v3-regressor-v3_default.ckpt",
        "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
    ),
    CheckpointSpec(
        "v3.5",
        "Prior-Labs/tabpfn_3_5",
        "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
        "tabpfn-v3.5-20260909.safetensors",
        "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    ),
    CheckpointSpec(
        "v3.5",
        "Prior-Labs/tabpfn_3_5",
        "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
        "tabpfn-v3.5-fast-20260909.safetensors",
        "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550",
    ),
)


def _verify(path: Path, spec: CheckpointSpec) -> None:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != spec.sha256:
        raise ValueError(f"{path}: checkpoint checksum mismatch; use --overwrite to fetch the pinned weights")


def download_checkpoint(spec: CheckpointSpec, models_root: Path, *, overwrite: bool = False) -> str:
    """Verify an existing checkpoint or fetch its pinned revision with native HF authentication."""
    destination = models_root / spec.filename
    if destination.exists() and not overwrite:
        _verify(destination, spec)
        return "skipped (verified existing file)"
    # Leaving token unset preserves HF_TOKEN and the user's saved Hugging Face login.
    cached = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            force_download=overwrite,
        )
    )
    _verify(cached, spec)
    models_root.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=models_root, prefix=spec.filename + ".", suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(cached, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def ensure_checkpoint(path: Path) -> None:
    """Fetch a missing pinned checkpoint on first use; existing files and unpinned paths are left as they are."""
    if path.exists():
        return
    spec = next((spec for spec in CHECKPOINTS if spec.filename == path.name), None)
    if spec is None:
        return
    logger.info("Fetching the missing TabPFN checkpoint %s", spec.filename)
    download_checkpoint(spec, path.parent)


def main(argv: list[str] | None = None) -> None:
    """Download all weights, or select ``--group v2`` / ``v2.5`` / ``v3`` / ``v3.5`` (v3.5 includes v3.5-Fast)."""
    parser = argparse.ArgumentParser(description="Download and verify the pinned TabPFN checkpoints.")
    parser.add_argument("--models-root", type=Path, default=get_repo_root() / "models")
    parser.add_argument("--group", choices=("all", "v2", "v2.5", "v3", "v3.5"), default="all")
    parser.add_argument("--overwrite", action="store_true", help="Fetch and replace existing checkpoints.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failed = False
    for spec in CHECKPOINTS:
        if args.group != "all" and spec.version != args.group:
            continue
        try:
            status = download_checkpoint(spec, args.models_root, overwrite=args.overwrite)
            logger.info("%s: %s", spec.filename, status)
        except Exception as exc:
            failed = True
            logger.error("%s: %s", spec.filename, exc)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
