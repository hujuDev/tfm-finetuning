"""Fetch the four thesis TabPFN weights: ``python -m tfm_finetuning.data.download.tabpfn_checkpoints``."""

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
    """An immutable upstream file and the SHA-256 of the thesis checkpoint."""

    version: str
    repo_id: str
    revision: str
    filename: str
    sha256: str


# Verified against the original thesis files and Hugging Face LFS metadata.
# Explicit v2 classifier selection matters: the package default has since changed.
CHECKPOINTS = (
    CheckpointSpec(
        "v2", "Prior-Labs/TabPFN-v2-clf", "a7e9a4696dd0efbd0e64a705645e0d137ca2de72",
        "tabpfn-v2-classifier.ckpt", "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f",
    ),
    CheckpointSpec(
        "v2", "Prior-Labs/TabPFN-v2-reg", "4972a65a1b30806315c6f92499959ffbfc69a673",
        "tabpfn-v2-regressor.ckpt", "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
    ),
    CheckpointSpec(
        "v2.5", "Prior-Labs/tabpfn_2_5", "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
        "tabpfn-v2.5-classifier-v2.5_default-2.ckpt", "2d0fbd257a5a274f2a219836d5a68e9c27224845936b4685d85dd42e2b8092aa",
    ),
    CheckpointSpec(
        "v2.5", "Prior-Labs/tabpfn_2_5", "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
        "tabpfn-v2.5-regressor-v2.5_default.ckpt", "6e0308b7e5f3a8325b798b96d986b2b282c3608b1cda6e2991425a2a261d40b8",
    ),
)


def _verify(path: Path, spec: CheckpointSpec) -> None:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != spec.sha256:
        raise ValueError(f"{path}: checkpoint checksum mismatch; use --overwrite to fetch the thesis weights")


def download_checkpoint(spec: CheckpointSpec, models_root: Path, *, overwrite: bool = False) -> str:
    """Verify an existing checkpoint or fetch its pinned revision with native HF authentication."""
    destination = models_root / spec.filename
    if destination.exists() and not overwrite:
        _verify(destination, spec)
        return "skipped (verified existing file)"
    # Leaving token unset preserves HF_TOKEN and the user's saved Hugging Face login.
    cached = Path(hf_hub_download(
        repo_id=spec.repo_id, filename=spec.filename, revision=spec.revision, force_download=overwrite,
    ))
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


def main(argv: list[str] | None = None) -> None:
    """Download all thesis weights, or select ``--group v2`` / ``--group v2.5``."""
    parser = argparse.ArgumentParser(description="Download and verify the four TabPFN checkpoints used in the thesis.")
    parser.add_argument("--models-root", type=Path, default=get_repo_root() / "models")
    parser.add_argument("--group", choices=("all", "v2", "v2.5"), default="all")
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
