"""Binned regression for Mitra-v2's 1,000-bin regression head.

The Mitra-v2 regressor (``autogluon/mitra-regressor-2``) predicts a distribution over value bins instead of a
single value. AutoGluon's Mitra trainer already bins the min-max normalized target and trains with cross entropy
when ``regression_loss`` is ``CROSS_ENTROPY``, but its public estimator always configures MSE, and it decodes the
bins by their argmax. Following Amazon's ``mitra-finetune`` recipe (Apache-2.0), regression with a binned head
switches the loss and decodes the softmax-weighted mean of the bin centers. Mitra-v1 regression uses MSE and never
reaches the patched branches.

Example: ``install_mean_bin_decode()`` once per process, before any Mitra trainer predicts.
"""

from __future__ import annotations

import importlib
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

# Each vendor method decodes the bin logits in two consecutive lines; the mean decode replaces exactly those.
_DECODES = {
    "train": (
        "y_hat = torch.argmax(y_hat, dim=-1)",
        "y_hat = self.bins[y_hat] + self.bin_width / 2",
        "y_hat = mean_bin_decode(self, y_hat)",
    ),
    "evaluate": (
        "y_hat = torch.argmax(y_hat, dim=-1)",
        "y_hat = self.bins[y_hat] + self.bin_width / 2",
        "y_hat = mean_bin_decode(self, y_hat)",
    ),
    "predict": (
        "y_hat = np.argmax(y_hat, axis=-1)",
        "y_hat = (self.bins[y_hat] + self.bin_width / 2).cpu().numpy()",
        "y_hat = mean_bin_decode(self, torch.as_tensor(y_hat)).cpu().numpy()",
    ),
}


def mean_bin_decode(trainer: Any, logits: Any) -> Any:
    """Return the softmax-weighted mean of the bin centers, in the trainer's normalized target space."""
    torch = importlib.import_module("torch")
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    centers = (trainer.bins[:-1] + trainer.bin_width / 2).to(logits.device)
    return (torch.softmax(logits, dim=-1) * centers).sum(dim=-1)


def install_mean_bin_decode() -> None:
    """Make AutoGluon's Mitra trainer decode binned regression by the mean over bins (idempotent)."""
    module = importlib.import_module("autogluon.tabular.models.mitra._internal.core.trainer_finetune")
    trainer_cls = module.TrainerFinetune
    if getattr(trainer_cls, "_tfm_mean_bin_decode", False):
        return
    module.mean_bin_decode = mean_bin_decode
    for name, (argmax_line, bins_line, replacement) in _DECODES.items():
        setattr(trainer_cls, name, _rewrite_decode(getattr(trainer_cls, name), argmax_line, bins_line, replacement))
    trainer_cls._tfm_mean_bin_decode = True


def _rewrite_decode(method: Callable[..., Any], argmax_line: str, bins_line: str, replacement: str) -> Callable:
    """Swap one method's two-line argmax decode for the mean decode, leaving the rest of its source unchanged."""
    lines = textwrap.dedent(inspect.getsource(method)).split("\n")
    index = next((i for i, line in enumerate(lines) if line.strip() == argmax_line), None)
    if index is None or lines[index + 1].strip() != bins_line:
        raise RuntimeError(
            f"AutoGluon's Mitra bin decode in {method.__qualname__} changed; update regression_bins.py."
        )
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    lines[index : index + 2] = [indent + replacement]
    namespace: dict[str, Any] = {}
    code = compile("\n".join(lines), f"<tfm mean bin decode {method.__qualname__}>", "exec")
    exec(code, method.__globals__, namespace)  # noqa: S102 - recompiles the vendor method with the decode swapped
    return namespace[method.__name__]


def binned_regression_config(cfg: Any, regression_bins: int) -> Any:
    """Switch a regression config to cross entropy over ``regression_bins`` bins; other configs are unchanged."""
    if int(regression_bins) <= 1:
        return cfg
    enums = importlib.import_module("autogluon.tabular.models.mitra._internal.config.enums")
    if cfg.task == enums.Task.REGRESSION:
        cfg.hyperparams["regression_loss"] = enums.LossName.CROSS_ENTROPY
        cfg.hyperparams["dim_output"] = int(regression_bins)
    return cfg
