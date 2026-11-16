# Third-party software

This repository's original code is licensed under the MIT license.

- The implementation uses TabArena, TabPFN, TabICL, AutoGluon/Mitra, PyTorch, Submitit, and other dependencies declared in `pyproject.toml` and pinned in `uv.lock` (TabArena and TabICL at Git commits, `tabpfn` 9.2.0 from PyPI)
    - Their original licenses and notices remain applicable. Upstream source trees and pretrained model weights are not redistributed here

- Datasets and model weights have their own licenses and access conditions. Obtain them from their providers. The MIT license for this repository does not relicense those resources or its dependencies.

## Model weights

The TabICL and Mitra weights are downloaded from Hugging Face on first use; the TabPFN weights are fetched and verified with `python -m tfm_finetuning.data.download.tabpfn_checkpoints`. They are neither redistributed nor relicensed here, and neither are fine-tuned checkpoints. Verify each license before reuse; the table summarizes the licenses as published on the model cards.

| Model | Source | License | Commercial use |
| --- | --- | --- | --- |
| TabPFNv2 | Hugging Face `Prior-Labs/TabPFN-v2-clf`, `Prior-Labs/TabPFN-v2-reg` | Prior Labs License 1.1 (Apache-2.0 with an attribution requirement) | Allowed |
| TabPFN-2.5 | Hugging Face `Prior-Labs/tabpfn_2_5` | TabPFN-2.5 License v1.1 (non-commercial) | No |
| TabPFN-3 | Hugging Face `Prior-Labs/tabpfn_3` | TABPFN-3 License v1.0 (non-commercial) | No |
| TabPFN-3.5, TabPFN-3.5-Fast | Hugging Face `Prior-Labs/tabpfn_3_5` | TABPFN-3.5 License v1.0 (non-commercial) | No |
| TabICLv2 | Hugging Face `jingang/TabICL` | BSD-3-Clause | Allowed |
| Mitra | Hugging Face `autogluon/mitra-classifier`, `autogluon/mitra-regressor` | Apache-2.0 | Allowed |
| Mitra-v2 | Hugging Face `autogluon/mitra-classifier-2`, `autogluon/mitra-regressor-2` | Apache-2.0 | Allowed |

- The TabPFN-2.5, TabPFN-3 and TabPFN-3.5 licenses permit testing, evaluation, internal benchmarking and academic research, and exclude commercial or production use. Fine-tuned models are Derivatives and carry the same restriction. Distributing weights or Derivatives requires a copy of the license and the attribution notice, and hosting them as a service requires a commercial license
- The TabPFN-3.5-Plus, TabPFN-3.5-Thinking, TabPFN-3-Plus and TabPFN-3-Thinking variants are available only through Prior Labs' API. They cannot be fine-tuned locally and are not used here
- All TabPFN models run with `tabpfn` 9.2; the published TabPFNv2 and TabPFN-2.5 runs used `tabpfn` 8.0.3. The code of the `tabpfn` package is released under Apache-2.0
