# Fine-tuning tabular foundation models

Reproduction code for my master’s thesis

- The experiments compare ICL with and without supervised fine-tuning for **TabPFNv2, TabPFN-2.5, TabICLv2, and Mitra**. They cover dataset characteristics, controlled transformations, and cross-dataset transfer.

## Setup

- Use Linux, Python 3.12, and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/hujuDev/tfm-finetuning.git
cd tfm-finetuning
uv sync --locked
```

- Full experiments require a suitable CUDA GPU
- Model weights and datasets are downloaded separately and keep their upstream licenses
    - Missing TabArena datasets and model checkpoints are fetched automatically on first use, so the first run needs network access (a Hugging Face login or `HF_TOKEN` may be required for some weights)
- Optional settings (W&B, API keys, paths) go in a local `.env`; see [.env.example](.env.example)

## Run the experiments
- Start with the [introductory notebook](notebooks/introductory.ipynb) for a small example:

```bash
uv run jupyter lab notebooks/introductory.ipynb
```

- Then use the [experiments notebook](notebooks/hydra_experiments.ipynb) to select, preview and launch the thesis experiments directly through Submitit, either locally or on SLURM
    - The supplied HPC setup targets **HYDRA at TU Berlin**. It should also work on other SLURM clusters with some configuration

The author’s code is available under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream software.

## Citation

If you use this code, please cite the thesis. Add its title and link here once published.
