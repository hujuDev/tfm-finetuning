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
- Optional settings (W&B, API keys, paths) go in a local `.env` (see [.env.example](.env.example))

## Run the experiments
- Launch the thesis experiments from the command line. All experiments use the same launcher, locally or on SLURM. Preview the runs first (the default):

```bash
uv run python -m tfm_finetuning.experiments.execution.launch distribution_shift_label_noise --local --dataset jm1
```

- Run them locally:

```bash
uv run python -m tfm_finetuning.experiments.execution.launch distribution_shift_label_noise --local --dataset jm1 --run
```

- Or submit them to SLURM:

```bash
uv run python -m tfm_finetuning.experiments.execution.launch distribution_shift_label_noise --cluster configs/slurm/hydra_cluster.yaml --submit
```

- Use `--list` to list the experiments and `--help` for all options. The Bash scripts in [scripts/experiments](scripts/experiments) wrap the launcher for each experiment
- The supplied HPC setup targets **HYDRA at TU Berlin**. It should also work on other SLURM clusters with some configuration

- Alternatively, use the Jupyter notebooks. Start with the [introductory notebook](notebooks/introductory.ipynb) for a small example:

```bash
uv run jupyter lab notebooks/introductory.ipynb
```

- Then use the [experiments notebook](notebooks/launch_experiments.ipynb) to select, preview and launch the same experiments interactively

## Repository layout

- `src/tfm_finetuning/`
    - `finetuning/`: fine-tuning pipeline, model adapters (TabPFN, TabICL, Mitra) and evaluation
    - `experiments/`: experiment definitions, launching (local or SLURM) and cross-dataset transfer
    - `data/`: dataset download and preprocessing
    - `plots/`: thesis figures and tables
    - `utils/`: paths and configuration helpers
- `configs/`: experiment definitions, model defaults, SLURM settings and plot configurations
- `scripts/`: Bash scripts for launching the experiments (`scripts/experiments/`)
- `notebooks/`: introductory example and the experiments launcher
- `docs/`: notes on running the transfer experiments
- `tests/`: unit tests mirroring the source layout

The code is available under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for the licenses of the libraries it builds on, the model weights and the datasets.
