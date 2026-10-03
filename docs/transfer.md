# Transfer experiment prerequisites

Transfer runs keep their configured update budgets, preprocess datasets independently
and select checkpoints using validation data. Pooled training alternates dataset-specific
episodes. Each sequential stage starts a fresh optimizer from the preceding stage's
selected weights. Context evaluations change the labeled ICL context without further
training.

Finish the required training stages before starting experiments that reuse their weights:

| Run first | Then run |
| --- | --- |
| [Training order](../configs/experiments/dataset_transfer.yaml) | [Target-data budgets](../configs/experiments/dataset_transfer_data_budget.yaml), [balanced classification budgets](../configs/experiments/dataset_transfer_balanced_budget_classification.yaml), [source-only classification](../configs/experiments/dataset_transfer_source_only_eval_classification.yaml) |
| [Houses source checkpoint](../configs/experiments/dataset_transfer_houses_source_checkpoint.yaml) | [Balanced regression budgets](../configs/experiments/dataset_transfer_balanced_budget_regression.yaml), [source-only regression](../configs/experiments/dataset_transfer_source_only_eval_regression.yaml) |
| [Natural-domain training](../configs/experiments/dataset_transfer_natural_domain_training.yaml) | [Natural-domain context evaluation](../configs/experiments/dataset_transfer_natural_domain.yaml) |
| [Retention training](../configs/experiments/dataset_transfer_learned_retention_train.yaml) | [Retention evaluation](../configs/experiments/dataset_transfer_learned_retention_eval.yaml) |

Keep these runs under the same results root. Checkpoints must come from completed runs
with matching model, preprocessing, seed, split, row budgets and training settings.
Missing checkpoint files cause an error. The evaluation does not silently retrain them.
Set `dataset_transfer.checkpoint_path` to select weights manually.

The Bank/Covertype retention comparisons also use the Covertype target-only 100-update
reference from `dataset_transfer` and Bank's 100-update reference from
[dataset_transfer_target_controls](../configs/experiments/dataset_transfer_target_controls.yaml).
Shorter-budget references are selected using validation within their own update cap.
They share an optimization path only when learning-rate scheduling and early stopping
are disabled.
