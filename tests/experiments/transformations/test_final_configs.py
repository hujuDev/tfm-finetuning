from pathlib import Path

import yaml

from tfm_finetuning.experiments.configuration import expand_runs, recipe_groups

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / 'configs/experiments'


def _load(name):
    payload = yaml.safe_load((CONFIG_ROOT / name.removeprefix('foundation_models_')).read_text())
    # Scientific assertions below inspect the first fully resolved resource group.
    return recipe_groups(payload)[0]


def test_contextual_scale_config_uses_original_and_matched_full_role_conditions() -> None:
    payload = _load("foundation_models_distribution_shift_contextual_scale.yaml")
    conditions = payload["axes"]["contextualScaleCondition"]

    assert [entry["value"] for entry in conditions] == [
        "scale_1x_no_unit_column",
        "scale_1x_with_unit_column",
        "scale_sx_no_unit_column",
        "scale_sx_with_unit_column",
    ]
    assert payload["settings"]["preprocess.contextual_scale.scale_spread"] == "calibrated"
    assert payload["settings"]["preprocess.contextual_scale.scale_spread_by_dataset"] == {
        "houses": 3,
        "sdss17": 3,
    }
    assert payload["settings"]["preprocess.contextual_scale.feature_count"] == "calibrated"
    assert payload["settings"]["preprocess.contextual_scale.feature_count_by_dataset"] == {
        "houses": 1,
        "sdss17": 1,
    }
    assert payload["settings"]["foundation_model.test_eval_interval_epochs"] == 0


def test_heavy_tail_config_crosses_fit_side_contamination_with_mitigation() -> None:
    payload = _load("foundation_models_distribution_shift_heavy_tail.yaml")
    contamination = payload["axes"]["heavyTailContamination"]

    assert [entry["value"] for entry in contamination] == ["original", "fit_side_contaminated"]
    assert payload["settings"]["preprocess.steps"] == [
        "heavy_tail_contamination",
        "target_transform",
        "clamp_train_pool",
    ]
    assert payload["settings"]["preprocess.heavy_tail_contamination.selection_by_dataset"] == {
        "healthcare_insurance_expenses": {"rate": 0.05, "magnitude_iqr": 10},
        "houses": {"rate": 0.01, "magnitude_iqr": 10},
        "miami_housing": {"rate": 0.05, "magnitude_iqr": 10},
    }
    assert payload["settings"]["foundation_model.test_context_comparison.enabled"] is False
    assert payload["settings"]["foundation_model.test_eval_interval_epochs"] == 0
    assert payload["settings"]["foundation_model.epochs"] == 50
    assert payload["settings"]["foundation_model.early_stopping"] is False


def test_representation_configs_change_only_finetune_update_states() -> None:
    feature_order = _load("foundation_models_distribution_shift_feature_order.yaml")
    numeric_scale = _load("foundation_models_distribution_shift_numeric_scale.yaml")

    assert [entry["value"] for entry in feature_order["axes"]["featureOrderUpdateState"]] == [
        "canonical",
        "fixed_permutation",
    ]
    assert [entry["value"] for entry in numeric_scale["axes"]["numericScaleUpdateState"]] == [
        "raw",
        "standardized",
    ]
    assert feature_order["settings"]["foundation_model.test_eval_interval_epochs"] == 0
    assert numeric_scale["settings"]["foundation_model.test_eval_interval_epochs"] == 0






def test_calibration_configs_are_validation_only_tabpfn_runs() -> None:
    for name in (
        "foundation_models_distribution_shift_contextual_scale_calibration.yaml",
        "foundation_models_distribution_shift_contextual_scale_feature_count_calibration.yaml",
        "foundation_models_distribution_shift_contextual_scale_coil_screen.yaml",
        "foundation_models_distribution_shift_heavy_tail_calibration.yaml",
    ):
        payload = _load(name)
        assert [model["name"] for model in payload["models"]] == ["tabpfnv25"]
        assert payload["settings"]["foundation_model.baseline_only"] is True
        assert payload["settings"]["foundation_model.evaluation_splits"] == ["val"]


def test_contextual_feature_count_followup_reuses_reference_and_runs_staged_feature_counts() -> None:
    payload = _load("foundation_models_distribution_shift_contextual_scale_feature_count_calibration.yaml")

    assert [entry["value"] for entry in payload["axes"]["contextualFeatureCount"]] == [
        "3",
        "5",
    ]
    assert [entry["value"] for entry in payload["axes"]["contextualScaleSpread"]] == [
        "3",
        "10",
        "30",
        "100",
    ]
    assert payload["settings"]["preprocess.contextual_scale.condition"] == "scale_sx_no_unit_column"


def test_contextual_coil_screen_uses_aggressive_validation_only_pair() -> None:
    payload = _load("foundation_models_distribution_shift_contextual_scale_coil_screen.yaml")

    assert [entry["value"] for entry in payload["axes"]["contextualFeatureCount"]] == ["8"]
    assert [entry["value"] for entry in payload["axes"]["contextualScaleSpread"]] == [
        "1",
        "100",
    ]
    assert payload["split"]["seeds"] == [0, 1, 2]
    assert payload["settings"]["preprocess.contextual_scale.condition"] == "scale_sx_no_unit_column"






def test_majority_downsampling_followup_retains_minority_and_has_two_conditions() -> None:
    payload = _load("foundation_models_distribution_shift_class_prior_majority_downsampling.yaml")
    override = payload["settings"]
    conditions = payload["axes"]["classPriorMajorityRate"]

    assert [entry["value"] for entry in conditions] == ["original", "0.50"]
    assert [entry["settings"]["preprocess.rebalance_train.target_focal_rate"] for entry in conditions] == [
        "original",
        0.5,
    ]
    assert override["preprocess.rebalance_train.focal_label_policy"] == "original_fit_majority"
    assert override["preprocess.rebalance_train.budget_policy"] == "retain_original_minority"
    assert "target_rates" not in str(override)




def test_context_only_downsampling_finetunes_full_data_and_pairs_icl_contexts() -> None:
    payload = _load("foundation_models_distribution_shift_class_prior_context_only_downsampling.yaml")
    override = payload["settings"]
    conditions = payload["axes"]["testContextComparison"]

    assert [entry["value"] for entry in conditions] == ["full_vs_balanced_icl"]
    assert override["preprocess.steps"] == []
    assert override["foundation_model.test_context_comparison.enabled"] is True
    assert override["foundation_model.test_context_comparison.mode"] == "original_majority_downsampling"
    assert override["foundation_model.test_context_comparison.target_focal_rate"] == 0.5
    assert "preprocess.rebalance_train" not in str(override)




def test_final_rebalancing_scopes_and_folds():
    for name, count in [('majority_downsampling',162), ('context_only_downsampling',81)]:
        recipe = yaml.safe_load((CONFIG_ROOT / f'distribution_shift_class_prior_{name}.yaml').read_text())
        runs = expand_runs(recipe, repo_root=ROOT)
        assert len(runs) == count
        assert {r['dataset'] for r in runs} == {'Amazon_employee_access','bank-marketing','blood-transfusion-service-center','credit-g','jm1'}
