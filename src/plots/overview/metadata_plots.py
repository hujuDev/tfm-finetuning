"""Categorical and continuous metadata figures for the Section 1 overview.

Example:
    ``plot_metadata_scatters(frame, features, output_dir, style)``
"""

from __future__ import annotations

FEATURE_LABELS = {
    "baseTrainRows": "Training rows",
    "nFeaturesTotal": "Total features",
    "trainingTableCells": "Training table cells",
    "rowToFeatureRatio": "Row-to-feature ratio",
    "numericFeatureShare": "Numeric-feature share",
    "numHighCardinalityCategoricalFeatures": "High-cardinality categorical features",
    "maxCategoricalCardinality": "Maximum categorical cardinality",
    "missingValueShare": "Missing-value share",
    "labelEntropyNormalized": "Normalized class entropy",
    "targetP99OverMedian": "Target P99 / median",
    "targetRobustTailIndex": "Robust target-tail index",
    "univariateSignalTop5Share": "Top-five univariate signal share",
    "numericEffectiveRankRatio": "Normalized effective rank",
    "numericMeanAbsoluteSpearman": "Mean absolute numeric rank correlation",
    "numericTailWeightMedian": "Median numeric tail weight",
    "categoricalRareValueShare": "Rare category values",
    "missingnessPatternEntropyNormalized": "Missingness-pattern entropy",
    "duplicateRowShare": "Duplicate-row share",
    "schemaTypeEntropyNormalized": "Numeric/categorical feature balance",
    "localClassDisagreementRate": "Nearest-neighbour target variation",
    "localTargetVariationIqr": "Nearest-neighbour target variation",
    "localTargetVariationByTask": "Nearest-neighbour target variation",
    "localTargetComplexity": "Nearest-neighbour target variation",
    "marginalShapeHeterogeneity": "Variation in feature distributions",
    "sparseRegionOutlierMass": "Isolated-row tail",
    "categoricalCooccurrenceSparsity": "Rare category combinations",
}
AXIS_FEATURE_LABELS = {
    **FEATURE_LABELS,
    "numHighCardinalityCategoricalFeatures": "High-cardinality\ncategorical features",
    "maxCategoricalCardinality": "Maximum categorical\ncardinality",
    "univariateSignalTop5Share": "Top-five univariate\nsignal share",
    "numericEffectiveRankRatio": "Normalized effective\nrank",
    "numericMeanAbsoluteSpearman": "Mean absolute numeric\nrank correlation",
    "numericTailWeightMedian": "Median numeric\ntail weight",
    "categoricalRareValueShare": "Rare category\nvalues",
    "missingnessPatternEntropyNormalized": "Missingness-pattern\nentropy",
    "duplicateRowShare": "Duplicate-row share",
    "schemaTypeEntropyNormalized": "Numeric/categorical\nfeature balance",
    "localClassDisagreementRate": "Nearest-neighbour\ntarget variation",
    "localTargetVariationIqr": "Nearest-neighbour\ntarget variation",
    "localTargetVariationByTask": "Nearest-neighbour\ntarget variation",
    "localTargetComplexity": "Nearest-neighbour\ntarget variation",
    "marginalShapeHeterogeneity": "Variation in feature\ndistributions",
    "sparseRegionOutlierMass": "Isolated-row\ntail",
    "categoricalCooccurrenceSparsity": "Rare category\ncombinations",
}
SOURCE_LABELS = {"tabarena": "TabArena panel", "openml_addendum": "OpenML addendum"}


__all__ = ["FEATURE_LABELS"]
