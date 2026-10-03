"""Stable model and task display metadata for overview figures."""

MODEL_ORDER = ["tabpfnv2", "tabpfnv25", "tabiclv2", "mitra"]
MODEL_LABELS = {
    "tabpfnv2": "TabPFNv2",
    "tabpfnv25": "TabPFN-2.5",
    "tabiclv2": "TabICLv2",
    "mitra": "Mitra",
}
MODEL_COLORS = {
    "tabpfnv2": "#2f8f83",
    "tabpfnv25": "#4f78cf",
    "tabiclv2": "#d94f4f",
    "mitra": "#e5b82f",
}
TASK_ORDER = ["binary", "multiclass", "regression"]
TASK_LABELS = {"binary": "Binary", "multiclass": "Multiclass", "regression": "Regression"}

CATASTROPHIC_FINAL_ERROR_RATIO = 10.0
CRITICAL_DIFFERENCE_ALPHA = 0.05
