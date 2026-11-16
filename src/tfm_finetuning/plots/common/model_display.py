"""Stable model and task display metadata for all figures."""

# Labels and colors of every model. TabPFN generations run from red (TabPFNv2) to blue (TabPFN-3.5), with
# TabPFN-3.5-Fast a lighter shade of the TabPFN-3.5 blue; TabICLv2 is green, Mitra yellow and Mitra-v2 a deeper
# golden yellow. The eight paper colors were checked in figure order with the dataviz palette validator (lightness
# band, colour-blind and normal-vision separation of neighbours).
MODEL_LABELS = {
    "tabpfnv2": "TabPFNv2",
    "tabpfnv25": "TabPFN-2.5",
    "tabpfnv3": "TabPFN-3",
    "tabpfnv35": "TabPFN-3.5",
    "tabpfnv35fast": "TabPFN-3.5-Fast",
    "tabiclv2": "TabICLv2",
    "mitra": "Mitra",
    "mitra_v2": "Mitra-v2",
}
MODEL_COLORS = {
    "tabpfnv2": "#cf3f45",
    "tabpfnv25": "#b04aa6",
    "tabpfnv3": "#9b8cf0",
    "tabpfnv35": "#1b4ea6",
    "tabpfnv35fast": "#6aa6ec",
    "tabiclv2": "#3a9a3a",
    "mitra": "#e5a50a",
    "mitra_v2": "#a87b00",
}
# The models of the paper: TabPFN generations first, then the other model families.
PAPER_MODELS = ["tabpfnv2", "tabpfnv25", "tabpfnv3", "tabpfnv35", "tabpfnv35fast", "tabiclv2", "mitra", "mitra_v2"]
# The models of overview_baseline and the transformation and transfer studies.
MODEL_ORDER = ["tabpfnv2", "tabpfnv25", "tabiclv2", "mitra"]
# Weights usable commercially: TabPFNv2 (Prior Labs License 1.1, Apache-2.0 plus attribution), TabICLv2
# (BSD-3-Clause), Mitra and Mitra-v2 (Apache-2.0). The weights of TabPFN-2.5, -3 and -3.5 are released under
# non-commercial licenses. See THIRD_PARTY_NOTICES.md.
PERMISSIVE_MODELS = ("tabpfnv2", "tabiclv2", "mitra", "mitra_v2")
TASK_ORDER = ["binary", "multiclass", "regression"]
TASK_LABELS = {"binary": "Binary", "multiclass": "Multiclass", "regression": "Regression"}

CATASTROPHIC_FINAL_ERROR_RATIO = 10.0
CRITICAL_DIFFERENCE_ALPHA = 0.05
