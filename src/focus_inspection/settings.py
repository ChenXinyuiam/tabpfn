"""Paths and fixed feature sets for focus classification."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data" / "rawdata" / "2026-09-04"
DEFAULT_LABEL_FILE = DEFAULT_RAW_DIR / "三角形聚焦汇总.xlsx"
DEFAULT_BOUNDARY_FEATURES = (
    ROOT / "results" / "focus_inspection" / "2026-09-04" / "focus_boundary_features" / "features.csv"
)
DEFAULT_OUT_DIR = ROOT / "results" / "focus_inspection" / "2026-09-04" / "focus_tabpfn_classification"
DEFAULT_MODEL_PATH = ROOT / "models" / "models" / "tabpfn-v2-classifier.ckpt"

GEOMETRY_FEATURES = [
    "bc",
    "ab",
    "ac",
    "h",
    "angle_a",
    "angle_b",
    "angle_c",
    "perimeter",
    "area",
]
CONTEXT_FEATURES = ["re", "R", *GEOMETRY_FEATURES, "orientation_reverse"]
BOUNDARY_FEATURE = "boundary_distance_abs"
BOUNDARY_MIDPOINT_FEATURE = "boundary_midpoint_x"
MODEL_FEATURE_SETS = {
    "tabpfn_boundary_only": [BOUNDARY_FEATURE],
    "tabpfn_context_only": CONTEXT_FEATURES,
    "tabpfn_context_plus_boundary": [*CONTEXT_FEATURES, BOUNDARY_FEATURE],
}
MAIN_MODEL_NAME = "tabpfn_context_plus_boundary"
