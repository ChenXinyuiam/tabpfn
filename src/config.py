"""Central configuration for paths, columns, features, and experiments."""

from dataclasses import dataclass
from pathlib import Path

from utils.features import ENGINEERED_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "models" / "offline_tabpfn_2_5"

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "rawdata"
CLEANED_DIR = DATA_DIR / "cleaned"
SCALED_DIR = DATA_DIR / "scaled"
PROCESSED_DIR = DATA_DIR / "processedv2"
PUBLISHED_PROCESSED_ROOT = DATA_DIR / "processed"
PUBLISHED_PROCESSED_DIR = DATA_DIR / "processed" / "2026-04-22"
TRIANGLE_GROUPS_ROOT = DATA_DIR / "triangle_groups_fixed_v2_focus" / "combined"

RESULTS_DIR = ROOT_DIR / "results"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ANALYSIS_DIR = ROOT_DIR / "dataAnalysis"


@dataclass(frozen=True)
class GeometryColumns:
    base: str = "三角形底边"
    x: str = "X"
    z: str = "Z"
    re: str = "雷诺数"
    perimeter: str = "周长"
    area: str = "面积"
    height: str = "高"
    angle_a: str = "角度1"
    angle_b: str = "角度2"
    angle_c: str = "角度3"
    centroid_x: str = "重心坐标X"
    centroid_y: str = "重心坐标Y"
    d_base: str = "点到底边距离"
    d_left: str = "点到左边距离"
    d_right: str = "点到右边距离"
    d_centroid: str = "点到重心距离"
    left_side: str = "左边长度"
    right_side: str = "右边长度"


@dataclass(frozen=True)
class ExperimentConfig:
    target: str
    transform: str
    features: list[str]
    n_splits: int = 5
    max_time: int = 60

    def as_dict(self, **overrides) -> dict:
        data = {
            "target": self.target,
            "transform": self.transform,
            "features": self.features,
            "n_splits": self.n_splits,
            "max_time": self.max_time,
        }
        data.update(overrides)
        return data


GEOMETRY = GeometryColumns()

BASE_COL = GEOMETRY.base
X_COL = GEOMETRY.x
Z_COL = GEOMETRY.z
RE_COL = GEOMETRY.re
PERIMETER_COL = GEOMETRY.perimeter
AREA_COL = GEOMETRY.area
HEIGHT_COL = GEOMETRY.height
ANGLE_A_COL = GEOMETRY.angle_a
ANGLE_B_COL = GEOMETRY.angle_b
ANGLE_C_COL = GEOMETRY.angle_c
D_BASE_COL = GEOMETRY.d_base
D_LEFT_COL = GEOMETRY.d_left
D_RIGHT_COL = GEOMETRY.d_right
LEFT_SIDE_COL = GEOMETRY.left_side
RIGHT_SIDE_COL = GEOMETRY.right_side

SCALE_1E12 = {"面积"}
SCALE_1E6 = {
    "X",
    "Z",
    "周长",
    "重心坐标X",
    "重心坐标Y",
    "点到底边距离",
    "点到左边距离",
    "点到右边距离",
    "点到重心距离",
    "FL",
}

BASE_FEATURES = RAW_FEATURE_COLUMNS.copy()
ENGINEERED_FEATURES = RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

FD_EXPERIMENT = ExperimentConfig(
    target="FD",
    transform="log1p",
    features=ENGINEERED_FEATURES,
    n_splits=5,
    max_time=30,
)

FL_EXPERIMENT = ExperimentConfig(
    target="FL",
    transform="identity",
    features=ENGINEERED_FEATURES,
    n_splits=5,
    max_time=60,
)

FD_CONFIG = FD_EXPERIMENT.as_dict()
FL_CONFIG = FL_EXPERIMENT.as_dict()

N_SPLITS = 5
TABPFN_IGNORE_PRETRAINING_LIMITS = True

FD_TRANSFORM = FD_EXPERIMENT.transform
FL_TRANSFORM = FL_EXPERIMENT.transform
FD_MAX_TIME = FD_EXPERIMENT.max_time
FL_MAX_TIME = FL_EXPERIMENT.max_time
CLASSIFIER_RANDOM_STATE = 42

TARGET_RE_LIST = list(range(6, 66, 6))
DEFAULT_ROUND_DECIMALS = 6
