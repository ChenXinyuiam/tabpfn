"""Central configuration for paths, columns, features, and experiments."""

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "models" / "offline_tabpfn_2_5"

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "rawdata"
INTERIM_DIR = DATA_DIR / "interim"
CLEANED_DIR = INTERIM_DIR / "cleaned"  # 原data下文件夹,已经归档到interim
SCALED_DIR = INTERIM_DIR / "scaled"    # 原data下文件夹,已经归档到interim
PROCESSED_DIR = INTERIM_DIR / "features" # 原data下文件夹,已经归档到interim
PUBLISHED_PROCESSED_ROOT = DATA_DIR / "processed"

RESULTS_DIR = ROOT_DIR / "results"


@dataclass(frozen=True)
class GeometryColumns:
    base: str = "三角形底边"  # BC
    x: str = "X"
    z: str = "Z"
    re: str = "雷诺数"   # re
    perimeter: str = "周长" # perimeter
    area: str = "面积" # area
    height: str = "高" # h
    angle_a: str = "角度1" # angle_a
    angle_b: str = "角度2" # angle_b
    angle_c: str = "角度3" # angle_c
    centroid_x: str = "重心坐标X" # ???? Wrong?
    centroid_y: str = "重心坐标Y" # ???? Wrong?
    d_base: str = "点到底边距离" #水平距离
    d_left: str = "点到左边距离" #水平距离
    d_right: str = "点到右边距离" #水平距离
    d_centroid: str = "点到重心距离"
    left_side: str = "左边长度" # AB
    right_side: str = "右边长度" # AC


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
