# predict_fl.py
from pathlib import Path
from fd_fl_common import setup_offline_tabpfn, run_experiment

setup_offline_tabpfn(Path(r"D:\tabpfn_project\offline_tabpfn_2_5"))

config = {
    "target": "FL",

    # 🔥 关键：FL不要用log1p
    "transform": "identity",

    # 🔥 加交互特征（重点）
    "features": [
    "底角θ1", "底角θ2", "Re(雷诺数)", "R(曲率半径）",
    "z/H", "流速(m/s)", "r(粒径)", "X",
    "angle_sum", "angle_diff",
    "Re_R", "Re_r", "Re_X",

    ],

    "data_root": r"D:\tabpfn_project\data\triangle_groups_fixed_v2_focus\combined",
    "n_splits": 5,

    # 🔥 FL给更多时间
    "max_time": 60,
}

run_experiment(config)