# predict_fd.py
from pathlib import Path
from fd_fl_common import setup_offline_tabpfn, run_experiment

# ===== 离线模型路径 =====
setup_offline_tabpfn(Path(r"D:\tabpfn_project\offline_tabpfn_2_5"))

config = {
    "target": "FD",
    "transform": "log1p",
    "features": [
        "底角θ1", "底角θ2", "Re(雷诺数)", "R(曲率半径）",
        "z/H", "流速(m/s)", "r(粒径)", "X"
    ],
    "data_root": r"D:\tabpfn_project\data\triangle_groups_fixed_v2_focus\combined",
    "n_splits": 5,
    "max_time": 30,
}

run_experiment(config)