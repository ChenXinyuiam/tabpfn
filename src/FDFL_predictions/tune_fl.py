# tune_fl.py
from pathlib import Path
import itertools
import pandas as pd

from fd_fl_common import setup_offline_tabpfn, run_experiment

# ===== 离线模型 =====
setup_offline_tabpfn(Path(r"D:\tabpfn_project\offline_tabpfn_2_5"))

# =============================
# 1️⃣ 定义搜索空间
# =============================

TRANSFORMS = ["identity", "log1p", "sqrt"]

FEATURE_SETS = {
    "base": [
        "底角θ1", "底角θ2", "Re(雷诺数)", "R(曲率半径）",
        "z/H", "流速(m/s)", "r(粒径)", "X"
    ],

    "interaction": [
        "底角θ1", "底角θ2", "Re(雷诺数)", "R(曲率半径）",
        "z/H", "流速(m/s)", "r(粒径)", "X",
        "angle_sum", "angle_diff",
        "Re_R", "Re_r",
        "R_zH"
    ],
}

MAX_TIMES = [10, 30, 60]

# =============================
# 2️⃣ 调参主循环
# =============================

results = []

exp_id = 0

for transform, feat_key, max_time in itertools.product(
    TRANSFORMS, FEATURE_SETS.keys(), MAX_TIMES
):
    exp_id += 1

    print("\n" + "=" * 50)
    print(f"EXP {exp_id}")
    print(f"transform={transform}, features={feat_key}, max_time={max_time}")
    print("=" * 50)

    config = {
        "target": "FL",
        "transform": transform,
        "features": FEATURE_SETS[feat_key],
        "data_root": r"D:\tabpfn_project\data\triangle_groups_fixed_v2_focus\combined",
        "n_splits": 5,
        "max_time": max_time,
    }

    try:
        r2 = run_experiment(config)

        results.append({
            "transform": transform,
            "feature_set": feat_key,
            "max_time": max_time,
            "r2": r2
        })

    except Exception as e:
        print(f"❌ failed: {e}")
        results.append({
            "transform": transform,
            "feature_set": feat_key,
            "max_time": max_time,
            "r2": None
        })

# =============================
# 3️⃣ 保存结果
# =============================

df = pd.DataFrame(results)

df = df.sort_values(by="r2", ascending=False)

output_path = Path("tune_fl_results.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n🎯 调参完成，结果已保存:")
print(output_path)
print("\nTop results:")
print(df.head(10))