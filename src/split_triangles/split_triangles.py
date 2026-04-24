### 原 split_triangles.py(同目录)
### 将原始数据拆分为 大三角-小三角类型的数据
import os
import pandas as pd

# 读取数据
input_csv = r"D:\pmc\tabpfn_project\data_tabpfn\数据更新20251004.csv"
output_dir = r"/data/triangle_groups_test"   ###change this value when rerunning this program

df = pd.read_csv(input_csv)

# 定义分组键
big_group_keys = ["底角θ1", "底角θ2", "顶角θ"]
small_group_keys = ["周长(mm)", "面积(mm^2)", "Re(雷诺数)", "流速(m/s)",
                    "r(粒径)", "R(曲率半径）", "L/H"]

# 创建分组标识
# df["big_group"]中存放的是底角θ1_底角θ2_顶角θ
df["big_group"] = df[big_group_keys].astype(str).agg("_".join, axis=1)
# df["small_group"] 中存放的是 周长(mm)_面积(mm^2)_Re(雷诺数)_流速(m/s)_r(粒径)_R(曲率半径）_L/H
df["small_group"] = df[small_group_keys].astype(str).agg("_".join, axis=1)

# 遍历大组
for big_name, big_df in df.groupby("big_group"):
    big_folder = os.path.join(output_dir, big_name)
    os.makedirs(big_folder, exist_ok=True)

    # 遍历小组
    for small_name, small_df in big_df.groupby("small_group"):
        file_path = os.path.join(big_folder, f"{small_name}.csv")
        small_df.drop(columns=["big_group", "small_group"]).to_csv(file_path, index=False, encoding="utf-8-sig")

print("✅ 分组完成，每个大组是一个文件夹，小组三角数据存成单独 CSV 文件。")
