### 原chaifensanjiao_1.py
### 根据 大三角-小三角 文件结构生成索引文件，并且要判断是否聚焦
import os
import pandas as pd

# 分组结果的主目录
output_dir = r"/data/triangle_groups2"
index_file = os.path.join(output_dir, "index.csv")

records = []

# 遍历大组文件夹
for big_name in os.listdir(output_dir):
    big_folder = os.path.join(output_dir, big_name)
    if not os.path.isdir(big_folder):
        continue

    for file in os.listdir(big_folder):
        if not file.endswith(".csv"):
            continue
        small_group = file.replace(".csv", "")
        csv_path = os.path.join(big_folder, file)

        df = pd.read_csv(csv_path)

        # 判断聚焦与否
        if (df["聚焦判定"] != "不聚焦").any():
            label = "聚焦"
        else:
            label = "不聚焦"

        records.append({
            "big_group": big_name,
            "small_group": small_group,
            "csv_path": csv_path,
            "label": label
        })

# 保存索引文件
index_df = pd.DataFrame(records)
index_df.to_csv(index_file, index=False, encoding="utf-8-sig")

print("✅ 已生成索引文件:", index_file)
