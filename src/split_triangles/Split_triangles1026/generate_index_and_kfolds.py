import os
import pandas as pd

ALLOWED_SHAPES = {"23_125_32", "125_23_32", "40_40_100", "25_90_65", "90_25_65"}
ALLOWED_RADII = {10, 20, 30}
ALLOWED_RE = set(range(6, 61, 6))


def assign_kfold_indices(n, k=5, shuffle=True, seed=None):
    '''
    生成fold_k索引, 给n个样本分配k个fold的索引.
    :param n: 样本数量
    :param k: fold数量
    :return: fold索引
    例如: n=10, k=5, 则返回[1,1,2,2,3,3,4,4,5,5].
    '''
    if n <= 0:
        return []
    base = n // k
    rem = n % k
    sizes = [base + (rem if i == 0 else 0) for i in range(k)]
    result = []
    for i, sz in enumerate(sizes, start=1):
        result.extend([i] * sz)
    result = np.array(result[:n])

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(result)

    return result.tolist()


def main():
    groups_root = r"D:\\pmc\\tabpfn_project\\data\\triangle_groups_fixed"
    index_out = os.path.join(groups_root, "index.csv")
    records = []

    for shape in sorted(os.listdir(groups_root)):
        shape_dir = os.path.join(groups_root, shape)
        if not os.path.isdir(shape_dir) or shape not in ALLOWED_SHAPES:
            continue
        for radius in os.listdir(shape_dir):
            try:
                r_val = int(radius)
            except Exception:
                continue
            if r_val not in ALLOWED_RADII:
                continue
            r_dir = os.path.join(shape_dir, radius)
            for re_name in os.listdir(r_dir):
                try:
                    re_val = int(re_name)
                except Exception:
                    continue
                if re_val not in ALLOWED_RE:
                    continue
                leaf = os.path.join(r_dir, re_name)
                data_csv = os.path.join(leaf, "data.csv")
                if not os.path.isfile(data_csv):
                    continue

                df = pd.read_csv(data_csv)
                focus_col = "聚焦判定" if "聚焦判定" in df.columns else None
                label = "聚焦" if focus_col and (df[focus_col] != "不聚焦").any() else "不聚焦"
                n = len(df)

                records.append({
                    "shape": shape, "radius": r_val, "Re": re_val,
                    "csv_path": data_csv, "rows": n, "label": label
                })

                if n > 0:
                    df["_fold"] = assign_kfold_indices(n, k=5)
                    for f in range(1, 6):
                        fold_df = df[df["_fold"] == f].drop(columns=["_fold"])
                        fold_df.to_csv(os.path.join(leaf, f"fold{f}.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame(records).to_csv(index_out, index=False, encoding="utf-8-sig")
    print(f"✅ 索引文件已生成: {index_out}")


if __name__ == "__main__":
    main()
