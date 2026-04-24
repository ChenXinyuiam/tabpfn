# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd

# === 修改这里为你的真实根目录 ===
ROOT = r"D:\pmc\tabpfn_project\data\triangle_groups_fixed_v2_focus"

ALLOWED_SHAPES = {"23_125_32", "125_23_32", "40_40_100", "25_90_65", "90_25_65"}
ALLOWED_RADII = {10, 20, 30}
ALLOWED_RE = set(range(6, 61, 6))  # 6,12,...,60

COMBINED_DIRNAME = "combined"
INDEX_OUT = os.path.join(ROOT, "index_focus_combined.csv")


def assign_kfold_indices(n, k=5, shuffle=True, seed=None):
    if n <= 0:
        return []
    base, rem = divmod(n, k)
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    v = []
    for i, sz in enumerate(sizes, start=1):
        v.extend([i] * sz)
    arr = np.array(v[:n])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(arr)
    return arr.tolist()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_focus_dirs(root):
    """模糊匹配“聚焦/不聚焦”两个子目录，返回 dict：{'聚焦':path, '不聚焦':path}"""
    if not os.path.isdir(root):
        raise RuntimeError(f"ROOT 不存在或不是目录：{root}")
    entries = os.listdir(root)
    mapping = {}
    for name in entries:
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        # 去掉不可见空白后做包含匹配
        key = name.strip().lower()
        if "聚焦" in key and "不" not in key and "neg" not in key:
            mapping["聚焦"] = full
        if "不聚焦" in key or "neg" in key or "unfocus" in key:
            mapping["不聚焦"] = full
    return mapping


def find_csvs_in_leaf(leaf_dir):
    """优先 data.csv；否则 *.csv 全部合并使用（排除 index 等）"""
    preferred = os.path.join(leaf_dir, "data.csv")
    if os.path.isfile(preferred):
        return [preferred]
    cands = [p for p in glob.glob(os.path.join(leaf_dir, "*.csv"))
             if os.path.basename(p).lower() not in {"index.csv", "index_focus_combined.csv"}]
    return sorted(cands)


def debug_scan(root, focus_dirs):
    """打印每一层扫描情况，返回 dict: focus -> {(shape,R,Re): [csvs...]}"""
    stats = {
        "shapes_scanned": 0,
        "shapes_accepted": 0,
        "r_dirs_scanned": 0,
        "r_dirs_accepted": 0,
        "re_dirs_scanned": 0,
        "re_dirs_accepted": 0,
        "leaves_with_csv": 0,
        "leaves_without_csv": 0,
        "reasons": {"shape_not_allowed": 0, "r_parse_fail": 0, "r_not_allowed": 0,
                    "re_parse_fail": 0, "re_not_allowed": 0, "no_csv": 0}
    }
    leaf_maps = {}
    for focus, froot in focus_dirs.items():
        print(f"\n[SCAN] focus目录: {focus} -> {froot}")
        leaf_map = {}
        if not os.path.isdir(froot):
            print("  !!! 非目录，跳过")
            leaf_maps[focus] = leaf_map
            continue
        for shape in os.listdir(froot):
            stats["shapes_scanned"] += 1
            s_dir = os.path.join(froot, shape)
            if not os.path.isdir(s_dir):
                continue
            if shape not in ALLOWED_SHAPES:
                stats["reasons"]["shape_not_allowed"] += 1
                continue
            stats["shapes_accepted"] += 1
            print(f"  [shape] {shape}")

            for radius in os.listdir(s_dir):
                stats["r_dirs_scanned"] += 1
                r_dir = os.path.join(s_dir, radius)
                if not os.path.isdir(r_dir):
                    continue
                try:
                    R_val = int(radius)
                except Exception:
                    stats["reasons"]["r_parse_fail"] += 1
                    continue
                if R_val not in ALLOWED_RADII:
                    stats["reasons"]["r_not_allowed"] += 1
                    continue
                stats["r_dirs_accepted"] += 1
                print(f"    [R] {R_val}")

                for re_name in os.listdir(r_dir):
                    stats["re_dirs_scanned"] += 1
                    leaf = os.path.join(r_dir, re_name)
                    if not os.path.isdir(leaf):
                        continue
                    try:
                        Re_val = int(re_name)
                    except Exception:
                        stats["reasons"]["re_parse_fail"] += 1
                        continue
                    if Re_val not in ALLOWED_RE:
                        stats["reasons"]["re_not_allowed"] += 1
                        continue
                    stats["re_dirs_accepted"] += 1
                    csvs = find_csvs_in_leaf(leaf)
                    if csvs:
                        stats["leaves_with_csv"] += 1
                        print(f"      [Re] {Re_val}  -> csv: {', '.join(os.path.basename(p) for p in csvs)}")
                        leaf_map[(shape, R_val, Re_val)] = csvs
                    else:
                        stats["leaves_without_csv"] += 1
                        stats["reasons"]["no_csv"] += 1
                        print(f"      [Re] {Re_val}  -> ⚠️ 无 csv")
        leaf_maps[focus] = leaf_map
    print("\n[SUMMARY] 扫描统计：", stats)
    return leaf_maps


def main():
    # 0) 定位两个焦点目录
    focus_dirs = find_focus_dirs(ROOT)
    print("[ROOT]", ROOT)
    print("[FOCUS DIRS FOUND]", focus_dirs)
    if "聚焦" not in focus_dirs and "不聚焦" not in focus_dirs:
        raise RuntimeError("根目录下未找到“聚焦/不聚焦”文件夹（可能是命名或空白问题）")

    # 1) DEBUG 扫描：把每一步发现的元素打印出来
    leaf_maps = debug_scan(ROOT, focus_dirs)

    # 2) 并集键
    all_keys = set()
    for m in leaf_maps.values():
        all_keys |= set(m.keys())
    if not all_keys:
        raise RuntimeError("未发现任何叶子工况（三层都通过后仍无任何 *.csv）。请查看上面的 [SCAN] 输出信息。")

    # 3) 生成类内折 & 合并折
    combined_root = os.path.join(ROOT, COMBINED_DIRNAME)
    ensure_dir(combined_root)

    index_rows, made_files = [], 0
    seed = 20251112

    for (shape, R_val, Re_val) in sorted(all_keys):
        # 聚焦/不聚焦两侧的csv列表
        f_csvs = leaf_maps.get("聚焦", {}).get((shape, R_val, Re_val), [])
        u_csvs = leaf_maps.get("不聚焦", {}).get((shape, R_val, Re_val), [])

        # 纵向合并
        df_f = pd.concat([pd.read_csv(p) for p in f_csvs], axis=0, ignore_index=True) if f_csvs else pd.DataFrame()
        df_u = pd.concat([pd.read_csv(p) for p in u_csvs], axis=0, ignore_index=True) if u_csvs else pd.DataFrame()
        n_f, n_u = len(df_f), len(df_u)

        if n_f > 0: df_f["_fold"] = assign_kfold_indices(n_f, 5, True, seed)
        if n_u > 0: df_u["_fold"] = assign_kfold_indices(n_u, 5, True, seed + 1)

        # 写类内折（若该类存在，写回该类第一个叶子目录）
        f_leaf = os.path.dirname(f_csvs[0]) if f_csvs else None
        u_leaf = os.path.dirname(u_csvs[0]) if u_csvs else None
        for f in range(1, 6):
            if n_f > 0 and f_leaf:
                out_f = os.path.join(f_leaf, f"focus_fold{f}.csv")
                df_f[df_f["_fold"] == f].drop(columns=["_fold"], errors="ignore").to_csv(out_f, index=False,
                                                                                         encoding="utf-8-sig")
                made_files += 1
            if n_u > 0 and u_leaf:
                out_u = os.path.join(u_leaf, f"unfocus_fold{f}.csv")
                df_u[df_u["_fold"] == f].drop(columns=["_fold"], errors="ignore").to_csv(out_u, index=False,
                                                                                         encoding="utf-8-sig")
                made_files += 1

            parts = []
            if n_f > 0: parts.append(df_f[df_f["_fold"] == f].drop(columns=["_fold"], errors="ignore"))
            if n_u > 0: parts.append(df_u[df_u["_fold"] == f].drop(columns=["_fold"], errors="ignore"))
            merged = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()

            merged_leaf = os.path.join(combined_root, shape, str(R_val), str(Re_val))
            ensure_dir(merged_leaf)
            merged_out = os.path.join(merged_leaf, f"fold{f}.csv")
            merged.to_csv(merged_out, index=False, encoding="utf-8-sig")
            made_files += 1

        index_rows.append({
            "shape": shape, "radius": R_val, "Re": Re_val,
            "rows_focus": int(n_f), "rows_unfocus": int(n_u),
            "focus_csvs": "|".join(f_csvs), "unfocus_csvs": "|".join(u_csvs),
            "combined_dir": os.path.join(combined_root, shape, str(R_val), str(Re_val))
        })

    pd.DataFrame(index_rows).to_csv(INDEX_OUT, index=False, encoding="utf-8-sig")
    print(f"\n✅ DONE | 生成文件数: {made_files}")
    print(f"📄 索引: {INDEX_OUT}")
    print(f"📁 合并折根目录: {combined_root}")


if __name__ == "__main__":
    main()
