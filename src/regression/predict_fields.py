"""Native FD-u / FL prediction on paired workbook positions, without force conversion."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import MODEL_DIR
from src.data.template_schema import read_data_workbook, standardize_template_frame
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

from .fd_from_fl import radius_from_key, workbook_index
from .fd_to_fl import nearest_fd_distance_by_re, select_features, valid_rows
from .fl_at_fd import fit_predict_tabpfn, resolve_device

ROOT = Path(__file__).resolve().parents[2]


def sample_training_rows(data, max_rows, seed):
    if max_rows is None or len(data) <= max_rows:
        return data.reset_index(drop=True)
    groups = list(data.groupby("re", sort=True))
    if max_rows < len(groups):
        raise ValueError("Training cap must retain at least one row per Re")
    rng = np.random.default_rng(seed)
    chosen = []
    quota = max_rows // len(groups)
    for _, group in groups:
        chosen.extend(rng.choice(group.index.to_numpy(), size=min(len(group), quota), replace=False).tolist())
    remaining = data.index[~data.index.isin(chosen)].to_numpy()
    if len(chosen) < max_rows:
        chosen.extend(rng.choice(remaining, size=max_rows - len(chosen), replace=False).tolist())
    return data.loc[chosen].reset_index(drop=True)


def spatial_diagnostics(train, target, features):
    target = target.reset_index(drop=True)
    nearest = nearest_fd_distance_by_re(train, target)
    relation = np.full(len(target), "missing_training_re", dtype=object)
    for re_value, group in target.groupby("re", sort=False):
        source = train[train.re.eq(re_value)]
        if source.empty:
            continue
        points = np.unique(source[["x", "z"]].to_numpy(dtype=float), axis=0)
        try:
            hull = ConvexHull(points)
            coordinates = group[["x", "z"]].to_numpy(dtype=float)
            inside = np.all(coordinates @ hull.equations[:, :-1].T + hull.equations[:, -1] <= 1e-8, axis=1)
            relation[group.index] = np.where(inside, "inside_training_hull", "outside_training_hull")
        except (QhullError, ValueError):
            relation[group.index] = "degenerate_training_hull"
    values = target[features].replace([np.inf, -np.inf], np.nan)
    lower, upper = train[features].min(), train[features].max()
    outside = (values.lt(lower) | values.gt(upper)).any(axis=1).to_numpy()
    missing = values.isna().any(axis=1).to_numpy()
    same_position = np.isfinite(nearest) & (nearest <= 1e-8)
    reasons = []
    for index in range(len(target)):
        flags = []
        if relation[index] != "inside_training_hull":
            flags.append(relation[index])
        if outside[index]:
            flags.append("outside_training_feature_range")
        if missing[index]:
            flags.append("missing_model_features")
        if same_position[index]:
            flags.append("training_position_overlap")
        reasons.append(";".join(flags))
    return pd.DataFrame({"nearest_training_distance_um": nearest, "spatial_relation": relation,
                         "training_position_overlap": same_position, "review_reasons": reasons})


def predict_direction(train_frame, query_frame, *, target, model_path, device, n_estimators, seed,
                      chunk_size, max_train_rows, predictor=fit_predict_tabpfn):
    train_mask = valid_rows(train_frame, target)
    query_mask = valid_rows(query_frame)
    available = train_frame.loc[train_mask].reset_index(drop=True)
    query = query_frame.loc[query_mask].reset_index(drop=True)
    if available.empty or query.empty:
        raise ValueError("No usable native training rows or prediction positions")
    features = select_features(available, query)
    if target in features or "fl" in features or "u" in features:
        raise ValueError("Target leakage in regression features")
    train = sample_training_rows(available, max_train_rows, seed)
    processor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), MinMaxScaler())
    x_train = processor.fit_transform(train[features].replace([np.inf, -np.inf], np.nan))
    x_query = processor.transform(query[features].replace([np.inf, -np.inf], np.nan))
    values = np.asarray(predictor(x_train, train[target].to_numpy(dtype=float), x_query, model_path=model_path,
                                 device=device, n_estimators=n_estimators, random_state=seed, chunk_size=chunk_size), dtype=float)
    if values.shape != (len(query),) or not np.isfinite(values).all():
        raise ValueError("Regression predictions must be finite and align with query rows")
    diagnostics = spatial_diagnostics(train, query, features)
    out = query_frame[["re", "x", "z"]].reset_index(drop=True).copy()
    out["source_row"] = np.arange(len(out)) + 2
    out["prediction"] = np.nan
    out["status"] = "invalid_input"
    out["nearest_training_distance_um"] = np.nan
    out["spatial_relation"] = "invalid_input"
    out["training_position_overlap"] = False
    out["review_reasons"] = "invalid_input"
    positions = np.flatnonzero(query_mask.to_numpy())
    out.loc[positions, "prediction"] = values
    out.loc[positions, "status"] = "predicted"
    for column in diagnostics:
        out.loc[positions, column] = diagnostics[column].to_numpy()
    out["needs_review"] = out.review_reasons.ne("")
    audit = {"target_column": target, "training_input_rows": len(train_frame), "training_valid_rows": len(available),
             "training_used_rows": len(train), "training_invalid_rows": int((~train_mask).sum()),
             "query_rows": len(out), "predicted_rows": len(values), "invalid_query_rows": int((~query_mask).sum()),
             "features": features, "training_rows_by_re": train.groupby("re").size().to_dict(),
             "generated_training_rows": 0, "evaluation": False}
    return out, audit


def run(batch_dir, out_dir, *, model_path, device="prefer-cuda", only=None, allow_unpaired=False,
        max_train_rows=1000, chunk_size=2048, n_estimators=1, seed=42, length_unit="auto"):
    from .prediction_report import write_report

    fl_files = workbook_index(batch_dir / "FL", "fl")
    fd_files = workbook_index(batch_dir / "FD", "fd")
    common = set(fl_files) & set(fd_files)
    selected = sorted(set(only) if only else common)
    if not selected or set(selected) - common:
        raise ValueError(f"Selected keys must have both FD and FL workbooks: {sorted(set(selected) - common)}")
    if not only and not allow_unpaired and set(fl_files) != set(fd_files):
        raise ValueError("Unpaired workbooks exist; select --only or explicitly use --allow-unpaired")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    sources = [*Path(__file__).parent.glob("*.py"), ROOT / "src/data/template_schema.py", ROOT / "src/config.py", ROOT / "src/cli.py"]
    with experiment_run(out_dir, "regression-predict", extra_sources=sources) as directory:
        started = time.perf_counter()
        inventory = []
        for key in sorted(set(fl_files) | set(fd_files)):
            inventory.append({"pair_key": key, "fl_file": fl_files[key].name if key in fl_files else "",
                              "fd_file": fd_files[key].name if key in fd_files else "", "selected": key in selected,
                              "status": "paired" if key in common else "missing_fd" if key in fl_files else "missing_fl"})
        pd.DataFrame(inventory).to_csv(directory / "pair_inventory.csv", index=False, encoding="utf-8-sig")
        protocol = {"batch_dir": str(batch_dir.resolve()), "selected_pairs": selected, "model_sha256": file_sha256(model_path),
                    "n_estimators": n_estimators, "seed": seed, "chunk_size": chunk_size, "max_train_rows": max_train_rows,
                    "length_unit": length_unit, "target_units": "native values; physical meaning and units are unverified",
                    "force_sum_computed": False, "synthetic_training_rows": False, "evaluation": False,
                    "source_hashes": {str(p.resolve()): file_sha256(p) for key in selected for p in [fl_files[key], fd_files[key]]}}
        write_json(directory / "protocol.json", protocol)
        frames, audits = [], []
        device = resolve_device(device)
        for number, key in enumerate(selected, 1):
            print(f"[{number}/{len(selected)}] {key}: load native FD/FL", flush=True)
            fd, fd_transition = standardize_template_frame(read_data_workbook(fd_files[key]), kind="fd", source_length_unit=length_unit, target_column="u")
            fl, fl_transition = standardize_template_frame(read_data_workbook(fl_files[key]), kind="fl", source_length_unit=length_unit)
            for kind, source, query, target, source_path, query_path in [
                ("FD_u", fd, fl, "u", fd_files[key], fl_files[key]),
                ("FL", fl, fd, "fl", fl_files[key], fd_files[key]),
            ]:
                print(f"  {kind}: native training={len(source)}, query={len(query)}", flush=True)
                result, audit = predict_direction(source, query, target=target, model_path=model_path, device=device,
                    n_estimators=n_estimators, seed=seed, chunk_size=chunk_size, max_train_rows=max_train_rows)
                result["pair_key"] = key
                result["radius_label"] = radius_from_key(key)
                result["field"] = kind
                result["training_file"] = source_path.name
                result["query_file"] = query_path.name
                frames.append(result)
                audits.append({"pair_key": key, "field": kind, **audit, "fd_transition": fd_transition, "fl_transition": fl_transition})
                result.to_csv(directory / f"{key}-{kind}-predictions.csv", index=False, encoding="utf-8-sig")
        data = pd.concat(frames, ignore_index=True)
        data.to_csv(directory / "predictions.csv", index=False, encoding="utf-8-sig")
        for path, sha in protocol["source_hashes"].items():
            if file_sha256(Path(path)) != sha:
                raise ValueError("Source data changed during prediction")
        write_json(directory / "prediction_audit.json", {"directions": audits, "device": device,
                   "elapsed_seconds_before_report": time.perf_counter() - started, "evaluation": False})
        write_report(directory, data, audits, inventory)
        return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=ROOT / "data/rawdata/2026-06-20")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/regression_prediction/2026-06-20")
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--allow-unpaired", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-unit", choices=["auto", "um", "m"], default="auto")
    args = parser.parse_args()
    run(args.batch_dir, args.out_dir, model_path=args.model_path, device=args.device, only=args.only,
        allow_unpaired=args.allow_unpaired, max_train_rows=args.max_train_rows, chunk_size=args.chunk_size,
        n_estimators=args.n_estimators, seed=args.seed, length_unit=args.length_unit)


if __name__ == "__main__":
    main()
