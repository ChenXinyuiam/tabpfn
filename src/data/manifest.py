"""Write manifest.json files that describe every data batch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_DIR,
    DATA_DIR,
    PROCESSED_DIR,
    PUBLISHED_PROCESSED_ROOT,
    RAW_DIR,
    SCALED_DIR,
)
from utils.paths import ensure_dir

STAGES = {
    "raw": RAW_DIR,
    "cleaned": CLEANED_DIR,
    "scaled": SCALED_DIR,
    "features": PROCESSED_DIR,
    "processed": PUBLISHED_PROCESSED_ROOT,
}


def batch_names() -> list[str]:
    names = set()
    for root in STAGES.values():
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name[:4].isdigit():
                names.add(child.name)
    return sorted(names)


def workbook_info(path: Path) -> dict:
    info = {
        "name": path.name,
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "kind": classify_file(path),
    }
    if path.suffix.lower() == ".xlsx":
        try:
            df = pd.read_excel(path, engine="openpyxl", nrows=0)
            info["columns"] = list(df.columns)
        except Exception as exc:
            info["columns_error"] = str(exc)
    return info


def classify_file(path: Path) -> str:
    if path.suffix.lower() == ".png":
        return "figure"
    if path.suffix.lower() == ".zip":
        return "archive"
    if path.suffix.lower() == ".json":
        return "metadata"
    stem = path.stem
    if stem.startswith("_") or path.name == "manifest.json":
        return "metadata"
    if "_generated_per_re" in stem:
        return "augmented_per_re"
    if "_generated" in stem:
        return "augmented"
    if stem.endswith("-FL"):
        return "fl_original"
    if stem.endswith("-sym"):
        return "sym_original"
    if stem.endswith("-反"):
        return "reverse_original"
    if stem.endswith("_cleaned"):
        return "legacy_cleaned"
    return "other"


def stage_files(root: Path, batch: str) -> list[dict]:
    batch_dir = root / batch
    if not batch_dir.exists():
        return []
    return [
        workbook_info(path)
        for path in sorted(batch_dir.rglob("*"))
        if path.is_file() and not path.name.startswith("~$")
    ]


def build_manifest(batch: str) -> dict:
    stages = {stage: stage_files(root, batch) for stage, root in STAGES.items()}
    return {
        "batch": batch,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "roots": {stage: root.as_posix() for stage, root in STAGES.items()},
        "stages": stages,
        "summary": {
            stage: {
                "file_count": len(files),
                "xlsx_count": sum(1 for item in files if item["name"].endswith(".xlsx")),
                "kinds": count_kinds(files),
            }
            for stage, files in stages.items()
        },
    }


def count_kinds(files: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in files:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    return counts


def write_manifest(batch: str) -> Path:
    manifest = build_manifest(batch)
    out_dir = ensure_dir(DATA_DIR / "manifests" / batch)
    out_path = out_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    processed_batch = PUBLISHED_PROCESSED_ROOT / batch
    if processed_batch.exists():
        (processed_batch / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write manifest.json for data batches.")
    parser.add_argument("--batch", default=None, help="Batch name. Defaults to all discovered batches.")
    args = parser.parse_args()

    batches = [args.batch] if args.batch else batch_names()
    if not batches:
        raise SystemExit("No batch directories found.")

    for batch in batches:
        path = write_manifest(batch)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
