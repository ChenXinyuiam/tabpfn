"""
具体管线阶段定义。

每个阶段通过覆写 transform_file() 委托给已有模块的核心函数，
不移动/修改任何处理逻辑。
"""

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.pipeline import Pipeline, PipelineStage

# ── 辅助 ─────────────────────────────────────────────────────────────────────

def _is_fl_or_sym(path: Path) -> bool:
    """只选取 *-FL.xlsx 或 *-sym.xlsx 文件。"""
    return path.name.endswith("-FL.xlsx") or path.name.endswith("-sym.xlsx")


# ── 阶段定义 ─────────────────────────────────────────────────────────────────

class CleanStage(PipelineStage):
    """原始数据清洗：列名规范化、删空列/骨架行、插补缺失距离、删坏 FL 行。"""

    name = "clean"
    dependencies: list[str] = []
    input_dir_attr = "RAW_DIR"
    output_dir_attr = "CLEANED_DIR"

    def iter_inputs(self, batch_dir: Path) -> list[Path]:
        from utils.paths import iter_excel_files
        return iter_excel_files(batch_dir)

    def transform_file(self, src: Path, dst: Path) -> dict[str, Any] | None:
        from src.data.clean import clean_file, normalized_output_name
        # clean_file 内部读写，需要修正目标文件名
        actual_dst = dst.parent / normalized_output_name(src.name)
        return clean_file(src, actual_dst)


class ScaleStage(PipelineStage):
    """物理量纲缩放：面积 ×1e12，其他列 ×1e6。"""

    name = "scale"
    dependencies = ["clean"]
    input_dir_attr = "CLEANED_DIR"
    output_dir_attr = "SCALED_DIR"

    def transform_file(self, src: Path, dst: Path) -> None:
        from src.data.Feature_scale import scale_file
        scale_file(src, dst)


class CalculateStage(PipelineStage):
    """几何特征计算：面积、角和、边长（正弦定理）、重心坐标。"""

    name = "calculate"
    dependencies = ["scale"]
    input_dir_attr = "SCALED_DIR"
    output_dir_attr = "PROCESSED_DIR"

    def transform_file(self, src: Path, dst: Path) -> None:
        from src.data.calculate import compute_geometry_features
        df = pd.read_excel(src, engine="openpyxl")
        df = compute_geometry_features(df)
        df.to_excel(dst, index=False, engine="openpyxl")


class PublishStage(PipelineStage):
    """发布处理后数据：FL 文件直接复制，sym 文件 FL 取反。"""

    name = "publish"
    dependencies = ["calculate"]
    input_dir_attr = "PROCESSED_DIR"
    output_dir_attr = "PUBLISHED_PROCESSED_ROOT"
    file_filter = staticmethod(_is_fl_or_sym)

    def transform_file(self, src: Path, dst: Path) -> None:
        if src.name.endswith("-sym.xlsx"):
            df = pd.read_excel(src, engine="openpyxl")
            if "FL" in df.columns:
                df["FL"] = -df["FL"]
            df.to_excel(dst, index=False, engine="openpyxl")
        else:
            shutil.copy2(src, dst)


class GenerateStage(PipelineStage):
    """共享 Re 的稠密网格生成（二维插值）。"""

    name = "generate"
    dependencies = ["publish"]
    input_dir_attr = "PUBLISHED_PROCESSED_ROOT"
    output_dir_attr = "PUBLISHED_PROCESSED_ROOT"

    def iter_inputs(self, batch_dir: Path) -> list[Path]:
        from src.data.generate import is_source_file
        return sorted(
            p for p in batch_dir.glob("*.xlsx")
            if is_source_file(p)
        )

    def transform_file(self, src: Path, dst: Path) -> dict[str, Any] | None:
        from src.data.generate import generate_for_file, input_label
        label = input_label(src)
        gen_df = generate_for_file(src)
        if gen_df is None:
            return None
        out = src.parent / f"{label}_generated.xlsx"
        gen_df.to_excel(out, index=False)
        return {"file": out.name, "rows": len(gen_df)}


class GeneratePerReStage(PipelineStage):
    """逐 Re 独立稠密网格生成（每个 Re 级别独立插值）。"""

    name = "generate_per_re"
    dependencies = ["publish"]
    input_dir_attr = "PUBLISHED_PROCESSED_ROOT"
    output_dir_attr = "PUBLISHED_PROCESSED_ROOT"

    def iter_inputs(self, batch_dir: Path) -> list[Path]:
        from src.data.generate_per_re import is_source_file
        return sorted(
            p for p in batch_dir.glob("*.xlsx")
            if is_source_file(p)
        )

    def transform_file(self, src: Path, dst: Path) -> dict[str, Any] | None:
        from src.data.generate_per_re import generate_per_re, input_label
        label = input_label(src)
        gen_df = generate_per_re(src)
        if gen_df is None:
            return None
        out = src.parent / f"{label}_generated_per_re.xlsx"
        gen_df.to_excel(out, index=False)
        return {"file": out.name, "rows": len(gen_df)}


# ── 主管线实例 ───────────────────────────────────────────────────────────────

main_pipeline = Pipeline([
    CleanStage(),
    ScaleStage(),
    CalculateStage(),
    PublishStage(),
    GenerateStage(),
    GeneratePerReStage(),
])


def get_pipeline() -> Pipeline:
    """返回主管线实例。"""
    return main_pipeline
