"""
轻量级数据处理管线编排层。

提供 PipelineStage（单个阶段）和 Pipeline（有序阶段集合）两个类，
封装了 batch 选取 + 文件遍历 + 逐个转换的通用模式。

统一入口：
    python -m src.data.pipeline run [--batch X] [--from clean] [--until publish]
    python -m src.data.pipeline run --stage clean [--batch X]
    python -m src.data.pipeline show
    python -m src.data.pipeline list
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, Any

from utils.paths import ensure_dir, iter_excel_files, select_batch_dir


class PipelineStage:
    """单个数据处理阶段。

    子类只需覆写 transform_file(src, dst)（核心处理逻辑），
    以及有需要时覆写 iter_inputs(batch_dir)（非标准源文件选取）。

    属性
    ----
    name : str            阶段名，如 ``"clean"``。
    dependencies : list[str]
                           依赖的阶段名列表，构造 Pipeline 时校验。
    input_dir_attr : str  ``src.config`` 中输入目录的属性名。
    output_dir_attr : str ``src.config`` 中输出目录的属性名。
    file_filter : Callable | None
                           可选过滤器，对 iter_excel_files 的结果做二次过滤。
    """

    name: str = ""
    dependencies: list[str] = []
    input_dir_attr: str = ""
    output_dir_attr: str = ""
    file_filter: Callable[[Path], bool] | None = None

    # ── 子类覆写点 ──────────────────────────────────────────────────────────

    def transform_file(self, src: Path, dst: Path) -> dict[str, Any] | None:
        """处理单个文件。子类必须覆写。"""
        raise NotImplementedError(f"{self.name}: transform_file() must be overridden")

    def iter_inputs(self, batch_dir: Path) -> list[Path]:
        """发现源文件。默认使用 iter_excel_files；子类可覆写。"""
        files = iter_excel_files(batch_dir)
        if self.file_filter:
            files = [f for f in files if self.file_filter(f)]
        return files

    # ── 运行时 ───────────────────────────────────────────────────────────────

    def _resolve_dir(self, attr: str) -> Path:
        from src import config
        path = getattr(config, attr, None)
        if path is None:
            raise AttributeError(f"src.config has no attribute '{attr}' (stage={self.name})")
        return Path(path)

    def run(self, batch: str | None = None) -> list[dict[str, Any]]:
        """运行当前阶段，返回每个文件的处理结果 dict 列表。"""
        input_dir = self._resolve_dir(self.input_dir_attr)
        output_dir = self._resolve_dir(self.output_dir_attr)

        batch_dir = select_batch_dir(input_dir, batch)
        batch_name = batch_dir.name if batch_dir != input_dir else ""

        source_files = self.iter_inputs(batch_dir)
        if not source_files:
            raise FileNotFoundError(
                f"[{self.name}] 无匹配文件，目录: {batch_dir}"
            )

        dst_root = output_dir / batch_name if batch_name else output_dir
        ensure_dir(dst_root)

        print(f"[{self.name}] batch={batch_name or '(root)'}  "
              f"{len(source_files)} 个文件 -> {dst_root}/")

        results: list[dict[str, Any]] = []
        for src in source_files:
            dst = dst_root / src.name
            print(f"  {src.name} ... ", end="", flush=True)
            try:
                stats = self.transform_file(src, dst)
            except Exception:
                print("FAILED")
                raise
            print("done")
            if stats:
                stats.setdefault("file", str(dst.name))
                results.append(stats)

        return results


class Pipeline:
    """有序的阶段集合，构造时校验依赖 DAG。"""

    def __init__(self, stages: list[PipelineStage]) -> None:
        names = [s.name for s in stages]
        if len(names) != len(set(names)):
            seen = set()
            dupes = [n for n in names if n in seen or seen.add(n)]
            raise ValueError(f"重复的阶段名: {dupes}")

        self.stages: dict[str, PipelineStage] = {s.name: s for s in stages}
        self._order: list[str] = names

        # 校验依赖 DAG
        for s in stages:
            for dep in s.dependencies:
                if dep not in self.stages:
                    raise ValueError(
                        f"阶段 '{s.name}' 依赖 '{dep}'，但 '{dep}' 未注册"
                    )

    # ── 查询 ─────────────────────────────────────────────────────────────────

    def list_stages(self) -> list[dict[str, Any]]:
        """返回每个阶段的摘要信息。"""
        return [
            {
                "name": s.name,
                "dependencies": s.dependencies,
                "input": s.input_dir_attr,
                "output": s.output_dir_attr,
            }
            for s in self._stage_list()
        ]

    def show(self) -> str:
        """文本 DAG 图。"""
        lines = ["管线依赖 DAG:", "-" * 40]
        for s in self._stage_list():
            deps = " -> ".join(s.dependencies) if s.dependencies else "(无依赖)"
            lines.append(f"  [{s.name}]  依赖: {deps}")
        return "\n".join(lines)

    # ── 运行 ─────────────────────────────────────────────────────────────────

    def run(
        self,
        batch: str | None = None,
        *,
        from_stage: str | None = None,
        until: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """运行管线。

        参数
        ----
        batch : 指定 batch 名称，默认取最新。
        from_stage : 从此阶段开始运行（含），默认从头。
        until : 运行到此阶段结束（含），默认到末尾。
        """
        target = self._resolve_range(from_stage, until)
        results: dict[str, list[dict[str, Any]]] = {}
        for name in target:
            results[name] = self.stages[name].run(batch=batch)
        return results

    def run_stage(self, name: str, batch: str | None = None) -> list[dict[str, Any]]:
        """运行单个阶段。"""
        if name not in self.stages:
            raise KeyError(f"未知阶段 '{name}'。可用: {list(self.stages)}")
        return self.stages[name].run(batch=batch)

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _stage_list(self) -> list[PipelineStage]:
        return [self.stages[n] for n in self._order]

    def _resolve_range(self, from_stage: str | None, until: str | None) -> list[str]:
        order = self._order
        start = order.index(from_stage) if from_stage else 0
        end = order.index(until) + 1 if until else len(order)
        if start >= end:
            raise ValueError(f"无效范围: from={from_stage!r} 在 until={until!r} 之后")
        return order[start:end]


# ── 统一 CLI ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TabPFN 数据处理管线",
        prog="python -m src.data.pipeline",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # run
    run_parser = sub.add_parser("run", help="运行管线阶段")
    run_parser.add_argument("--batch", default=None, help="Batch 名称，默认最新")
    run_parser.add_argument("--stage", default=None, help="仅运行指定阶段")
    run_parser.add_argument("--from", dest="from_stage", default=None, help="从此阶段开始")
    run_parser.add_argument("--until", default=None, help="运行到此阶段为止")

    # show
    sub.add_parser("show", help="显示管线依赖 DAG")

    # list
    sub.add_parser("list", help="列出所有阶段")

    return parser


def main(argv: list[str] | None = None) -> None:
    from src.data.stages import main_pipeline

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.stage:
            main_pipeline.run_stage(args.stage, batch=args.batch)
        elif args.from_stage or args.until:
            main_pipeline.run(
                batch=args.batch,
                from_stage=args.from_stage,
                until=args.until,
            )
        else:
            main_pipeline.run(batch=args.batch)
    elif args.command == "show":
        print(main_pipeline.show())
    elif args.command == "list":
        for info in main_pipeline.list_stages():
            deps = " <- ".join(info["dependencies"]) if info["dependencies"] else "(root)"
            print(f"  {info['name']:20s}  [{deps}]")
            print(f"    {info['input']}  ->  {info['output']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
