"""数据管线包。

提供：
- Pipeline / PipelineStage : 轻量级管线编排抽象
- main_pipeline            : 预定义的 6 阶段主管线实例
- 各阶段子模块（clean, scale, calculate, publish, generate 等）
"""

__all__ = [
    "Pipeline",
    "PipelineStage",
    "main_pipeline",
    "get_pipeline",
]


def __getattr__(name):
    if name in {"Pipeline", "PipelineStage"}:
        from src.data.pipeline import Pipeline, PipelineStage
        return {"Pipeline": Pipeline, "PipelineStage": PipelineStage}[name]
    if name in {"main_pipeline", "get_pipeline"}:
        from src.data.stages import get_pipeline, main_pipeline
        return {"main_pipeline": main_pipeline, "get_pipeline": get_pipeline}[name]
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
