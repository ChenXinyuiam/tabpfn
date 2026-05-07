"""数据管线包。

提供：
- Pipeline / PipelineStage : 轻量级管线编排抽象
- main_pipeline            : 预定义的 6 阶段主管线实例
- 各阶段子模块（clean, scale, calculate, publish, generate 等）
"""

from src.data.pipeline import Pipeline, PipelineStage
from src.data.stages import main_pipeline, get_pipeline

__all__ = [
    "Pipeline",
    "PipelineStage",
    "main_pipeline",
    "get_pipeline",
]
