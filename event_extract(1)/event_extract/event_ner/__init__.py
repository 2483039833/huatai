# -*- coding: utf-8 -*-
"""
多源事件识别与要素抽取 (Event Identification & Element Extraction)
=================================================================
对应赛题二任务1：多源金融事件识别与事件要素抽取、跨文本补全与标准化。

子模块：
  - schema             : 标准化事件知识库（加载事件体系 JSON）
  - llm_client         : LLM 客户端（Qwen / DeepSeek / GLM，OpenAI 兼容）
  - prompts            : Prompt 模板
  - extractor          : 事件识别 + 要素抽取 + 跨文本补全 + 标准化
  - postprocess        : 输出后处理与标准化
  - evaluate           : 模型评测报告
  - prepare_annotation : 标注数据处理（转 LLM 微调 / NER 格式）
  - run                : 一键运行入口
"""

__version__ = "0.1.0"

from .schema import EventSchema

__all__ = ["EventSchema"]
