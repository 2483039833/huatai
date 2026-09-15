# -*- coding: utf-8 -*-
"""
标注数据处理 (Annotation Processing)
=====================================
对应主要工作①「数据清洗与标注处理」。

把带标签数据（news_with_tags_*.xlsx / 清洗后 xlsx）转换成：
  1. LLM 微调格式（JSONL，Alpaca / Chat 风格）—— 用于 Qwen/DeepSeek/GLM 的 LoRA 微调
  2. NER 格式（BIO）—— 用于 NER + Relation Extraction 路线

输出到 outputs_event/ 目录。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from .paths import cleaned_dir, output_dir
from .prompts import SYSTEM_PROMPT
from .schema import EventSchema

BIO_LABELS = ["B-COMPANY", "I-COMPANY", "B-EVENT", "I-EVENT", "O"]


def load_tagged(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = ["article_title", "content", "event_name", "event_emotion"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"缺少列 {c}")
    return df


def build_gold_json(row: pd.Series, schema: EventSchema) -> Dict:
    """构造金标准结构化 JSON（作为微调的 assistant 输出）。"""
    secu_abbr = str(row.get("secu_abbr", "")).strip()
    trading_code = str(row.get("trading_code", "")).strip()
    industry_name = str(row.get("industry_name", "")).strip()
    industry_code = str(row.get("industry_code", "")).strip()

    return {
        "event_name": str(row.get("event_name", "")).strip(),
        "event_emotion": str(row.get("event_emotion", "")).strip(),
        "subjects": [{"secu_abbr": secu_abbr, "trading_code": trading_code}] if secu_abbr else [],
        "impact_objects": [{"name": secu_abbr, "type": "公司"}] if secu_abbr else [],
        "event_time": "",
        "impact_direction": str(row.get("event_emotion", "")).strip(),
        "industry": {"induname": industry_name, "inducode": industry_code} if industry_name else None,
        "confidence": 1.0,
        "reason": "人工标注",
    }


def build_llm_sft(
    df: pd.DataFrame,
    schema: EventSchema,
    style: str = "chat",
) -> List[Dict]:
    """
    构造 LLM 微调样本。

    style = "chat"   : OpenAI 风格 [{"messages": [...]}]
    style = "alpaca" : {"instruction","input","output"}
    """
    samples = []
    for _, row in df.iterrows():
        user_input = (
            f"新闻标题：{row.get('article_title','')}\n"
            f"发布时间：{row.get('article_publish_time','')}\n"
            f"来源：{row.get('article_source','')}\n"
            f"正文：{row.get('content','')}"
        )
        gold = build_gold_json(row, schema)
        gold_str = json.dumps(gold, ensure_ascii=False)

        if style == "chat":
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": gold_str},
                ]
            })
        else:  # alpaca
            samples.append({
                "instruction": SYSTEM_PROMPT,
                "input": user_input,
                "output": gold_str,
            })
    return samples


def build_ner_bio(df: pd.DataFrame, schema: EventSchema) -> List[Dict]:
    """
    构造 NER 训练样本（BIO 标注）。

    只标注公司名（secu_abbr 出现的位置），事件类型作为句子级标签（分类头）。
    """
    samples = []
    for _, row in df.iterrows():
        text = str(row.get("content", ""))
        secu_abbr = str(row.get("secu_abbr", "")).strip()
        tokens = list(text)  # 字符级
        labels = ["O"] * len(tokens)

        # 标注公司简称出现的位置（字符级 BIO）
        if secu_abbr:
            start = text.find(secu_abbr)
            if start != -1:
                labels[start] = "B-COMPANY"
                for i in range(start + 1, start + len(secu_abbr)):
                    labels[i] = "I-COMPANY"

        samples.append({
            "text": text,
            "tokens": tokens,
            "labels": labels,
            "event_name": str(row.get("event_name", "")),
            "event_emotion": str(row.get("event_emotion", "")),
        })
    return samples


def run(
    tagged_path: str = None,
    style: str = "chat",
    limit: int = None,
) -> Dict[str, str]:
    """一键生成标注数据。"""
    schema = EventSchema()
    if tagged_path is None:
        tagged_path = os.path.join(cleaned_dir(), "train_tagged_cleaned.xlsx")

    df = load_tagged(tagged_path)
    if limit:
        df = df.head(limit)

    out = output_dir()

    # 1. LLM 微调
    sft = build_llm_sft(df, schema, style=style)
    sft_path = os.path.join(out, "llm_sft.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for s in sft:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 2. NER BIO
    ner = build_ner_bio(df, schema)
    ner_path = os.path.join(out, "ner_bio.jsonl")
    with open(ner_path, "w", encoding="utf-8") as f:
        for s in ner:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[标注处理] 输入 {len(df)} 条")
    print(f"  LLM 微调样本 -> {sft_path}")
    print(f"  NER BIO 样本 -> {ner_path}")
    return {"llm_sft": sft_path, "ner_bio": ner_path}


if __name__ == "__main__":
    run()
