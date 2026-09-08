# -*- coding: utf-8 -*-
"""
一键运行入口
=============
用法：
  cd event_extract

  # 1. 事件识别 + 要素抽取（对带标签测试集，同时出评测报告）
  python -m event_ner.run

  # 2. 对无标签数据批量抽取（输出结构化 JSON）
  python -m event_ner.run --source unlabeled_test --limit 1000

  # 3. 生成标准化事件知识（事件体系导出）
  python -m event_ner.run --export-schema

  # 4. 标注数据处理（转微调格式）
  python -m event_ner.run --prepare-annotation
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .extractor import EventExtractor
from .paths import cleaned_dir, data_dir, output_dir
from .schema import EventSchema


def export_schema(schema: EventSchema) -> str:
    """导出标准化事件知识（最终产出③）。"""
    out = output_dir()
    knowledge = {
        "companies": schema.companies,
        "industries": schema.industries,
        "event_definitions": schema.event_definitions,
        "summary": schema.summary(),
    }
    path = os.path.join(out, "event_knowledge.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)
    print(f"[标准化事件知识] 已导出 {path}")
    print(f"  公司 {schema.summary()['n_companies']} 家 / 行业 {schema.summary()['n_industries']} 个 / 事件类型 {schema.summary()['n_event_types']} 种")
    return path


def extract_dataset(source: str, limit: int = None) -> str:
    """对指定数据集批量抽取，输出结构化 JSON。"""
    mapping = {
        "tagged_test": ("test_tagged_cleaned.xlsx", True),
        "tagged_train": ("train_tagged_cleaned.xlsx", True),
        "unlabeled_test": ("unlabeled_test_cleaned.xlsx", False),
        "unlabeled_train": ("unlabeled_train_cleaned.xlsx", False),
    }
    if source not in mapping:
        raise ValueError(f"source 必须是 {list(mapping.keys())} 之一")

    fname, tagged = mapping[source]
    path = os.path.join(cleaned_dir(), fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}")

    df = pd.read_excel(path)
    if limit:
        df = df.head(limit)

    extractor = EventExtractor()
    articles = df.to_dict("records")
    results = extractor.extract_batch(articles)

    out = output_dir()
    out_path = os.path.join(out, f"extracted_{source}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[事件结构化 JSON] 抽取 {len(results)} 条 -> {out_path}")

    # 若有标签，顺带评测并保存评测报告（产出④）
    if tagged:
        from .evaluate import evaluate, print_report, save_evaluation
        result = evaluate(extractor, df, limit=limit)
        print_report(result)
        save_evaluation(result)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="多源事件识别与要素抽取")
    parser.add_argument("--source", default="tagged_test",
                        choices=["tagged_test", "tagged_train", "unlabeled_test", "unlabeled_train"])
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    parser.add_argument("--export-schema", action="store_true", help="导出标准化事件知识")
    parser.add_argument("--prepare-annotation", action="store_true", help="生成微调标注数据")
    args = parser.parse_args()

    if args.export_schema:
        export_schema(EventSchema())
        return

    if args.prepare_annotation:
        from .prepare_annotation import run as prep
        prep()
        return

    extract_dataset(args.source, limit=args.limit)


if __name__ == "__main__":
    main()
