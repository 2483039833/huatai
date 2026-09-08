# -*- coding: utf-8 -*-
"""
模型评测 (Model Evaluation)
============================
对应最终产出④「模型评测报告」。

以带标签数据（train/test tagged）为金标准，评测：
  - 事件类型识别准确率（event_name 精确匹配）
  - 情感 / 影响方向准确率（event_emotion）
  - 主体抽取召回率（secu_abbr 是否命中）
  - 事件时间抽取准确率（可选）
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from .extractor import EventExtractor
from .paths import cleaned_dir, output_dir


def _gold_subjects(row: pd.Series) -> List[str]:
    abbr = str(row.get("secu_abbr", "")).strip()
    return [abbr] if abbr and abbr != "nan" else []


def evaluate(
    extractor: EventExtractor,
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> Dict:
    """
    对带标签数据做评测。

    Parameters
    ----------
    extractor : EventExtractor
    df        : 带标签数据（含 event_name / event_emotion / secu_abbr）
    limit     : 只评测前 N 条（调试用），None 表示全部
    """
    if limit:
        df = df.head(limit)

    n = len(df)
    correct_event = 0
    correct_emotion = 0
    subject_hit = 0
    time_total = time_correct = 0

    records = []

    for _, row in df.iterrows():
        article = {
            "article_file_id": str(row.get("article_file_id", "")),
            "article_title": str(row.get("article_title", "")),
            "content": str(row.get("content", "")),
            "article_source": str(row.get("article_source", "")),
            "article_publish_time": str(row.get("article_publish_time", "")),
        }
        pred = extractor.extract(article)

        gold_event = str(row.get("event_name", "")).strip()
        gold_emotion = str(row.get("event_emotion", "")).strip()
        gold_subjects = _gold_subjects(row)

        # 事件类型
        if pred["event_name"] and pred["event_name"] == gold_event:
            correct_event += 1

        # 情感
        if pred["event_emotion"] == gold_emotion:
            correct_emotion += 1

        # 主体召回
        pred_subjects = {s["secu_abbr"] for s in pred["subjects"]}
        if gold_subjects and (pred_subjects & set(gold_subjects)):
            subject_hit += 1

        # 时间（金标准用发布日期近似，仅统计有事件时间的）
        if pred.get("event_time"):
            time_total += 1
            pub = str(row.get("article_publish_time", ""))[:10]
            if pred["event_time"] == pub:
                time_correct += 1

        records.append({
            "article_file_id": pred["article_file_id"],
            "title": pred["article_title"][:40],
            "gold_event": gold_event,
            "pred_event": pred["event_name"],
            "gold_emotion": gold_emotion,
            "pred_emotion": pred["event_emotion"],
            "event_correct": pred["event_name"] == gold_event,
            "emotion_correct": pred["event_emotion"] == gold_emotion,
        })

    metrics = {
        "n": n,
        "event_accuracy": round(correct_event / n, 4) if n else 0.0,
        "emotion_accuracy": round(correct_emotion / n, 4) if n else 0.0,
        "subject_recall": round(subject_hit / n, 4) if n else 0.0,
        "time_accuracy": round(time_correct / time_total, 4) if time_total else None,
        "correct_event": correct_event,
        "correct_emotion": correct_emotion,
    }

    return {"metrics": metrics, "records": records}


def print_report(result: Dict) -> None:
    m = result["metrics"]
    print("\n" + "=" * 60)
    print("事件识别与要素抽取 评测报告")
    print("=" * 60)
    print(f"样本数          : {m['n']}")
    print(f"事件类型准确率   : {m['event_accuracy']:.2%}  ({m['correct_event']}/{m['n']})")
    print(f"情感准确率       : {m['emotion_accuracy']:.2%}  ({m['correct_emotion']}/{m['n']})")
    print(f"主体抽取召回率   : {m['subject_recall']:.2%}")
    if m["time_accuracy"] is not None:
        print(f"时间抽取准确率   : {m['time_accuracy']:.2%}")
    print("=" * 60)


def save_evaluation(result: Dict, out: Optional[str] = None) -> Dict[str, str]:
    """把评测结果落盘为 evaluation_report.json + evaluation_errors.csv。"""
    out = out or output_dir()
    report_path = os.path.join(out, "evaluation_report.json")
    errors_path = os.path.join(out, "evaluation_errors.csv")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 错误样本明细（事件类型或情感任一判错）
    err_df = pd.DataFrame(result.get("records", []))
    if not err_df.empty:
        err_df = err_df[(~err_df["event_correct"]) | (~err_df["emotion_correct"])]
    err_df.to_csv(errors_path, index=False, encoding="utf-8-sig")

    print(f"\n评测结果已保存到 {out}/")
    print(f"  - {report_path}")
    print(f"  - {errors_path}")
    return {"report": report_path, "errors": errors_path}


def run_evaluation(
    extractor: Optional[EventExtractor] = None,
    limit: Optional[int] = 200,
    save: bool = True,
) -> Dict:
    """加载带标签测试集做评测，返回结果 dict。"""
    extractor = extractor or EventExtractor()
    path = os.path.join(cleaned_dir(), "test_tagged_cleaned.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}")

    df = pd.read_excel(path)
    print(f"评测数据: {len(df)} 条（{path}）")
    result = evaluate(extractor, df, limit=limit)
    print_report(result)

    if save:
        save_evaluation(result)
    return result
