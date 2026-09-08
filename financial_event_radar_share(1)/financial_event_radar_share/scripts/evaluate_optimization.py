"""Build a reproducible, consolidated evaluation report for the event radar."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "event_radar" / "reports"
DERIVED_DIR = ROOT / "event_radar" / "data" / "derived"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def classification_diagnostics(path: Path) -> dict[str, Any]:
    records = read_jsonl_gz(path)
    actual = [record["reference"]["event_name"] for record in records]
    predicted = [record["event"]["event_name"] for record in records]
    labels = sorted(set(actual) | set(predicted))
    rows = []
    f1_values = []
    weighted_f1_sum = 0.0
    for label in labels:
        tp = sum(left == label and right == label for left, right in zip(actual, predicted))
        fp = sum(left != label and right == label for left, right in zip(actual, predicted))
        fn = sum(left == label and right != label for left, right in zip(actual, predicted))
        support = sum(left == label for left in actual)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        weighted_f1_sum += support * f1
        rows.append({
            "label": label,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    correct = sum(left == right for left, right in zip(actual, predicted))
    errors = Counter((left, right) for left, right in zip(actual, predicted) if left != right)
    confidence = [float(record["event"]["confidence"]) for record in records]
    review_flags = [bool(record["decision_trace"]["review_required"]) for record in records]
    selective = {}
    for threshold in (0.80, 0.88, 0.90, 0.92, 0.94, 0.95):
        selected = [(left, right) for left, right, score in zip(actual, predicted, confidence) if score >= threshold]
        selective[f">={threshold:.2f}"] = {
            "coverage": round(len(selected) / len(records), 4),
            "accuracy": round(sum(left == right for left, right in selected) / len(selected), 4) if selected else 0.0,
        }
    return {
        "records": len(records),
        "accuracy": round(correct / len(records), 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "weighted_f1": round(weighted_f1_sum / len(records), 4),
        "labels": len(labels),
        "review_required": {"count": sum(review_flags), "rate": round(sum(review_flags) / len(records), 4)},
        "selective_quality": selective,
        "worst_supported_classes": sorted((row for row in rows if row["support"] >= 3), key=lambda row: (row["f1"], row["support"], row["label"]))[:10],
        "top_error_pairs": [
            {"actual": actual_label, "predicted": predicted_label, "count": count}
            for (actual_label, predicted_label), count in errors.most_common(10)
        ],
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    event_report = read_json(REPORT_DIR / "event_extraction_baseline_report.json")
    semantic_experiment = read_json(REPORT_DIR / "gensim_semantic_retrieval_experiment.json")
    deduplication = read_json(REPORT_DIR / "deduplication_baseline_report.json")
    credibility = read_json(REPORT_DIR / "credibility_baseline_report.json")
    timeline = read_json(REPORT_DIR / "timeline_baseline_report.json")
    alerting = read_json(REPORT_DIR / "alerting_baseline_report.json")
    company = classification_diagnostics(DERIVED_DIR / "event_predictions_test_company.jsonl.gz")
    industry = classification_diagnostics(DERIVED_DIR / "event_predictions_test_industry.jsonl.gz")
    original = semantic_experiment["provided_test_diagnostic"]["character_nn"]
    selected = semantic_experiment["provided_test_diagnostic"]["hybrid_char_0.1"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation_protocol": {
            "model_selection": "stratified validation subset of the labelled training data; Macro-F1 first, accuracy second",
            "diagnostic_holdout": "labelled holdout data is only used for final diagnosis, not for model/threshold selection",
            "unscored_modules": "credibility, timeline and alert priority have no ground-truth labels in the supplied data; report operational coverage rather than fabricated accuracy",
        },
        "classification": {"company": company, "industry": industry},
        "optimization_comparison": {
            "baseline_character_nn": original,
            "selected_semantic_hybrid": selected,
            "accuracy_gain": round(selected["accuracy"] - original["accuracy"], 4),
            "macro_f1_gain": round(selected["macro_f1"] - original["macro_f1"], 4),
        },
        "deduplication": deduplication["test"],
        "operational_coverage": {
            "event_clusters": credibility["assessed_event_clusters"],
            "multi_source_events": credibility["events_with_multiple_sources"],
            "official_cue_events": credibility["events_with_official_cues"],
            "timelines": timeline["timelines"],
            "alerts": alerting["alerts"],
            "manual_reviews": alerting["manual_reviews"],
            "immediate_pushes": alerting["immediate_pushes"],
        },
        "source_reports": {
            "event": event_report["generated_at"],
            "deduplication": deduplication["generated_at"],
            "credibility": credibility["generated_at"],
            "timeline": timeline["generated_at"],
            "alerting": alerting["generated_at"],
        },
        "next_optimization_priority": [
            "对个股事件中低 F1 的相近类别补充对比学习样本或人工标注示例，重点降低高频混淆对。",
            "使用事件簇级人工审核样本校准可信度评分和告警优先级；当前数据没有相应真值，不能报告其准确率。",
            "为去重的边界相似新闻引入语义重排序，在保持当前高精度的前提下改善 85.79% 的成对召回率。",
        ],
    }

    lines = [
        "# 评测与优化报告", "",
        "## 结论", "",
        f"当前个股事件模型在带标签留出集诊断上取得 Accuracy **{pct(company['accuracy'])}**、Macro-F1 **{pct(company['macro_f1'])}**。相对原字符近邻，Accuracy 提升 **{pct(report['optimization_comparison']['accuracy_gain'])}**，Macro-F1 提升 **{pct(report['optimization_comparison']['macro_f1_gain'])}**，语义融合方案通过训练集内部验证后已接入正式链路。", "",
        "## 评测规范", "",
        "- 模型与阈值只由带标签训练集的分层验证集选择，先比较 Macro-F1，再比较 Accuracy。", "- 带标签留出集仅用于最终诊断；本报告不根据该集合继续调参。", "- 可信度、时间线状态和告警等级缺少真值标签，因此只报告覆盖与产出分布，不虚构准确率或 F1。", "",
        "## 事件识别指标", "",
        "| 任务 | 样本数 | Accuracy | Macro-F1 | Weighted-F1 | 类别数 | 人工复核标记 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| 个股事件 | {company['records']} | {pct(company['accuracy'])} | {pct(company['macro_f1'])} | {pct(company['weighted_f1'])} | {company['labels']} | {company['review_required']['count']}（{pct(company['review_required']['rate'])}） |",
        f"| 行业事件 | {industry['records']} | {pct(industry['accuracy'])} | {pct(industry['macro_f1'])} | {pct(industry['weighted_f1'])} | {industry['labels']} | {industry['review_required']['count']}（{pct(industry['review_required']['rate'])}） |", "",
        "### 个股优化对比", "",
        "| 模型 | Accuracy | Macro-F1 |", "| --- | ---: | ---: |",
        f"| 原同公司字符近邻 | {pct(original['accuracy'])} | {pct(original['macro_f1'])} |",
        f"| 当前语义融合 | {pct(selected['accuracy'])} | {pct(selected['macro_f1'])} |", "",
        "### 个股置信度分层质量", "",
        "| 置信度下限 | 覆盖率 | Accuracy |", "| --- | ---: | ---: |",
    ]
    for threshold, values in company["selective_quality"].items():
        lines.append(f"| {threshold} | {pct(values['coverage'])} | {pct(values['accuracy'])} |")
    lines.extend(["", "## 个股误差分析（支持样本不少于 3）", "", "| 实际类别 | 支持数 | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in company["worst_supported_classes"]:
        lines.append(f"| {row['label']} | {row['support']} | {pct(row['precision'])} | {pct(row['recall'])} | {pct(row['f1'])} |")
    lines.extend(["", "### 高频混淆对", "", "| 实际类别 | 预测类别 | 次数 |", "| --- | --- | ---: |"])
    for row in company["top_error_pairs"]:
        lines.append(f"| {row['actual']} | {row['predicted']} | {row['count']} |")
    dedup = report["deduplication"]
    pair = dedup["pair_metrics"]
    b3 = dedup["b_cubed"]
    lines.extend([
        "", "## 同源事件聚合 / 去重", "",
        f"在 {dedup['records']} 条测试新闻上，成对去重 Precision 为 **{pct(pair['precision'])}**、Recall 为 **{pct(pair['recall'])}**、F1 为 **{pct(pair['f1'])}**；簇级 B³ F1 为 **{pct(b3['f1'])}**。当前策略更偏向查准率，下一轮应只对边界相似样本提升召回。", "",
        "## 端到端产出与可评测边界", "",
        f"- 已形成 {report['operational_coverage']['event_clusters']} 个事件簇和 {report['operational_coverage']['timelines']} 条时间线；其中 {report['operational_coverage']['multi_source_events']} 个事件拥有多源报道，{report['operational_coverage']['official_cue_events']} 个事件含官方线索。", f"- 当前生成 {report['operational_coverage']['alerts']} 条告警，含 {report['operational_coverage']['immediate_pushes']} 条立即推送和 {report['operational_coverage']['manual_reviews']} 条人工复核。", "- 可信度、时间线状态和告警等级没有人工真值，当前数值是可解释业务规则的输出，不能与分类 F1 混为同一种性能指标。", "",
        "## 下一轮优化建议", "",
    ])
    for index, suggestion in enumerate(report["next_optimization_priority"], start=1):
        lines.append(f"{index}. {suggestion}")
    lines.append("")

    (REPORT_DIR / "evaluation_optimization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "evaluation_optimization_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "report": "evaluation_optimization_report.md",
        "company": {key: company[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
        "industry": {key: industry[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
        "deduplication_pair_f1": pair["f1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
