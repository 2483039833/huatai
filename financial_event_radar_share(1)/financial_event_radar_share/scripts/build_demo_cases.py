"""Create a complete category casebook and a short live-demo walkthrough."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "event_radar" / "data" / "processed"
DERIVED_DIR = ROOT / "event_radar" / "data" / "derived"
DOCS_DIR = ROOT / "event_radar" / "docs"
TAXONOMY_PATH = DATA_DIR / "event_taxonomy.jsonl"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def escape(value: Any) -> str:
    return str(value or "—").replace("|", "\\|").replace("\n", " ").strip()


def title_keyword(title: str) -> str:
    compact = "".join((title or "").split())
    return compact[:18] if compact else "事件类型"


def load_taxonomy() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {"company": {}, "industry": {}}
    for line in TAXONOMY_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result[row["scope"]].setdefault(row["event_name"], row.get("definition") or "")
    return result


def select_event_cases(scope: str, taxonomy: dict[str, str]) -> list[dict[str, Any]]:
    test_records = list(iter_jsonl(DATA_DIR / f"news_labeled_test_{scope}.jsonl.gz"))
    train_records = list(iter_jsonl(DATA_DIR / f"news_labeled_train_{scope}.jsonl.gz"))
    predictions = list(iter_jsonl(DERIVED_DIR / f"event_predictions_test_{scope}.jsonl.gz"))
    raw_by_id = {record["article_id"]: record for record in test_records}
    correct: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        actual = prediction.get("reference", {}).get("event_name")
        predicted = prediction.get("event", {}).get("event_name")
        raw = raw_by_id.get(prediction["article_id"])
        if raw and actual == predicted:
            correct[actual].append({"raw": raw, "prediction": prediction})
    train_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        train_by_label[record.get("event_name")].append(record)

    cases = []
    for index, event_name in enumerate(sorted(taxonomy), start=1):
        candidates = correct.get(event_name, [])
        if candidates:
            selected = max(candidates, key=lambda item: (float(item["prediction"]["event"]["confidence"]), len(item["raw"].get("title") or "")))
            raw = selected["raw"]
            prediction = selected["prediction"]
            event = prediction["event"]
            trace = prediction["decision_trace"]
            subject = next((item.get("name") for item in event.get("subjects", []) if item.get("name")), raw.get("security_name") or raw.get("industry_name") or "—")
            cases.append({
                "case_id": f"{'C' if scope == 'company' else 'I'}-{index:02d}",
                "scope": scope,
                "event_type": event_name,
                "definition": taxonomy[event_name],
                "title": raw.get("title") or "",
                "source": raw.get("source") or "",
                "published_at": raw.get("published_at") or "",
                "subject": subject,
                "direction": event.get("impact_direction"),
                "confidence": event.get("confidence"),
                "review_required": trace.get("review_required", False),
                "article_id": raw["article_id"],
                "origin": "labelled_holdout_correct_prediction",
                "evidence": (event.get("evidence") or [""])[0][:180],
            })
        else:
            fallback = max(train_by_label.get(event_name, []), key=lambda item: len(item.get("title") or ""), default=None)
            if fallback is None:
                raise ValueError(f"No demonstration sample found for {scope}/{event_name}")
            subject = fallback.get("security_name") or fallback.get("industry_name") or "—"
            cases.append({
                "case_id": f"{'C' if scope == 'company' else 'I'}-{index:02d}",
                "scope": scope,
                "event_type": event_name,
                "definition": taxonomy[event_name],
                "title": fallback.get("title") or "",
                "source": fallback.get("source") or "",
                "published_at": fallback.get("published_at") or "",
                "subject": subject,
                "direction": fallback.get("event_emotion") or "—",
                "confidence": None,
                "review_required": None,
                "article_id": fallback["article_id"],
                "origin": "labelled_training_category_coverage",
                "evidence": (fallback.get("content") or "")[:180],
            })
    return cases


def load_cluster_data() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    clusters = {row["cluster_id"]: row for row in iter_jsonl(DERIVED_DIR / "deduplication_clusters_test.jsonl.gz")}
    credibility = {row["cluster_id"]: row for row in iter_jsonl(DERIVED_DIR / "credibility_assessment_test.jsonl.gz")}
    timelines = {row["cluster_id"]: row for row in iter_jsonl(DERIVED_DIR / "event_timelines_test.jsonl.gz")}
    alerts = list(iter_jsonl(DERIVED_DIR / "alert_feed_test.jsonl.gz"))
    return clusters, credibility, timelines, alerts


def compact_alert(alert: dict[str, Any]) -> dict[str, Any]:
    return {
        "alert_id": alert["alert_id"],
        "cluster_id": alert["cluster_id"],
        "title": alert["title"],
        "event_type": alert["event_type"],
        "priority": alert["priority"],
        "action": alert["action"],
        "impact_score": alert["impact_score"],
        "timeline_status": alert["timeline_status"],
        "credibility_level": alert["credibility"]["level"],
        "credibility_score": alert["credibility"]["score"],
        "search_keyword": title_keyword(alert["title"]),
    }


def build_functional_cases() -> dict[str, Any]:
    clusters, credibility, timelines, alerts = load_cluster_data()
    priorities = {}
    for priority in ("P3", "P2", "P1", "P0"):
        candidates = [alert for alert in alerts if alert["priority"] == priority]
        priorities[priority] = compact_alert(max(candidates, key=lambda alert: float(alert["impact_score"])))
    manual_candidates = [alert for alert in alerts if alert["action"] == "manual_review"]
    manual = compact_alert(manual_candidates[0]) if manual_candidates else None
    statuses = {}
    for status in ("confirmed", "corroborated", "reported", "watching", "clarified"):
        candidates = [row for row in timelines.values() if row["current_status"] == status]
        selected = max(candidates, key=lambda row: (row.get("article_count", 0), row.get("source_count", 0)))
        statuses[status] = {
            "cluster_id": selected["cluster_id"],
            "title": selected["canonical_title"],
            "article_count": selected["article_count"],
            "source_count": selected["source_count"],
            "current_status": status,
            "search_keyword": title_keyword(selected["canonical_title"]),
        }
    credibility_levels = {}
    for level in ("high", "medium", "low", "very_low"):
        candidates = [row for row in credibility.values() if row["credibility"]["level"] == level]
        selected = max(candidates, key=lambda row: (row.get("article_count", 0), row["credibility"]["score"]))
        credibility_levels[level] = {
            "cluster_id": selected["cluster_id"],
            "title": selected["canonical_title"],
            "article_count": selected["article_count"],
            "score": selected["credibility"]["score"],
            "reasons": selected["credibility"].get("reasons", []),
            "search_keyword": title_keyword(selected["canonical_title"]),
        }
    largest = max(clusters.values(), key=lambda row: (row["article_count"], len(row["sources"])))
    return {
        "alert_priorities": priorities,
        "manual_review": manual,
        "timeline_statuses": statuses,
        "credibility_levels": credibility_levels,
        "deduplication": {
            "cluster_id": largest["cluster_id"],
            "title": largest["canonical_title"],
            "article_count": largest["article_count"],
            "source_count": len(largest["sources"]),
            "search_keyword": title_keyword(largest["canonical_title"]),
        },
    }


def event_case_table(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["| 案例 | 事件类型 | 标的/行业 | 预测置信度 | 方向 | 标题 | 来源 |", "| --- | --- | --- | ---: | --- | --- | --- |"]
    for case in cases:
        confidence = f"{case['confidence']:.2f}" if case["confidence"] is not None else "补充样例"
        lines.append(f"| {case['case_id']} | {escape(case['event_type'])} | {escape(case['subject'])} | {confidence} | {escape(case['direction'])} | {escape(case['title'])} | {escape(case['source'])} |")
    return lines


def build_markdown(company_cases: list[dict[str, Any]], industry_cases: list[dict[str, Any]], functional: dict[str, Any]) -> tuple[str, str]:
    test_company = sum(case["origin"] == "labelled_holdout_correct_prediction" for case in company_cases)
    test_industry = sum(case["origin"] == "labelled_holdout_correct_prediction" for case in industry_cases)
    casebook = [
        "# 赛题二演示案例库", "",
        "本案例库用于答辩与产品演示，覆盖事件体系中的全部公司事件类别和行业事件类别。优先选取带标签留出集上的正确预测；若该类别在留出集中没有正确样例，则以带标签训练新闻补充，并明确标注为“补充样例”。", "",
        "## 覆盖统计", "",
        f"- 公司事件：{len(company_cases)}/{len(company_cases)} 类，留出集正确预测案例 {test_company} 条，补充训练样例 {len(company_cases) - test_company} 条。", f"- 行业事件：{len(industry_cases)}/{len(industry_cases)} 类，留出集正确预测案例 {test_industry} 条，补充训练样例 {len(industry_cases) - test_industry} 条。", "- 每个案例均保留文章编号、标题、来源、时间、主体、方向和模型置信度，便于追溯。", "",
        "## 公司事件案例", "",
    ]
    casebook.extend(event_case_table(company_cases))
    casebook.extend(["", "## 行业事件案例", ""])
    casebook.extend(event_case_table(industry_cases))
    casebook.extend(["", "## 使用提示", "", "- 案例编号以 `C-` 开头的是公司事件，以 `I-` 开头的是行业事件。", "- “补充样例”用于展示事件类型覆盖，不应被描述为留出集评测结果。", "- 主讲时优先使用下方的演示路线展示系统能力，再按需翻到本案例库回答类别覆盖问题。", ""])

    walkthrough = [
        "# 赛题二演示路线", "",
        "建议演示时长：6–8 分钟。打开 <http://localhost:8080/web/>，依次用筛选器和搜索框定位以下真实案例。", "",
        "## 1. 事件识别与可解释结果（约 1 分钟）", "",
        f"- 从案例库任选 `C-01` 至 `C-{len(company_cases):02d}` 中的公司事件，说明系统输出事件类型、主体、方向、置信度、证据片段和人工复核标记。", f"- 再任选 `I-01` 至 `I-{len(industry_cases):02d}` 中的行业事件，说明行业事件使用独立事件体系约束。", "- 可补充说明：公司事件 Macro-F1 为 80.09%，行业事件 Macro-F1 为 96.82%。", "",
        "## 2. 多源去重与证据链（约 1 分钟）", "",
        f"- 搜索“{escape(functional['deduplication']['search_keyword'])}”。该事件簇 `{functional['deduplication']['cluster_id']}` 聚合了 {functional['deduplication']['article_count']} 篇报道、{functional['deduplication']['source_count']} 个来源。", "- 展开右侧时间线，强调系统把重复报道归并为一个事件，避免重复推送。", "",
        "## 3. 可信度分级（约 1 分钟）", "",
    ]
    for level, case in functional["credibility_levels"].items():
        walkthrough.append(f"- {level}：搜索“{escape(case['search_keyword'])}”，可信度 {case['score']:.2f}，事件簇 `{case['cluster_id']}`。")
    walkthrough.extend(["", "## 4. 时间线状态变化（约 1 分钟）", ""])
    for status, case in functional["timeline_statuses"].items():
        walkthrough.append(f"- {status}：搜索“{escape(case['search_keyword'])}”，{case['article_count']} 篇报道 / {case['source_count']} 个来源，事件簇 `{case['cluster_id']}`。")
    walkthrough.extend(["", "## 5. 分级预警与行动建议（约 1 分钟）", "", "先在左侧优先级下拉框依次选择 P3、P2、P1、P0："])
    for priority, case in functional["alert_priorities"].items():
        walkthrough.append(f"- {priority}：搜索“{escape(case['search_keyword'])}”，影响分 {case['impact_score']:.2f}，建议动作 `{case['action']}`，状态 `{case['timeline_status']}`。")
    if functional["manual_review"]:
        case = functional["manual_review"]
        walkthrough.extend(["", "## 6. 人工复核闭环（约 1 分钟）", "", f"- 搜索“{escape(case['search_keyword'])}”。该事件被标记为人工复核，说明系统不会把低置信度的高影响事件直接自动推送。事件簇 `{case['cluster_id']}`。"])
    walkthrough.extend(["", "## 7. 收束（约 30 秒）", "", "总结系统闭环：事件体系约束保证识别范围，多源聚合降低重复，可信度与时间线解释信息质量，分级预警将事件转化为可执行动作。完整类别案例见 `docs/demo_cases.md`。", ""])
    return "\n".join(casebook), "\n".join(walkthrough)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy()
    company_cases = select_event_cases("company", taxonomy["company"])
    industry_cases = select_event_cases("industry", taxonomy["industry"])
    functional = build_functional_cases()
    casebook, walkthrough = build_markdown(company_cases, industry_cases, functional)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "coverage": {"company_event_types": len(company_cases), "industry_event_types": len(industry_cases)},
        "company_cases": company_cases,
        "industry_cases": industry_cases,
        "functional_cases": functional,
    }
    (DERIVED_DIR / "demo_cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (DOCS_DIR / "demo_cases.md").write_text(casebook, encoding="utf-8")
    (DOCS_DIR / "demo_walkthrough.md").write_text(walkthrough, encoding="utf-8")
    print(json.dumps({
        "company_event_types": len(company_cases),
        "industry_event_types": len(industry_cases),
        "company_holdout_correct_cases": sum(case["origin"] == "labelled_holdout_correct_prediction" for case in company_cases),
        "industry_holdout_correct_cases": sum(case["origin"] == "labelled_holdout_correct_prediction" for case in industry_cases),
        "outputs": ["docs/demo_cases.md", "docs/demo_walkthrough.md", "data/derived/demo_cases.json"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
