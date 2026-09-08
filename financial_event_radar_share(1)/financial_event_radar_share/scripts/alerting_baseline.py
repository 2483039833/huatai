"""Generate explainable impact scores and alert actions from event timelines."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "event_radar" / "data" / "derived"
REPORT_DIR = ROOT / "event_radar" / "reports"
TIMELINE_PATH = DERIVED_DIR / "event_timelines_test.jsonl.gz"
PREDICTION_PATHS = (
    DERIVED_DIR / "event_predictions_duplicates_test_company.jsonl.gz",
    DERIVED_DIR / "event_predictions_duplicates_test_industry.jsonl.gz",
)
OUTPUT_PATH = DERIVED_DIR / "alert_feed_test.jsonl.gz"

STATUS_SCORE = {
    "confirmed": 95,
    "corroborated": 78,
    "reported": 55,
    "watching": 35,
    "clarified": 10,
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def materiality(event_name: str, direction: str) -> tuple[int, str]:
    name = event_name or "未分类事件"
    if any(word in name for word in ("风险", "违约", "处罚", "调查", "诉讼", "减持", "亏损")):
        return 88, "风险或监管类事件"
    if any(word in name for word in ("资本", "并购", "重组", "重大项目", "订单", "中标", "投资", "融资")):
        return 78, "资本、项目或订单类事件"
    if any(word in name for word in ("政策", "调控", "补贴", "利率", "产量")):
        return 72, "政策或行业供需类事件"
    if any(word in name for word in ("技术", "创新", "突破", "研发", "产品")):
        return 62, "技术或产品进展事件"
    return (60 if direction != "neutral" else 45), "一般经营动态"


def recency_score(timestamp: str | None, anchor: str | None) -> int:
    if not timestamp or not anchor:
        return 50
    try:
        current = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        latest = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
        age_days = max(0, (latest - current).total_seconds() / 86400)
        return round(max(15, 100 - age_days * 3))
    except ValueError:
        return 50


def alert_level(score: float, status: str) -> tuple[str, str]:
    if status == "clarified":
        return "P0", "withdraw_or_update"
    if score >= 80:
        return "P3", "immediate_push"
    if score >= 65:
        return "P2", "priority_push"
    if score >= 45:
        return "P1", "watchlist"
    return "P0", "archive"


def reasons(
    level: str, event_category: str, credibility: dict[str, Any], status: str,
    source_count: int, direction: str,
    review_required: bool,
) -> list[str]:
    result = [event_category]
    if credibility.get("level") in {"high", "medium"}:
        result.append(f"证据可信度为 {credibility['level']}")
    if source_count >= 2:
        result.append(f"已获得 {source_count} 个来源跟进")
    if status == "confirmed":
        result.append("时间线出现官方确认线索")
    if status == "clarified":
        result.append("出现澄清信息，建议撤回或更新旧预警")
    if direction != "neutral":
        result.append(f"影响方向判定为 {direction}")
    if level == "P3":
        result.append("达到即时推送阈值")
    if review_required:
        result.append("事件类型分类置信度较低，需人工复核")
    return result


def main() -> None:
    if not TIMELINE_PATH.exists() or not all(path.exists() for path in PREDICTION_PATHS):
        raise FileNotFoundError("Run event_extraction_baseline.py and timeline_baseline.py before alerting_baseline.py")
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = {row["article_id"]: row["event"] for path in PREDICTION_PATHS for row in iter_jsonl(path)}
    timelines = list(iter_jsonl(TIMELINE_PATH))
    anchor = max((item.get("last_updated_at") for item in timelines if item.get("last_updated_at")), default=None)
    alerts = []
    for timeline in timelines:
        event = predictions.get(timeline["canonical_article_id"], {})
        event_name = event.get("event_name", "未分类事件")
        direction = event.get("impact_direction", "neutral")
        materiality_score, event_category = materiality(event_name, direction)
        credibility = timeline.get("credibility", {})
        credibility_score = float(credibility.get("score", 0))
        breadth_score = min(100, 18 * timeline["source_count"] + 10 * max(0, timeline["article_count"] - 1))
        state_score = STATUS_SCORE.get(timeline["current_status"], 35)
        freshness_score = recency_score(timeline.get("last_updated_at"), anchor)
        score = (
            0.30 * credibility_score
            + 0.25 * materiality_score
            + 0.15 * breadth_score
            + 0.15 * state_score
            + 0.15 * freshness_score
        )
        if timeline["current_status"] == "clarified":
            score = min(score, 29)
        score = round(score, 2)
        priority, action = alert_level(score, timeline["current_status"])
        review_required = bool(event.get("scope") == "company" and event.get("confidence", 0) < 0.75)
        if review_required and action in {"immediate_push", "priority_push"}:
            priority, action = "P1", "manual_review"
        alerts.append({
            "alert_id": f"ALERT-{timeline['cluster_id']}",
            "cluster_id": timeline["cluster_id"],
            "title": timeline["canonical_title"],
            "event_type": event_name,
            "subjects": event.get("subjects", []),
            "event_time": event.get("event_time", {}),
            "impact_direction": direction,
            "event_classification_confidence": event.get("confidence"),
            "review_required": review_required,
            "timeline_status": timeline["current_status"],
            "credibility": credibility,
            "impact_score": score,
            "priority": priority,
            "action": action,
            "push_reason": reasons(priority, event_category, credibility, timeline["current_status"], timeline["source_count"], direction, review_required),
            "score_components": {
                "evidence_confidence": round(credibility_score, 2),
                "event_materiality": materiality_score,
                "information_breadth": breadth_score,
                "state_confirmation": state_score,
                "freshness": freshness_score,
                "event_classification_confidence": round(100 * float(event.get("confidence", 0)), 2),
            },
            "evidence": timeline.get("milestones", [])[:3],
            "last_updated_at": timeline.get("last_updated_at"),
        })
    alerts.sort(key=lambda item: (item["priority"], item["impact_score"]), reverse=True)
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as target:
        for alert in alerts:
            target.write(json.dumps(alert, ensure_ascii=False) + "\n")

    priority_distribution = Counter(item["priority"] for item in alerts)
    action_distribution = Counter(item["action"] for item in alerts)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "weighted evidence confidence + event materiality + information breadth + state confirmation + freshness",
        "alerts": len(alerts),
        "priority_distribution": dict(priority_distribution),
        "action_distribution": dict(action_distribution),
        "clarification_updates": sum(item["action"] == "withdraw_or_update" for item in alerts),
        "manual_reviews": sum(item["action"] == "manual_review" for item in alerts),
        "immediate_pushes": sum(item["action"] == "immediate_push" for item in alerts),
        "output_file": OUTPUT_PATH.name,
        "limitation": "Impact score weights are initial business rules. The data has no labelled alert priority, so production thresholds require investment-research and risk-control review.",
    }
    (REPORT_DIR / "alerting_baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# 事件影响分级与预警报告", "",
        "预警分数综合证据可信度、事件重要性、信息传播范围、状态确认程度和时效性；澄清事件不会触发新预警，而会生成撤回/更新动作。", "",
        f"- 已生成预警决策：{report['alerts']} 条",
        f"- 预警等级分布：`{json.dumps(report['priority_distribution'], ensure_ascii=False)}`",
        f"- 即时推送：{report['immediate_pushes']} 条",
        f"- 撤回/更新：{report['clarification_updates']} 条",
        "",
        "限制：赛题数据未提供预警优先级标签，当前权重是可解释的初始业务规则，应通过投研与风控人工复核进行阈值校准。",
    ]
    (REPORT_DIR / "alerting_baseline_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
