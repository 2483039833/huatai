"""Create event timelines and state transitions from aggregated news clusters."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "event_radar" / "data" / "processed"
DERIVED_DIR = ROOT / "event_radar" / "data" / "derived"
REPORT_DIR = ROOT / "event_radar" / "reports"
CLUSTER_PATH = DERIVED_DIR / "deduplication_clusters_test.jsonl.gz"
CREDIBILITY_PATH = DERIVED_DIR / "credibility_assessment_test.jsonl.gz"
OUTPUT_PATH = DERIVED_DIR / "event_timelines_test.jsonl.gz"

OFFICIAL_CUES = ("公告", "交易所", "证监会", "官网", "正式发布", "官方", "审议通过", "批复", "备案", "中标通知书")
CLARIFICATION_CUES = ("澄清", "否认", "不实", "辟谣", "失实", "不存在", "未发生", "不属实")
IMPACT_CUES = ("影响", "扩大", "增长", "下滑", "订单", "签约", "中标", "停产", "亏损", "涨停", "跌停", "处罚", "终止")
RUMOUR_CUES = ("传闻", "网传", "据悉", "市场消息", "或将", "可能", "知情人士", "消息称", "未经证实")
SENTENCE_SPLIT = re.compile(r"[。！？；\n]+")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def first_evidence(text: str, cues: tuple[str, ...]) -> str | None:
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if sentence and any(cue in sentence for cue in cues):
            return sentence[:260]
    return None


def milestone(article: dict[str, Any], index: int, distinct_sources: set[str]) -> tuple[str, str | None]:
    text = f"{article.get('title') or ''} {article.get('content') or ''}"
    if first_evidence(text, CLARIFICATION_CUES):
        return "clarification", first_evidence(text, CLARIFICATION_CUES)
    if first_evidence(text, OFFICIAL_CUES):
        return "official_confirmation", first_evidence(text, OFFICIAL_CUES)
    if first_evidence(text, IMPACT_CUES) and index > 0:
        return "impact_update", first_evidence(text, IMPACT_CUES)
    if article.get("source") and article["source"] not in distinct_sources:
        return "multi_source_follow_up", article.get("title")
    return ("initial_report" if index == 0 else "follow_up"), article.get("title")


def next_state(current: str, event_type: str, article: dict[str, Any], source_count: int) -> str:
    text = f"{article.get('title') or ''} {article.get('content') or ''}"
    if event_type == "clarification":
        return "clarified"
    if event_type == "official_confirmation":
        return "confirmed"
    if current == "clarified":
        return "clarified"
    if source_count >= 2:
        return "corroborated"
    if any(cue in text for cue in RUMOUR_CUES):
        return "watching"
    return "reported"


def timeline_summary(milestones: list[dict[str, Any]], current_status: str) -> str:
    first = milestones[0]
    pieces = [f"{first['published_at'] or '未知时间'}首发"]
    if any(item["type"] == "official_confirmation" for item in milestones):
        pieces.append("后续出现官方确认线索")
    if any(item["type"] == "clarification" for item in milestones):
        pieces.append("后续出现澄清/反转信息")
    if len({item.get('source') for item in milestones if item.get('source')}) >= 2:
        pieces.append("获得多来源跟进")
    pieces.append(f"当前状态：{current_status}")
    return "；".join(pieces)


def main() -> None:
    if not CLUSTER_PATH.exists() or not CREDIBILITY_PATH.exists():
        raise FileNotFoundError("Run deduplication_baseline.py and credibility_baseline.py before timeline_baseline.py")
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    article_map = {
        row["article_id"]: row
        for filename in ("duplicates_test_company.jsonl.gz", "duplicates_test_industry.jsonl.gz")
        for row in iter_jsonl(DATA_DIR / filename)
    }
    credibility_by_cluster = {row["cluster_id"]: row["credibility"] for row in iter_jsonl(CREDIBILITY_PATH)}
    timelines = []
    for cluster in iter_jsonl(CLUSTER_PATH):
        articles = sorted(
            (article_map[item] for item in cluster["article_ids"] if item in article_map),
            key=lambda item: item.get("published_at") or "9999-12-31",
        )
        distinct_sources: set[str] = set()
        state = "watching"
        milestones = []
        for index, article in enumerate(articles):
            event_type, evidence = milestone(article, index, distinct_sources)
            if article.get("source"):
                distinct_sources.add(article["source"])
            state = next_state(state, event_type, article, len(distinct_sources))
            milestones.append({
                "sequence": index + 1,
                "type": event_type,
                "published_at": article.get("published_at"),
                "article_id": article["article_id"],
                "source": article.get("source"),
                "title": article.get("title"),
                "evidence": evidence,
                "status_after": state,
            })
        timelines.append({
            "cluster_id": cluster["cluster_id"],
            "canonical_article_id": cluster["canonical_article_id"],
            "canonical_title": cluster["canonical_title"],
            "current_status": state,
            "first_published_at": milestones[0]["published_at"] if milestones else None,
            "last_updated_at": milestones[-1]["published_at"] if milestones else None,
            "source_count": len(distinct_sources),
            "article_count": len(articles),
            "credibility": credibility_by_cluster.get(cluster["cluster_id"], {}),
            "timeline_summary": timeline_summary(milestones, state) if milestones else "无可用新闻",
            "milestones": milestones,
        })
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as target:
        for timeline in timelines:
            target.write(json.dumps(timeline, ensure_ascii=False) + "\n")

    status_distribution = Counter(item["current_status"] for item in timelines)
    milestone_distribution = Counter(milestone["type"] for item in timelines for milestone in item["milestones"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "chronological event state machine using official-confirmation, clarification, impact and rumour cues",
        "timelines": len(timelines),
        "average_articles_per_timeline": round(sum(item["article_count"] for item in timelines) / len(timelines), 2),
        "current_status_distribution": dict(status_distribution),
        "milestone_distribution": dict(milestone_distribution),
        "timelines_with_official_confirmation": sum(any(item["type"] == "official_confirmation" for item in timeline["milestones"]) for timeline in timelines),
        "timelines_with_clarification": sum(any(item["type"] == "clarification" for item in timeline["milestones"]) for timeline in timelines),
        "output_file": OUTPUT_PATH.name,
        "limitation": "The dataset does not label event lifecycle states; state transitions are explainable rule-based signals for monitoring and should be reviewed in production.",
    }
    (REPORT_DIR / "timeline_baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# 事件时间线追踪报告", "",
        "按事件簇内新闻的发布时间排序，识别首发、媒体跟进、官方确认、影响更新和澄清节点，并以状态机生成当前状态。", "",
        f"- 事件时间线：{report['timelines']} 条",
        f"- 平均每条时间线新闻数：{report['average_articles_per_timeline']}",
        f"- 当前状态分布：`{json.dumps(report['current_status_distribution'], ensure_ascii=False)}`",
        f"- 含官方确认节点：{report['timelines_with_official_confirmation']} 条",
        f"- 含澄清节点：{report['timelines_with_clarification']} 条",
        "",
        "限制：赛题数据不含事件生命周期标注，当前状态用于可解释的监测与展示；上线后应通过人工反馈或公告对齐来校准状态转移规则。",
    ]
    (REPORT_DIR / "timeline_baseline_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
