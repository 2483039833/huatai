"""Build explainable credibility assessments for aggregated news events.

The supplied data has no ground-truth truthfulness label.  This module therefore
reports a transparent *evidence confidence* score rather than claiming to decide
truth.  It combines source authority, independent corroboration, official cues,
specificity, agreement and rumour-style language.
"""

from __future__ import annotations

import gzip
import json
import math
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
EDGE_PATH = DERIVED_DIR / "deduplication_edges_test.jsonl.gz"
OUTPUT_PATH = DERIVED_DIR / "credibility_assessment_test.jsonl.gz"

SOURCE_RULES = (
    ("official", 100, ("交易所", "证监会", "人民银行", "国务院", "政府", "发改委", "财政部", "工信部", "国家能源局", "法院", "检察院")),
    ("authoritative_financial", 85, ("证券时报", "上海证券报", "上证报", "中国证券报", "证券日报", "财联社", "新华社", "人民日报", "经济日报", "央视", "中国基金报")),
    ("mainstream_media", 70, ("第一财经", "21世纪", "澎湃", "每日经济新闻", "界面新闻", "中国新闻", "南方", "新浪财经")),
)
OFFICIAL_CUES = ("公告", "交易所", "证监会", "官网", "正式发布", "官方", "审议通过", "批复", "备案", "中标通知书")
RUMOUR_CUES = ("传闻", "网传", "据悉", "市场消息", "或将", "可能", "知情人士", "消息称", "未经证实")
SPECIFIC_CUES = ("亿元", "万元", "同比", "环比", "合同", "协议", "项目", "日期", "年", "月", "日")
SENTENCE_SPLIT = re.compile(r"[。！？；\n]+")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:%|亿元|万元|万|亿|年|月|日|份|项|台|吨)")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def source_profile(source: str | None) -> tuple[str, int]:
    source = source or ""
    for tier, score, patterns in SOURCE_RULES:
        if any(pattern in source for pattern in patterns):
            return tier, score
    return "unclassified", 50


def evidence_sentences(text: str, cues: tuple[str, ...], max_items: int = 2) -> list[str]:
    matches = []
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if sentence and any(cue in sentence for cue in cues):
            matches.append(sentence[:280])
    return matches[:max_items]


def article_specificity(text: str) -> int:
    numbers = len(NUMBER_PATTERN.findall(text))
    cues = sum(cue in text for cue in SPECIFIC_CUES)
    return min(100, 15 * min(numbers, 4) + 8 * min(cues, 5))


def cluster_similarity(article_ids: set[str], edges: list[dict[str, Any]]) -> float:
    values = [edge["similarity"] for edge in edges if edge["same_event"] and {edge["article_id_left"], edge["article_id_right"]} <= article_ids]
    if not values:
        return 0.5
    # Similarity thresholds are calibrated around 0.73; scale the accepted edge range to 0-1.
    return min(1.0, max(0.0, (sum(values) / len(values) - 0.65) / 0.35))


def level(score: float) -> str:
    if score >= 80:
        return "high"
    if score >= 60:
        return "medium"
    if score >= 40:
        return "low"
    return "very_low"


def assess_cluster(cluster: dict[str, Any], article_map: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    articles = [article_map[item] for item in cluster["article_ids"] if item in article_map]
    source_rows = []
    official_evidence = []
    rumour_matches = []
    specificity_scores = []
    authority_scores = []
    for article in articles:
        tier, authority = source_profile(article.get("source"))
        authority_scores.append(authority)
        text = article.get("text") or ""
        official = evidence_sentences(text, OFFICIAL_CUES)
        rumour = evidence_sentences(text, RUMOUR_CUES)
        official_evidence.extend({"article_id": article["article_id"], "sentences": official} for _ in [0] if official)
        rumour_matches.extend(rumour)
        specificity_scores.append(article_specificity(text))
        source_rows.append({
            "article_id": article["article_id"],
            "source": article.get("source"),
            "source_tier": tier,
            "authority_score": authority,
            "published_at": article.get("published_at"),
            "title": article.get("title"),
        })

    unique_sources = {item["source"] for item in source_rows if item["source"]}
    max_authority = max(authority_scores, default=0)
    average_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0
    authority_score = 0.65 * max_authority + 0.35 * average_authority
    corroboration_score = 100 * (1 - math.exp(-0.65 * max(0, len(unique_sources) - 1)))
    official_score = min(100, 55 * len(official_evidence) + (25 if max_authority == 100 else 0))
    specificity_score = sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0
    agreement_score = 100 * cluster_similarity({item["article_id"] for item in articles}, edges)
    rumour_penalty = min(25, 7 * len(rumour_matches))
    total = max(0, min(100, 0.35 * authority_score + 0.20 * corroboration_score + 0.18 * official_score + 0.17 * specificity_score + 0.10 * agreement_score - rumour_penalty))

    reasons = []
    if max_authority >= 85:
        reasons.append("包含权威财经媒体或官方来源")
    if len(unique_sources) >= 2:
        reasons.append(f"得到 {len(unique_sources)} 个独立来源交叉印证")
    if official_evidence:
        reasons.append("原文包含公告、监管或官方发布线索")
    if rumour_matches:
        reasons.append("包含传闻式或不确定性表述，已降权")
    if not reasons:
        reasons.append("证据较少，建议人工复核")

    return {
        "cluster_id": cluster["cluster_id"],
        "canonical_article_id": cluster["canonical_article_id"],
        "canonical_title": cluster["canonical_title"],
        "article_count": len(articles),
        "credibility": {
            "score": round(total, 2),
            "level": level(total),
            "status": "corroborated" if len(unique_sources) >= 2 else "unverified",
            "components": {
                "source_authority": round(authority_score, 2),
                "cross_source_corroboration": round(corroboration_score, 2),
                "official_evidence": round(official_score, 2),
                "content_specificity": round(specificity_score, 2),
                "cluster_agreement": round(agreement_score, 2),
                "rumour_penalty": round(rumour_penalty, 2),
            },
            "reasons": reasons,
        },
        "evidence_chain": {
            "sources": sorted(source_rows, key=lambda item: item["authority_score"], reverse=True),
            "official_cues": official_evidence,
            "rumour_cues": rumour_matches[:4],
        },
    }


def main() -> None:
    if not CLUSTER_PATH.exists() or not EDGE_PATH.exists():
        raise FileNotFoundError("Run deduplication_baseline.py before credibility_baseline.py")
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    article_map = {
        row["article_id"]: row
        for filename in ("duplicates_test_company.jsonl.gz", "duplicates_test_industry.jsonl.gz")
        for row in iter_jsonl(DATA_DIR / filename)
    }
    clusters = list(iter_jsonl(CLUSTER_PATH))
    edges = list(iter_jsonl(EDGE_PATH))
    assessments = [assess_cluster(cluster, article_map, edges) for cluster in clusters]
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as target:
        for assessment in assessments:
            target.write(json.dumps(assessment, ensure_ascii=False) + "\n")

    distribution = Counter(item["credibility"]["level"] for item in assessments)
    source_tiers = Counter(
        source["source_tier"]
        for item in assessments for source in item["evidence_chain"]["sources"]
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "evidence confidence, not ground-truth fact verification",
        "method": "weighted source authority + cross-source corroboration + official cues + specificity + cluster agreement - rumour penalty",
        "assessed_event_clusters": len(assessments),
        "credibility_level_distribution": dict(distribution),
        "source_tier_distribution": dict(source_tiers),
        "events_with_multiple_sources": sum(len(item["evidence_chain"]["sources"]) >= 2 for item in assessments),
        "events_with_official_cues": sum(bool(item["evidence_chain"]["official_cues"]) for item in assessments),
        "output_file": OUTPUT_PATH.name,
        "limitation": "The competition data contains no factual truth labels. Scores express evidentiary confidence and must be treated as a review priority, not an automated fact verdict.",
    }
    (REPORT_DIR / "credibility_baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# 可信度评估基线报告", "",
        "本模块输出的是**证据可信度**，不是未经证实的自动事实裁决。它用于决定复核优先级与预警力度。", "",
        f"- 已评估事件簇：{report['assessed_event_clusters']} 个",
        f"- 多来源交叉印证事件：{report['events_with_multiple_sources']} 个",
        f"- 含公告/官方线索事件：{report['events_with_official_cues']} 个",
        f"- 可信度等级分布：`{json.dumps(report['credibility_level_distribution'], ensure_ascii=False)}`",
        "",
        "评分构成：35% 来源权威性、20% 多源交叉印证、18% 官方线索、17% 文本具体性、10% 簇内一致性，并扣除传闻式表述风险。",
        "",
        "限制：赛题数据未提供事实真伪标签，因此不能报告真实性分类准确率。后续应接入公告、监管文件等权威外部证据，并通过人工复核回流校准权重。",
    ]
    (REPORT_DIR / "credibility_baseline_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
