"""Member-B style credibility and impact scorer for main event groups.

This module ports the useful parts of member B's offline baseline into the main
system contract: input EventGroupSchema, output ScoredEventGroupSchema.  It does
not consume B's generated cluster ids or dashboard artifacts.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

from src.api.schemas.cluster import EventGroupSchema
from src.api.schemas.scoring import ScoredEventGroupSchema
from src.services.b_event_classifier import BEventClassifier, get_default_b_event_classifier


SOURCE_RULES = (
    ("official", 1.00, ("交易所", "证监会", "人民银行", "国务院", "政府", "发改委", "财政部", "工信部", "法院")),
    ("authoritative_financial", 0.85, ("证券时报", "上海证券报", "中国证券报", "证券日报", "财联社", "新华社", "人民日报", "经济日报")),
    ("mainstream_media", 0.70, ("第一财经", "21世纪", "澎湃", "每日经济新闻", "界面新闻", "新浪财经")),
)

OFFICIAL_CUES = ("公告", "交易所", "证监会", "官网", "正式发布", "官方", "审议通过", "批复", "备案", "中标通知书")
RUMOUR_CUES = ("传闻", "网传", "据悉", "市场消息", "或将", "可能", "知情人士", "消息称", "未经证实")
SPECIFIC_CUES = ("亿元", "万元", "同比", "环比", "合同", "协议", "项目", "日期", "年", "月", "日")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:%|亿元|万元|元|年|月|日|倍|项|家|股)?")

SENTIMENT_MAP = {
    "正面": 0.7,
    "利好": 0.7,
    "positive": 0.7,
    "负面": -0.7,
    "利空": -0.7,
    "negative": -0.7,
    "中性": 0.0,
    "neutral": 0.0,
}

STATUS_SCORE = {
    "official_confirmation": 0.95,
    "confirmed": 0.95,
    "impact_update": 0.78,
    "follow_up": 0.65,
    "initial_report": 0.55,
    "clarification": 0.10,
    "reported": 0.55,
    "watching": 0.35,
    "clarified": 0.10,
}


class BAlgorithmScorer:
    """Direct scorer using B's evidence-confidence and impact ideas."""

    @classmethod
    def score_batch(
        cls,
        groups: list[EventGroupSchema],
        classifier: BEventClassifier | None = None,
        use_event_classifier: bool = True,
    ) -> list[ScoredEventGroupSchema]:
        if use_event_classifier and classifier is None:
            classifier = get_default_b_event_classifier()
        return [cls.score(group, classifier=classifier if use_event_classifier else None) for group in groups]

    @classmethod
    def score(
        cls,
        group: EventGroupSchema,
        classifier: BEventClassifier | None = None,
    ) -> ScoredEventGroupSchema:
        classification = classifier.classify_group(group) if classifier else None
        if classification:
            group = classifier.apply_classification(group, classification)

        credibility_score, credibility_components, credibility_reason = cls._credibility(group)
        sentiment_score = cls._sentiment(group)
        impact_score, impact_level, impact_components, impact_reason = cls._impact(
            group,
            credibility_score,
            sentiment_score,
        )

        components: dict[str, Any] = {
            **{f"credibility_{key}": value for key, value in credibility_components.items()},
            **{f"impact_{key}": value for key, value in impact_components.items()},
        }
        if classification:
            components.update(
                {
                    "classification_event_name": classification.event_name,
                    "classification_scope": classification.scope,
                    "classification_classifier": classification.classifier,
                    "classification_confidence": classification.confidence,
                    "classification_margin": classification.margin,
                    "classification_review_required": classification.review_required,
                }
            )
        reason = "; ".join(part for part in (credibility_reason, impact_reason) if part)
        if classification:
            reason = (
                f"分类模型={cls._classifier_desc(classification.classifier)}, "
                f"分类置信度={classification.confidence:.2f}; {reason}"
            )

        return ScoredEventGroupSchema(
            **group.model_dump(),
            credibility_score=round(credibility_score, 4),
            impact_level=impact_level,
            impact_score=round(impact_score, 4),
            sentiment_score=round(sentiment_score, 4),
            b_score_components=components,
            b_score_reason=reason,
            b_match_method="algorithm",
            b_match_score=1.0,
        )

    @classmethod
    def _credibility(cls, group: EventGroupSchema) -> tuple[float, dict[str, float], str]:
        events = group.events
        source_profiles = [cls._source_profile(event.article_source) for event in events]
        authority_values = [score for _, score in source_profiles]
        max_authority = max(authority_values, default=0.50)
        avg_authority = sum(authority_values) / len(authority_values) if authority_values else 0.50
        source_authority = 0.65 * max_authority + 0.35 * avg_authority

        unique_sources = {event.article_source for event in events if event.article_source}
        cross_source = 1 - math.exp(-0.65 * max(0, len(unique_sources) - 1))

        official_hits = sum(cls._contains_any(cls._article_text(event), OFFICIAL_CUES) for event in events)
        official_evidence = min(1.0, 0.55 * official_hits + (0.25 if max_authority >= 1.0 else 0.0))

        specificity_scores = [cls._specificity(cls._article_text(event)) for event in events]
        content_specificity = (
            sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0.0
        )

        cluster_agreement = max(0.0, min(1.0, group.cluster_confidence or 0.0))
        if group.evidence_edges:
            edge_mean = sum(edge.similarity for edge in group.evidence_edges) / len(group.evidence_edges)
            cluster_agreement = max(cluster_agreement, max(0.0, min(1.0, (edge_mean - 0.65) / 0.35)))

        rumour_hits = sum(cls._contains_any(cls._article_text(event), RUMOUR_CUES) for event in events)
        rumour_penalty = min(0.25, 0.07 * rumour_hits)

        total = (
            0.35 * source_authority
            + 0.20 * cross_source
            + 0.18 * official_evidence
            + 0.17 * content_specificity
            + 0.10 * cluster_agreement
            - rumour_penalty
        )
        total = max(0.0, min(1.0, total))

        components = {
            "source_authority": round(source_authority, 4),
            "cross_source_corroboration": round(cross_source, 4),
            "official_evidence": round(official_evidence, 4),
            "content_specificity": round(content_specificity, 4),
            "cluster_agreement": round(cluster_agreement, 4),
            "rumour_penalty": round(rumour_penalty, 4),
        }

        reasons = []
        if max_authority >= 0.85:
            reasons.append("包含权威或官方来源证据")
        if len(unique_sources) >= 2:
            reasons.append(f"得到 {len(unique_sources)} 个独立来源印证")
        if official_hits:
            reasons.append("包含公告、监管或官方发布线索")
        if rumour_hits:
            reasons.append("检测到传闻式表述并降权")
        if not reasons:
            reasons.append("证据较少，建议人工复核")
        return total, components, "; ".join(reasons)

    @classmethod
    def _impact(
        cls,
        group: EventGroupSchema,
        credibility_score: float,
        sentiment_score: float,
    ) -> tuple[float, str, dict[str, float], str]:
        materiality, materiality_reason = cls._materiality(group.event_name or "", sentiment_score)
        source_count = len({event.article_source for event in group.events if event.article_source})
        breadth = min(1.0, 0.18 * source_count + 0.10 * max(0, group.article_count - 1))
        state = cls._state_confirmation(group)
        freshness = cls._freshness(group)

        score = (
            0.30 * credibility_score
            + 0.25 * materiality
            + 0.15 * breadth
            + 0.15 * state
            + 0.15 * freshness
        )
        if cls._has_stage(group, "clarification"):
            score = min(score, 0.29)
        score = max(0.0, min(1.0, score))
        level = "high" if score >= 0.70 else ("moderate" if score >= 0.45 else "low")
        components = {
            "evidence_confidence": round(credibility_score, 4),
            "event_materiality": round(materiality, 4),
            "information_breadth": round(breadth, 4),
            "state_confirmation": round(state, 4),
            "freshness": round(freshness, 4),
        }
        reason = f"{materiality_reason}; 信息广度={breadth:.2f}; 状态确认={state:.2f}; 新鲜度={freshness:.2f}"
        return score, level, components, reason

    @staticmethod
    def _source_profile(source: str | None) -> tuple[str, float]:
        source = source or ""
        for tier, score, patterns in SOURCE_RULES:
            if any(pattern in source for pattern in patterns):
                return tier, score
        return "unclassified", 0.50

    @staticmethod
    def _article_text(event: Any) -> str:
        return f"{event.article_title or ''} {event.content or ''} {event.article_source or ''}"

    @staticmethod
    def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
        return any(cue in text for cue in cues)

    @staticmethod
    def _specificity(text: str) -> float:
        numbers = len(NUMBER_PATTERN.findall(text))
        cues = sum(cue in text for cue in SPECIFIC_CUES)
        return min(1.0, 0.15 * min(numbers, 4) + 0.08 * min(cues, 5))

    @staticmethod
    def _sentiment(group: EventGroupSchema) -> float:
        directions = [event.event_impact_direction for event in group.events if event.event_impact_direction]
        if not directions:
            return 0.0
        mode = Counter(directions).most_common(1)[0][0]
        return SENTIMENT_MAP.get(str(mode).strip().lower(), SENTIMENT_MAP.get(str(mode), 0.0))

    @staticmethod
    def _materiality(event_name: str, sentiment: float) -> tuple[float, str]:
        if any(word in event_name for word in ("风险", "违约", "处罚", "调查", "诉讼", "减持", "亏损")):
            return 0.88, "风险或监管类事件"
        if any(word in event_name for word in ("资本", "并购", "重组", "重大项目", "订单", "中标", "投资", "融资")):
            return 0.78, "资本、项目、订单或融资类事件"
        if any(word in event_name for word in ("政策", "调控", "补贴", "利率", "产量")):
            return 0.72, "政策或供需类事件"
        if any(word in event_name for word in ("技术", "创新", "突破", "研发", "产品")):
            return 0.62, "技术或产品进展事件"
        return (0.60 if abs(sentiment) > 0.05 else 0.45), "一般经营动态"

    @staticmethod
    def _classifier_desc(classifier: str) -> str:
        return {
            "b_issuer_title_semantic_hybrid": "标题语义融合分类",
            "b_issuer_title_character_nn": "标题近邻分类",
            "b_body_character_nb": "正文字符朴素贝叶斯",
        }.get(classifier, classifier)

    @classmethod
    def _state_confirmation(cls, group: EventGroupSchema) -> float:
        if not group.timeline_stages:
            return 0.55
        values = [STATUS_SCORE.get(stage.stage, 0.55) for stage in group.timeline_stages]
        return max(values, default=0.55)

    @staticmethod
    def _freshness(group: EventGroupSchema) -> float:
        if not group.timeline_start or not group.timeline_end:
            return 0.50
        start = group.timeline_start
        end = group.timeline_end
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            return 0.50
        age_days = max(0.0, (end - start).total_seconds() / 86400)
        return max(0.15, min(1.0, 1.0 - age_days * 0.03))

    @staticmethod
    def _has_stage(group: EventGroupSchema, stage_name: str) -> bool:
        return any(stage.stage == stage_name for stage in group.timeline_stages)
