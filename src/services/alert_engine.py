"""Configurable alert rules for scored event groups."""
from typing import Optional
from typing import Any

from src.api.schemas.scoring import ScoredEventGroupSchema
from src.config import settings


class AlertEngine:
    """Turn B-module scores into explainable P0-P3 alert levels."""

    @staticmethod
    def _sentiment_desc(sentiment: Optional[float]) -> str:
        if sentiment is None:
            return "中性"
        if sentiment > 0.3:
            return "正面"
        if sentiment < -0.3:
            return "负面"
        return "中性"

    @staticmethod
    def _factor_value(value: Any) -> float | int | str | None:
        if value is None or isinstance(value, (float, int, str)):
            return value
        return str(value)

    @staticmethod
    def evaluate(credibility: float, impact: float) -> str:
        """Map B-style comprehensive alert score to alert level.

        B's impact score already combines evidence confidence, event materiality,
        information breadth, state confirmation and freshness.  Credibility is
        therefore kept as an explanatory factor instead of a second hard gate.
        """
        alert_score = impact
        if alert_score >= settings.alert_p0_min_impact:
            return "P0"
        if alert_score >= settings.alert_p1_min_impact:
            return "P1"
        if alert_score >= settings.alert_p2_min_impact:
            return "P2"
        return "P3"

    @staticmethod
    def _change_reason(previous: Optional[str], current: str) -> Optional[str]:
        if previous is None or previous == current:
            return None
        order = {"P3": 0, "P2": 1, "P1": 2, "P0": 3}
        if order.get(current, 0) > order.get(previous, 0):
            return f"alert upgraded from {previous} to {current}"
        return f"alert downgraded from {previous} to {current}"

    @classmethod
    def apply(cls, group: ScoredEventGroupSchema) -> ScoredEventGroupSchema:
        """Fill alert level, explanation and structured factors."""
        cred = group.credibility_score or 0.0
        imp = group.impact_score or 0.0
        sent = group.sentiment_score or 0.0

        previous = group.alert_level
        level = cls.evaluate(cred, imp)
        alert_score = imp
        factors = {
            "credibility_score": round(cred, 4),
            "impact_score": round(imp, 4),
            "alert_score": round(alert_score, 4),
            "alert_policy": "b_style_comprehensive_score",
            "sentiment_score": round(sent, 4),
            "sentiment_label": cls._sentiment_desc(sent),
            "article_count": group.article_count,
            "source_diversity": group.source_diversity,
            "cluster_confidence": group.cluster_confidence,
            "impact_level": group.impact_level,
            "b_alert_level_hint": group.b_alert_level_hint,
        }
        for key, value in group.b_score_components.items():
            factors[f"b_{key}"] = cls._factor_value(value)
        reason = (
            f"综合分={alert_score:.2f}，可信度={cred:.2f}，"
            f"影响力={imp:.2f}（{group.impact_level}），"
            f"报道数={group.article_count}，来源多样性={group.source_diversity:.0%}，"
            f"情绪方向={cls._sentiment_desc(sent)}"
        )
        if group.b_score_reason:
            reason = f"{reason}; 评分依据: {group.b_score_reason}"
        return group.model_copy(update={
            "alert_level": level,
            "alert_reason": reason,
            "alert_factors": factors,
            "previous_alert_level": previous,
            "alert_changed_reason": cls._change_reason(previous, level),
        })

    @classmethod
    def apply_batch(cls, groups: list[ScoredEventGroupSchema]) -> list[ScoredEventGroupSchema]:
        return [cls.apply(g) for g in groups]
