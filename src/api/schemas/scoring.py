"""
评分结果 Schema — 与 B（ML 可信度/影响力评估）对齐的接口契约

B 队友的输出格式 = 本文件的 ScoredEventGroupSchema
"""
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.api.schemas.cluster import EventGroupSchema


class ScoredEventGroupSchema(EventGroupSchema):
    """带评分的事件组（B 模块输出→C 模块消费）"""
    # ── B 模块输出的评分 ──
    credibility_score: Optional[float] = Field(
        None, description="可信度评分 0~1", ge=0, le=1
    )
    impact_level: Optional[str] = Field(
        None, description="影响等级: high/moderate/low"
    )
    impact_score: Optional[float] = Field(
        None, description="影响力评分 0~1", ge=0, le=1
    )
    sentiment_score: Optional[float] = Field(
        None, description="情绪量化 -1~1", ge=-1, le=1
    )
    b_score_components: dict[str, Any] = Field(
        default_factory=dict, description="B module scoring components"
    )
    b_score_reason: Optional[str] = Field(None, description="B module score explanation")
    b_alert_level_hint: Optional[str] = Field(
        None, description="B module priority hint: P0/P1/P2/P3"
    )
    b_match_method: Optional[str] = Field(None, description="How B artifact was matched")
    b_match_score: Optional[float] = Field(None, description="Confidence of B artifact matching")

    # ── C 模块基于B评分产出的字段 ──
    alert_level: Optional[str] = Field(
        None, description="预警级别: P0/P1/P2/P3"
    )
    alert_reason: Optional[str] = Field(None, description="预警原因")

    alert_factors: dict[str, float | int | str | None] = Field(
        default_factory=dict, description="Structured factors used by the alert engine"
    )
    previous_alert_level: Optional[str] = Field(None, description="Previous alert level for replay/diff")
    alert_changed_reason: Optional[str] = Field(None, description="Why alert level changed")


class ScoringResultResponse(BaseModel):
    """评分结果列表响应"""
    total: int
    results: list[ScoredEventGroupSchema]

