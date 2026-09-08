"""
聚类结果 Schema — C 模块内部使用 + C→B 的输入
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from src.api.schemas.event import EventSchema

class EvidenceEdge(BaseModel):
    """Pairwise evidence used to explain why articles are in the same group."""
    source_article_id: str
    target_article_id: str
    similarity: float = Field(..., ge=0, le=1)


class TimelineStage(BaseModel):
    """A stage in the event development timeline."""
    stage: str
    article_id: str
    title: str = ""
    time: Optional[datetime] = None
    source: str = ""
    reason: str = ""

class EventGroupSchema(BaseModel):
    """聚类后的事件组（C 模块输出→B 模块输入）"""
    group_id: str = Field(..., description="聚类组唯一ID")
    events: list[EventSchema] = Field(default_factory=list, description="组内所有事件")
    cluster_confidence: float = Field(0.0, description="聚类置信度", ge=0, le=1)
    timeline_start: Optional[datetime] = Field(None, description="事件最早时间")
    timeline_end: Optional[datetime] = Field(None, description="事件最晚时间")
    merged_title: str = Field("", description="合并后标题")
    merged_content: str = Field("", description="合并后摘要")
    article_count: int = Field(0, description="文章数量")
    source_diversity: float = Field(0.0, description="来源多样性（不同来源数/总数）")

    # 事件类型 & 关联实体
    event_name: Optional[str] = Field(None, description="事件类型")
    trading_codes: list[str] = Field(default_factory=list, description="关联标的代码")
    secu_names: list[str] = Field(default_factory=list, description="关联标的名称")

    # C-module explainability and tracking fields.
    representative_article_id: Optional[str] = Field(None, description="Representative article for this event group")
    evidence_edges: list[EvidenceEdge] = Field(default_factory=list, description="High-similarity article pairs")
    cluster_reason: str = Field("", description="Human-readable clustering explanation")
    timeline_stages: list[TimelineStage] = Field(default_factory=list, description="Event development stages")


class EventGroupListResponse(BaseModel):
    """事件组列表响应"""
    total_groups: int
    total_articles: int
    groups: list[EventGroupSchema]

