"""
事件 Schema — 与 A（NLP/LLM 事件抽取）对齐的接口契约

A 队友的输出格式 = 本文件的 EventSchema
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class EventSchema(BaseModel):
    """单条事件（A 模块输出→C 模块输入）"""
    article_file_id: str = Field(..., description="原文唯一ID")
    article_title: str = Field("", description="新闻标题")
    article_publish_time: Optional[datetime] = Field(None, description="新闻发布时间")
    article_source: str = Field("", description="来源网站/媒体")
    content: str = Field("", description="新闻正文")

    # ── A 模块抽取的字段 ──
    trading_code: Optional[str] = Field(None, description="标的代码")
    secu_abbr: Optional[str] = Field(None, description="标的简称")
    industry_code: Optional[str] = Field(None, description="行业代码")
    industry_name: Optional[str] = Field(None, description="行业名称")
    event_name: Optional[str] = Field(None, description="事件类型名称")
    event_time: Optional[datetime] = Field(None, description="事件发生时间（A抽取）")
    event_entity: Optional[str] = Field(None, description="事件主体实体（A抽取）")
    event_impact_direction: Optional[str] = Field(
        None, description="影响方向: 正面/负面/中性"
    )


class EventListResponse(BaseModel):
    """事件列表响应"""
    total: int
    events: list[EventSchema]
