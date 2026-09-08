# -*- coding: utf-8 -*-
"""
事件识别与要素抽取 API
========================
对外对接接口（供其他组员调用）：

  POST /api/event/extract       单篇抽取
  POST /api/event/extract_batch 批量抽取
  GET  /api/event/schema        标准化事件知识摘要
  GET  /api/event/health        健康检查

输出字段（下游模块可直接消费）：
  event_name / event_emotion / subjects / impact_objects /
  industry / event_time / impact_direction / confidence
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from event_ner.extractor import EventExtractor
from event_ner.schema import EventSchema

router = APIRouter(prefix="/api/event", tags=["事件识别与要素抽取"])

# 全局单例（首次调用时加载事件体系）
_extractor: Optional[EventExtractor] = None
_schema: Optional[EventSchema] = None


def get_extractor() -> EventExtractor:
    global _extractor
    if _extractor is None:
        _extractor = EventExtractor()
    return _extractor


def get_schema() -> EventSchema:
    global _schema
    if _schema is None:
        _schema = EventSchema()
    return _schema


# ============================================================
# 请求模型
# ============================================================
class ArticleIn(BaseModel):
    """文章输入（原始舆情数据字段）。"""
    article_file_id: str = Field("", description="文章ID")
    article_title: str = Field(..., description="文章标题")
    content: str = Field(..., description="正文")
    article_source: str = Field("未知", description="来源网站")
    article_publish_time: Optional[str] = Field(None, description="发布时间")


class ExtractRequest(BaseModel):
    article: ArticleIn


class ExtractBatchRequest(BaseModel):
    articles: List[ArticleIn]


# ============================================================
# 接口
# ============================================================
@router.get("/health")
async def health():
    return {"status": "ok", "module": "event_extraction"}


@router.get("/schema")
async def schema_summary():
    """返回标准化事件知识摘要（公司/行业/事件类型数量）。"""
    return get_schema().summary()


@router.post("/extract")
async def extract(req: ExtractRequest):
    """抽取单篇文章的事件与要素。"""
    try:
        return get_extractor().extract(req.article.model_dump())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"抽取失败: {e}")


@router.post("/extract_batch")
async def extract_batch(req: ExtractBatchRequest):
    """批量抽取。"""
    try:
        articles = [a.model_dump() for a in req.articles]
        return get_extractor().extract_batch(articles)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"批量抽取失败: {e}")
