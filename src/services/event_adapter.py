"""
EventAdapter — legacy Excel-to-EventSchema adapter

将原始标注 DataFrame（Sheet1/Sheet3）转换为接口契约 EventSchema，
产出/读取 `data/processed/events.jsonl`（A→C 的联调落地格式）。

真实 A 抽取结果由 AEventAdapter 消费；本适配器保留为基线/回退入口。
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.api.schemas.event import EventSchema


def _safe_str(v) -> Optional[str]:
    """NaN/NaT/None → None，否则字符串"""
    if v is None or pd.isna(v):
        return None
    return str(v)


def _parse_time(v) -> Optional[datetime]:
    """解析 '2026-01-02 10:47:42' 等格式，失败返回 None"""
    if v is None or pd.isna(v):
        return None
    ts = pd.to_datetime(v, errors="coerce")
    return ts.to_pydatetime() if pd.notna(ts) else None


class EventAdapter:
    """原始标注 DataFrame ↔ 契约 EventSchema 的回退适配器"""

    # Sheet1（原文）有完整列；Sheet3（转载）仅有基础列，缺列自动容错
    COLUMN_MAP = {
        "article_file_id": "article_file_id",
        "article_title": "article_title",
        "article_publish_time": "article_publish_time",
        "article_source": "article_source",
        "content": "content",
        "trading_code": "trading_code",
        "secu_abbr": "secu_abbr",
        "industry_code": "industry_code",
        "industry_name": "industry_name",
        "event_name": "event_name",
        "event_emotion": "event_impact_direction",  # 原始"正面/负面" → 契约字段
    }

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list[EventSchema]:
        """
        将原始标注 DataFrame 转为 EventSchema 列表
        """
        events: list[EventSchema] = []
        for _, row in df.iterrows():
            def g(col: str):
                return row[col] if col in df.columns else None

            events.append(EventSchema(
                article_file_id=str(g("article_file_id")),
                article_title=_safe_str(g("article_title")) or "",
                article_publish_time=_parse_time(g("article_publish_time")),
                article_source=_safe_str(g("article_source")) or "",
                content=_safe_str(g("content")) or "",
                trading_code=_safe_str(g("trading_code")),
                secu_abbr=_safe_str(g("secu_abbr")),
                industry_code=_safe_str(g("industry_code")),
                industry_name=_safe_str(g("industry_name")),
                event_name=_safe_str(g("event_name")),
                event_time=None,   # A 抽取字段，标注数据未提供
                event_entity=None,
                event_impact_direction=_safe_str(g("event_emotion")),
            ))
        return events

    # ── JSONL 落地（A→C 联调格式） ──

    @staticmethod
    def to_jsonl(events: list[EventSchema], path: str | Path) -> int:
        """写入 JSONL，返回条数"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(e.model_dump_json() + "\n")
        return len(events)

    @staticmethod
    def read_jsonl(path: str | Path) -> list[EventSchema]:
        """读回 JSONL 并做 pydantic 校验"""
        events: list[EventSchema] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(EventSchema.model_validate_json(line))
        return events
