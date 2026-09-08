"""Adapter for member-A event extraction outputs.

Member A produces JSON records with extracted event fields.  This adapter maps
those records into the shared EventSchema consumed by clustering, scoring and
frontend APIs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.api.schemas.event import EventSchema
from src.services.event_adapter import EventAdapter


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _parse_time(value: Any) -> datetime | None:
    text = _safe_str(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed.to_pydatetime() if pd.notna(parsed) else None


def _first_subject(subjects: Any) -> dict[str, Any]:
    if not isinstance(subjects, list):
        return {}
    for item in subjects:
        if isinstance(item, dict) and _safe_str(item.get("trading_code")):
            return item
    for item in subjects:
        if isinstance(item, dict):
            return item
    return {}


def _event_entity(record: dict[str, Any], subject: dict[str, Any]) -> str | None:
    subject_name = _safe_str(subject.get("secu_abbr"))
    if subject_name:
        return subject_name
    impact_objects = record.get("impact_objects")
    if isinstance(impact_objects, list):
        names = [_safe_str(item.get("name")) for item in impact_objects if isinstance(item, dict)]
        names = [name for name in names if name]
        if names:
            return "、".join(names[:3])
    return None


def _fallback_content(record: dict[str, Any], subject: dict[str, Any]) -> str:
    parts = [
        _safe_str(record.get("article_title")),
        _safe_str(record.get("event_name")),
        _safe_str(record.get("impact_direction") or record.get("event_emotion")),
        _safe_str(subject.get("secu_abbr")),
    ]
    industry = record.get("industry")
    if isinstance(industry, dict):
        parts.append(_safe_str(industry.get("induname")))
    impact_objects = record.get("impact_objects")
    if isinstance(impact_objects, list):
        parts.extend(
            _safe_str(item.get("name"))
            for item in impact_objects
            if isinstance(item, dict)
        )
    return " ".join(part for part in parts if part)


class AEventAdapter:
    """Convert member-A extracted JSON records into EventSchema records."""

    @staticmethod
    def from_record(record: dict[str, Any]) -> EventSchema:
        subject = _first_subject(record.get("subjects"))
        industry = record.get("industry") if isinstance(record.get("industry"), dict) else {}
        content = _safe_str(record.get("content")) or _fallback_content(record, subject)

        return EventSchema(
            article_file_id=_safe_str(record.get("article_file_id")) or "",
            article_title=_safe_str(record.get("article_title")) or "",
            article_publish_time=_parse_time(
                record.get("article_publish_time") or record.get("publish_time")
            ),
            article_source=_safe_str(record.get("article_source")) or "",
            content=content,
            trading_code=_safe_str(subject.get("trading_code")),
            secu_abbr=_safe_str(subject.get("secu_abbr")),
            industry_code=_safe_str(industry.get("inducode") or industry.get("industry_code")),
            industry_name=_safe_str(industry.get("induname") or industry.get("industry_name")),
            event_name=_safe_str(record.get("event_name")),
            event_time=_parse_time(record.get("event_time")),
            event_entity=_event_entity(record, subject),
            event_impact_direction=_safe_str(
                record.get("impact_direction") or record.get("event_emotion")
            ),
        )

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> list[EventSchema]:
        return [cls.from_record(record) for record in records]

    @staticmethod
    def load_records(path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("events", [])
        return records

    @classmethod
    def from_json(cls, path: str | Path) -> list[EventSchema]:
        return cls.from_records(cls.load_records(path))

    to_jsonl = staticmethod(EventAdapter.to_jsonl)
    read_jsonl = staticmethod(EventAdapter.read_jsonl)
