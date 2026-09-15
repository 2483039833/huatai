# -*- coding: utf-8 -*-
"""
后处理与标准化 (Post-processing & Standardization)
===================================================
对 LLM / 规则抽取的原始结果做：
  1. 稳健的 JSON 解析（剥离 Markdown 代码块、容错）
  2. 事件类型标准化（映射到事件体系）
  3. 情感标准化（正面/中性/负面）
  4. 实体标准化（证券简称 -> 股票代码）
  5. 缺失字段补全（跨文本补全的兜底）
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Dict, Optional

from .schema import EventSchema

# 指代词集合：LLM 可能把指代词误当作主体/影响对象输出，标准化时直接过滤掉
PRONOUN_WORDS = {
    "该公司", "该集团", "该行", "该企业", "该券商", "该银行", "该基金",
    "本公司", "本集团", "其", "前者", "后者", "这家公司", "这家企业",
}


def _is_pronoun(name: str) -> bool:
    """判断是否为指代词（不作为有效主体/影响对象）。"""
    s = str(name or "").strip()
    return s in PRONOUN_WORDS or len(s) == 1 or s in {"它", "他", "她"}


def parse_json(text: str) -> Optional[Dict]:
    """从 LLM 输出中稳健解析 JSON（容忍 Markdown 代码块）。"""
    if not text:
        return None
    t = text.strip()

    # 去掉 ```json ... ``` 包裹
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # 直接解析
    try:
        return json.loads(t)
    except Exception:
        pass

    # 尝试截取第一个 { 到最后一个 }
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except Exception:
            pass
    return None


def standardize_result(raw: Dict, schema: EventSchema, article: Dict) -> Dict:
    """
    标准化一条抽取结果。

    raw     : 模型/规则的原始输出
    schema  : 事件体系
    article : 原始文章（用于补全兜底）
    """
    if not raw:
        raw = {}

    # 1. 事件类型标准化
    raw_event = raw.get("event_name", "")
    std_event, exact = schema.standardize_event_name(raw_event)
    if std_event is None:
        std_event = ""

    # 2. 情感标准化
    emotion = schema.standardize_emotion(raw.get("event_emotion", "") or raw.get("impact_direction", ""))

    # 3. 主体标准化（证券简称 -> 代码），过滤指代词
    subjects = []
    for s in raw.get("subjects", []) or []:
        abbr = s.get("secu_abbr", "") if isinstance(s, dict) else str(s)
        abbr = str(abbr).strip()
        if not abbr or _is_pronoun(abbr):
            continue
        code = schema.company_by_abbr.get(abbr, "")
        subjects.append({"secu_abbr": abbr, "trading_code": code})

    # 4. 影响对象，过滤指代词
    impact_objects = []
    for o in raw.get("impact_objects", []) or []:
        if isinstance(o, dict):
            name = str(o.get("name", "")).strip()
            otype = str(o.get("type", "其他")).strip()
        else:
            name, otype = str(o).strip(), "其他"
        if name and not _is_pronoun(name):
            impact_objects.append({"name": name, "type": otype})

    # 5. 行业标准化
    industry = raw.get("industry", {}) or {}
    induname = str(industry.get("induname", "")).strip()
    inducode = str(industry.get("inducode", "")).strip()
    if not inducode and induname in schema.industry_by_name:
        inducode = schema.industry_by_name[induname]
    if not induname and inducode in schema.industry_by_code:
        induname = schema.industry_by_code[inducode]

    # 6. 时间（含相对时间换算：今日/昨日/上周等，锚定发布时间）
    publish_time = article.get("article_publish_time", "") or ""
    event_time = str(raw.get("event_time", "")).strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", event_time):
        event_time = _extract_time(
            f"{article.get('content', '')} {article.get('article_title', '')} {event_time}",
            publish_time=publish_time,
        )

    # 7. 置信度
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {
        "event_name": std_event,
        "event_name_std": std_event,
        "event_name_exact_match": bool(exact),
        "event_emotion": emotion,
        "impact_direction": emotion,
        "subjects": subjects,
        "impact_objects": impact_objects,
        "industry": {"induname": induname, "inducode": inducode} if (induname or inducode) else None,
        "event_time": event_time,
        "confidence": round(confidence, 4),
        "reason": str(raw.get("reason", "")).strip(),
    }


def _parse_publish_time(publish_time) -> Optional[datetime]:
    """解析发布时间字符串为 datetime（只取年月日部分）。"""
    s = str(publish_time or "").strip()
    m = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def _shift_month(dt: datetime, delta: int) -> datetime:
    """月份平移（近似，保留到 28 号以内避免溢出）。"""
    month = dt.month + delta
    year = dt.year
    while month <= 0:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    day = min(dt.day, 28)
    return datetime(year, month, day)


def _extract_time(text: str, publish_time: Optional[str] = None) -> str:
    """
    从文本中抽取事件时间，返回 YYYY-MM-DD 或空字符串。

    - 绝对日期：2026年1月2日 / 2026-01-02 / 2026/01/02 / 1月1日
    - 相对日期：今日/昨日/前日/本周/上周/上月，锚定 publish_time 换算
    """
    if not text:
        return ""
    text = str(text)

    # 1. 绝对日期（带年份）
    m = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?", text)
    if m:
        try:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        except ValueError:
            pass

    # 2. 只有月日：用发布时间补年份
    m = re.search(r"(\d{1,2})月(\d{1,2})日", text)
    if m:
        base = _parse_publish_time(publish_time)
        if base:
            try:
                return f"{base.year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
            except ValueError:
                pass
        return ""

    # 3. 相对时间：锚定发布时间
    base = _parse_publish_time(publish_time)
    if base is None:
        return ""
    if re.search(r"今日|今天", text):
        return base.strftime("%Y-%m-%d")
    if re.search(r"昨日|昨天", text):
        return (base - timedelta(days=1)).strftime("%Y-%m-%d")
    if re.search(r"前日|前天", text):
        return (base - timedelta(days=2)).strftime("%Y-%m-%d")
    if re.search(r"上周", text):
        return (base - timedelta(days=7)).strftime("%Y-%m-%d")
    if re.search(r"本周", text):
        return base.strftime("%Y-%m-%d")
    if re.search(r"上月|上个月", text):
        return _shift_month(base, -1).strftime("%Y-%m-%d")
    if re.search(r"本月|当月", text):
        return base.strftime("%Y-%m-%d")
    return ""
