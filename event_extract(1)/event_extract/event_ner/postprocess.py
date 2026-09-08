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
from typing import Dict, Optional

from .schema import EventSchema


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

    # 3. 主体标准化（证券简称 -> 代码）
    subjects = []
    for s in raw.get("subjects", []) or []:
        abbr = s.get("secu_abbr", "") if isinstance(s, dict) else str(s)
        abbr = str(abbr).strip()
        if not abbr:
            continue
        code = schema.company_by_abbr.get(abbr, "")
        subjects.append({"secu_abbr": abbr, "trading_code": code})

    # 4. 影响对象
    impact_objects = []
    for o in raw.get("impact_objects", []) or []:
        if isinstance(o, dict):
            name = str(o.get("name", "")).strip()
            otype = str(o.get("type", "其他")).strip()
        else:
            name, otype = str(o).strip(), "其他"
        if name:
            impact_objects.append({"name": name, "type": otype})

    # 5. 行业标准化
    industry = raw.get("industry", {}) or {}
    induname = str(industry.get("induname", "")).strip()
    inducode = str(industry.get("inducode", "")).strip()
    if not inducode and induname in schema.industry_by_name:
        inducode = schema.industry_by_name[induname]
    if not induname and inducode in schema.industry_by_code:
        induname = schema.industry_by_code[inducode]

    # 6. 时间
    event_time = str(raw.get("event_time", "")).strip()
    if not event_time:
        event_time = _extract_time(article.get("content", "") + article.get("article_title", ""))

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


def _extract_time(text: str) -> str:
    """从文本中抽取时间，返回 YYYY-MM-DD 或空字符串。"""
    if not text:
        return ""
    # 2026年1月2日 / 2026-01-02 / 2026/01/02 / 1月1日
    patterns = [
        r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?",
        r"(\d{1,2})月(\d{1,2})日",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            g = m.groups()
            if len(g) == 3:
                return f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}"
            else:
                # 只有月日，缺少年份，返回空（无法确定年份）
                return ""
    return ""
