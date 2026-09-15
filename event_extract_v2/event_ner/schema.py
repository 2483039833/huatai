# -*- coding: utf-8 -*-
"""
标准化事件知识库 (Standardized Event Knowledge)
================================================
对应最终产出③「标准化事件知识」。

从事件体系 JSON 构建：
  1. 公司 → 证券简称/代码 → 事件类型列表（标的维度）
  2. 行业 → 行业代码/名称 → 事件类型列表（行业维度）
  3. 事件类型 → 定义描述（去重合并）

同时提供实体匹配与标准化工具，供事件识别 / 要素抽取 / 跨文本补全使用。
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from .paths import data_dir

# 事件体系 JSON
STOCK_SCHEMA_FILE = "事件类型_标的.json"
INDUSTRY_SCHEMA_FILE = "事件类型_行业.json"


class EventSchema:
    """事件体系知识库。"""

    def __init__(self, schema_dir: Optional[str] = None):
        self.schema_dir = schema_dir or data_dir()
        self._stock_path = os.path.join(self.schema_dir, STOCK_SCHEMA_FILE)
        self._industry_path = os.path.join(self.schema_dir, INDUSTRY_SCHEMA_FILE)

        if not os.path.exists(self._stock_path):
            raise FileNotFoundError(
                f"未找到事件体系文件 {self._stock_path}，请确认数据目录位置。"
            )

        self._load()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def _load(self) -> None:
        # 1. 标的维度（公司）
        self.companies: List[Dict] = []
        self.company_by_code: Dict[str, str] = {}   # trading_code -> secu_abbr
        self.company_by_abbr: Dict[str, str] = {}   # secu_abbr -> trading_code
        self.company_events: Dict[str, List[str]] = {}  # secu_abbr -> [event_name]

        with open(self._stock_path, encoding="utf-8") as f:
            stock_schema = json.load(f)

        for item in stock_schema:
            comp = item.get("company", {})
            code = str(comp.get("trading_code", "")).strip()
            abbr = str(comp.get("secu_abbr", "")).strip()
            if not code or not abbr:
                continue
            events = [e.get("event_name", "") for e in item.get("event_schema", [])]
            events = [e for e in events if e]
            self.companies.append({
                "trading_code": code,
                "secu_abbr": abbr,
                "events": events,
            })
            self.company_by_code[code] = abbr
            self.company_by_abbr[abbr] = code
            self.company_events[abbr] = events

        # 2. 行业维度
        self.industries: List[Dict] = []
        self.industry_by_code: Dict[str, str] = {}
        self.industry_by_name: Dict[str, str] = {}
        self.industry_events: Dict[str, List[str]] = {}

        if os.path.exists(self._industry_path):
            with open(self._industry_path, encoding="utf-8") as f:
                industry_schema = json.load(f)
            for item in industry_schema:
                ind = item.get("industry", {})
                code = str(ind.get("inducode", "")).strip()
                name = str(ind.get("induname", "")).strip()
                if not code or not name:
                    continue
                events = [e.get("event_name", "") for e in item.get("event_schema", [])]
                events = [e for e in events if e]
                self.industries.append({
                    "inducode": code,
                    "induname": name,
                    "events": events,
                })
                self.industry_by_code[code] = name
                self.industry_by_name[name] = code
                self.industry_events[name] = events

        # 3. 事件类型 -> 定义（合并去重）
        self.event_definitions: Dict[str, str] = {}
        for item in stock_schema:
            for e in item.get("event_schema", []):
                name = e.get("event_name", "")
                if name and name not in self.event_definitions:
                    self.event_definitions[name] = e.get("description", "")
        if os.path.exists(self._industry_path):
            with open(self._industry_path, encoding="utf-8") as f:
                industry_schema = json.load(f)
            for item in industry_schema:
                for e in item.get("event_schema", []):
                    name = e.get("event_name", "")
                    if name and name not in self.event_definitions:
                        self.event_definitions[name] = e.get("description", "")

        self.event_names: List[str] = sorted(self.event_definitions.keys())
        self.all_secu_abbr: List[str] = sorted(self.company_by_abbr.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------
    # 事件类型标准化
    # ------------------------------------------------------------------
    def standardize_event_name(self, name: str) -> Tuple[Optional[str], bool]:
        """
        把模型输出的事件类型标准化到事件体系。

        返回 (标准事件名, 是否精确命中)。未命中返回 (None, False)。
        匹配优先级：精确 -> 去除标点/空格后精确 -> 子串包含 -> 关键词重叠。
        """
        if not name:
            return None, False
        raw = name.strip()

        # 1. 精确匹配
        if raw in self.event_definitions:
            return raw, True

        # 2. 归一化（去空格、去括号内容差异）后精确匹配
        norm = re.sub(r"[（(].*?[)）]", "", raw).strip()
        for ev in self.event_names:
            ev_norm = re.sub(r"[（(].*?[)）]", "", ev).strip()
            if norm == ev_norm:
                return ev, True

        # 3. 子串包含（双向）
        for ev in self.event_names:
            if raw in ev or ev in raw:
                return ev, False

        # 4. 关键词重叠（模糊）
        best, best_score = None, 0
        raw_chars = set(raw)
        for ev in self.event_names:
            score = len(raw_chars & set(ev)) / max(len(raw_chars | set(ev)), 1)
            if score > best_score:
                best, best_score = ev, score
        if best_score >= 0.6:
            return best, False
        return None, False

    def get_event_definition(self, name: str) -> str:
        return self.event_definitions.get(name, "")

    # ------------------------------------------------------------------
    # 实体匹配（主体 / 影响对象）
    # ------------------------------------------------------------------
    def find_companies(self, text: str) -> List[Dict]:
        """
        在文本中匹配公司（证券简称）。
        按简称长度降序匹配，避免"长城汽车"与"长城"的歧义（这里简称唯一，无此问题，
        但保留长词优先策略更稳健）。
        """
        if not text:
            return []
        found = []
        for abbr in self.all_secu_abbr:
            if abbr and abbr in text:
                found.append({
                    "trading_code": self.company_by_abbr[abbr],
                    "secu_abbr": abbr,
                })
        return found

    def find_industries(self, text: str) -> List[Dict]:
        if not text:
            return []
        found = []
        for name, code in self.industry_by_name.items():
            if name and name in text:
                found.append({"inducode": code, "induname": name})
        return found

    def get_company_candidate_events(self, secu_abbr: str) -> List[str]:
        """某公司的候选事件类型（供 LLM 缩小范围）。"""
        return self.company_events.get(secu_abbr, [])

    def get_industry_candidate_events(self, induname: str) -> List[str]:
        return self.industry_events.get(induname, [])

    # ------------------------------------------------------------------
    # 情感标准化
    # ------------------------------------------------------------------
    @staticmethod
    def standardize_emotion(emo: str) -> str:
        """标准化情感：正面 / 中性 / 负面。"""
        if not emo:
            return "中性"
        e = emo.strip()
        if e in ("正面", "利好", "积极", "positive", "看多", "正向"):
            return "正面"
        if e in ("负面", "利空", "消极", "negative", "看空", "负向", "风险"):
            return "负面"
        return "中性"

    # ------------------------------------------------------------------
    # 摘要信息
    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        return {
            "n_companies": len(self.companies),
            "n_industries": len(self.industries),
            "n_event_types": len(self.event_names),
            "schema_dir": self.schema_dir,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"<EventSchema 公司={s['n_companies']} 行业={s['n_industries']} "
                f"事件类型={s['n_event_types']}>")
