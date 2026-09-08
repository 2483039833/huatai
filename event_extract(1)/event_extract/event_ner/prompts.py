# -*- coding: utf-8 -*-
"""
Prompt 模板 (Prompt Engineering)
=================================
对应技术路线「Prompt Engineering」。

设计：两段式
  1. 系统提示：角色、任务、输出 JSON 格式、情感定义
  2. 用户提示：候选事件类型（含定义，缩小范围）+ 文章正文

输出要求严格的 JSON，便于后处理解析与标准化。
"""

from __future__ import annotations

from typing import Dict, List, Optional

SYSTEM_PROMPT = """你是金融舆情事件识别与要素抽取专家。任务：
1. 从给定新闻中识别最匹配的事件类型（必须从"候选事件类型"里选一个）；
2. 抽取事件要素：主体（上市公司）、事件时间、影响对象、影响方向；
3. 若文本中要素缺失，基于上下文补全并标注"补全"。

情感（影响方向）只允许三个值：正面 / 中性 / 负面。

只输出一个 JSON 对象，不要输出任何解释、Markdown 代码块或其他文字。JSON 字段如下：
{
  "event_name": "从候选事件类型中选一个",
  "event_emotion": "正面/中性/负面",
  "subjects": [{"secu_abbr": "证券简称", "trading_code": "股票代码或空字符串"}],
  "impact_objects": [{"name": "影响对象名称", "type": "公司/行业/产品/其他"}],
  "event_time": "YYYY-MM-DD 或空字符串",
  "impact_direction": "正面/中性/负面",
  "industry": {"induname": "行业名称或空字符串", "inducode": "行业代码或空字符串"},
  "confidence": 0.0到1.0之间的小数,
  "reason": "一句话说明判断依据"
}"""


def build_user_prompt(
    article: Dict,
    candidate_events: Optional[List[str]] = None,
    event_definitions: Optional[Dict[str, str]] = None,
    candidate_companies: Optional[List[str]] = None,
) -> str:
    """
    构造用户提示。

    Parameters
    ----------
    article           : {"article_title", "content", "article_source", ...}
    candidate_events  : 候选事件类型列表（缩小范围）
    event_definitions : {事件名: 定义}
    candidate_companies : 预检测到的公司简称列表（辅助主体抽取）
    """
    parts = []

    # 1. 候选事件类型 + 定义
    if candidate_events and event_definitions:
        lines = ["## 候选事件类型（只能从以下选一个）"]
        for ev in candidate_events:
            desc = event_definitions.get(ev, "")
            # 定义可能很长，截断到 120 字
            desc = desc[:120] + "……" if len(desc) > 120 else desc
            lines.append(f"- {ev}：{desc}")
        parts.append("\n".join(lines))
    else:
        parts.append("## 候选事件类型：未提供，请根据文章内容自行判断一个金融事件类型")

    # 2. 预检测主体（辅助）
    if candidate_companies:
        parts.append("\n## 预检测到的公司（主体抽取时优先考虑）\n" + "、".join(candidate_companies))

    # 3. 文章
    title = article.get("article_title", "")
    content = article.get("content", "")
    source = article.get("article_source", "")
    publish_time = article.get("article_publish_time", "")
    parts.append(f"## 新闻标题\n{title}")
    if publish_time:
        parts.append(f"## 发布时间\n{publish_time}")
    if source:
        parts.append(f"## 来源\n{source}")
    parts.append(f"## 正文\n{content}")

    parts.append("\n请输出 JSON：")
    return "\n".join(parts)


def build_messages(
    article: Dict,
    candidate_events: Optional[List[str]] = None,
    event_definitions: Optional[Dict[str, str]] = None,
    candidate_companies: Optional[List[str]] = None,
) -> List[Dict]:
    """构造完整 messages（system + user）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                article,
                candidate_events=candidate_events,
                event_definitions=event_definitions,
                candidate_companies=candidate_companies,
            ),
        },
    ]
