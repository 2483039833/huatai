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
1. 从给定资讯中识别最匹配的事件类型（必须从"候选事件类型"里选一个）；
2. 抽取事件要素：主体（上市公司）、事件时间、影响对象、影响方向；
3. 若文本中要素缺失，基于上下文补全。

【多源理解】资讯可能来自新闻、公告、研报、政策文件等不同来源，请先判断来源类型并据此理解语义：
- 公告/研报：以事实性披露为准，优先采用确定性表述；
- 政策文件：关注政策对行业/公司的影响方向；
- 新闻：注意区分标题党、转载噪音、传闻与事实，只抽取可核实的事实性事件，忽略纯炒作、谣言式内容。

【事件判定】事件类型可能语义相近，请优先抓住正文的"核心动作"来区分，例如：核心动作是"发布新车型/公布售价"→新车发布定价；核心动作是"研发投入/技术指标/专利/测试验证"→技术研发进展；核心动作是"新建产线/扩产/产能"→产能布局调整；核心动作是"并购/增发/回购/上市"→资本市场运作。若正文同时出现多个动作，选最核心、影响最大的一件。

【指代消解】把"该公司/该集团/该行/其/前者/后者"等指代还原为具体主体；优先在正文中寻找上市公司证券简称，找不到则留空，不要把指代词本身当作主体。

【时间标准化】若文中出现"今日/昨日/本周/上周/上月"等相对时间，请结合给定的"发布时间"换算成 YYYY-MM-DD；无法确定则留空字符串。

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
    source_type: Optional[str] = None,
    exemplars: Optional[Dict[str, Dict]] = None,
) -> str:
    """
    构造用户提示。

    Parameters
    ----------
    article           : {"article_title", "content", "article_source", ...}
    candidate_events  : 候选事件类型列表（缩小范围）
    event_definitions : {事件名: 定义}
    candidate_companies : 预检测到的公司简称列表（辅助主体抽取）
    source_type       : 来源类型（新闻/公告/研报/政策文件），辅助多源理解
    exemplars         : {事件名: {"title","snippet"}}，少样本示例（可选）
    """
    parts = []

    # 1. 候选事件类型 + 完整定义 + 少样本示例
    if candidate_events and event_definitions:
        lines = ["## 候选事件类型（只能从以下选一个）"]
        for ev in candidate_events:
            desc = event_definitions.get(ev, "")
            # 定义较长时截断到 500 字（当前最长约 470 字，基本保留完整定义与示例）
            desc = desc[:500] + "……" if len(desc) > 500 else desc
            lines.append(f"- {ev}：{desc}")
            ex = (exemplars or {}).get(ev)
            if ex:
                lines.append(f"  示例：{ex.get('title','')}｜{ex.get('snippet','')}")
        parts.append("\n".join(lines))
    else:
        parts.append("## 候选事件类型：未提供，请根据文章内容自行判断一个金融事件类型")

    # 2. 来源类型（多源统一理解的辅助信息）
    if source_type:
        parts.append(f"\n## 来源类型\n{source_type}")

    # 3. 预检测主体（辅助）
    if candidate_companies:
        parts.append("\n## 预检测到的公司（主体抽取时优先考虑）\n" + "、".join(candidate_companies))

    # 4. 文章
    title = article.get("article_title", "")
    content = article.get("content", "")
    source = article.get("article_source", "")
    publish_time = article.get("article_publish_time", "")
    parts.append(f"## 资讯标题\n{title}")
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
    source_type: Optional[str] = None,
    exemplars: Optional[Dict[str, Dict]] = None,
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
                source_type=source_type,
                exemplars=exemplars,
            ),
        },
    ]
