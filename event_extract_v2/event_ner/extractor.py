# -*- coding: utf-8 -*-
"""
事件识别与要素抽取 (Event Identification & Element Extraction)
===============================================================
核心模块，串联：
  LLM 抽取（主） -> 规则抽取（回退） -> 后处理标准化 -> 跨文本补全

输出结构化 JSON（最终产出②「事件结构化 JSON」）。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .llm_client import LLMClient
from .postprocess import parse_json, standardize_result
from .prompts import build_messages
from .schema import EventSchema

# ============================================================
# 规则抽取用：事件类型关键词（标准化事件知识的一部分）
# ============================================================
EVENT_KEYWORDS: Dict[str, List[str]] = {
    "现金流风险": ["现金流", "债务", "违约", "展期", "兑付", "逾期", "偿债", "信用风险", "资金链"],
    "地缘政治": ["地缘", "制裁", "关税", "冲突", "战争", "贸易战", "出口管制"],
    "发生地缘政治事件": ["地缘政治事件", "制裁", "冲突", "战争"],
    "公共卫生事件": ["疫情", "流感", "猴痘", "尼帕病毒", "新冠", "传染病", "公共卫生"],
    "重大项目投资建设": ["投资", "建设", "开工", "项目", "签约", "投产", "扩产", "基地", "产业园"],
    "资本市场运作": ["重组", "并购", "增发", "定增", "股权", "回购", "上市", "停牌", "复牌", "再融资", "可转债"],
    "签订重要合同": ["合同", "签订", "中标", "订单", "协议"],
    "医药研发突破": ["药品", "获批", "临床", "适应症", "新药", "生物药", "研发", "上市申请"],
    "关键技术突破": ["突破", "核心技术", "攻克", "首创", "领先", "关键"],
    "技术突破与创新": ["技术突破", "创新", "突破", "首创"],
    "钢铁生产核心技术突破": ["钢铁", "高炉", "炼钢", "轧钢", "薄板", "硅钢"],
    "商业航天与低空经济发展": ["航天", "低空", "卫星", "火箭", "飞行汽车", "无人机", "商业航天"],
    "产品技术创新": ["发布", "新品", "车型", "产品", "技术平台", "创新", "上市", "亮相"],
    "光伏新技术迭代升级": ["光伏", "组件", "电池片", "硅片", "TOPCon", "HJT", "钙钛矿"],
    "化学制品科技创新": ["化工", "材料", "催化剂", "合成", "聚合物", "创新"],
    "光伏组件行业技术迭代": ["光伏组件", "组件", "技术迭代", "电池片"],
    "产能布局调整": ["产能", "投产", "扩产", "工厂", "基地", "生产线", "布局"],
    "海外产能进展": ["海外", "出海", "出口", "境外", "全球", "海外工厂"],
    "技术研发进展": ["研发", "技术", "专利", "突破", "进展", "测试", "验证"],
    "新产品升级发布": ["新品", "升级", "发布", "新款", "上市"],
    "新车发布定价": ["新车", "上市", "定价", "发布", "售价", "车型"],
    "新车出海": ["出海", "出口", "海外", "新车"],
    "动力电池产品迭代": ["动力电池", "电池", "固态电池", "磷酸铁锂", "三元锂"],
    "储能电池产品迭代": ["储能", "电池", "储能系统"],
    "充换电基础设施完善": ["充电", "换电", "充电桩", "超充", "基础设施"],
    "公司战略调整": ["战略", "调整", "规划", "转型", "布局"],
    "业务战略调整": ["业务战略", "战略调整", "调整"],
    "公司战略变化及对外投资": ["战略", "对外投资", "投资", "布局"],
    "新业务拓展": ["新业务", "拓展", "布局", "进军", "切入"],
    "业务拓展": ["业务拓展", "拓展", "布局"],
    "重大业务拓展": ["重大", "拓展", "布局", "进军"],
    "全球化市场扩张与品牌布局": ["全球", "海外", "品牌", "扩张", "市场"],
    "国际化布局": ["国际化", "海外", "全球", "布局"],
    "合资品牌战略": ["合资", "品牌", "战略"],
    "零售业务策略调整": ["零售", "门店", "调改", "超市", "商场"],
    "调改措施进展": ["调改", "门店", "改造", "升级"],
    "数字化转型深化": ["数字化", "智能", "转型", "信息化", "数智"],
    "结构升级": ["结构升级", "升级", "转型", "高端化"],
    "企业转型催化": ["转型", "催化", "改革"],
    "To B业务拓展（新能源、机器人等）": ["To B", "新能源", "机器人", "B端", "拓展"],
    "市场竞争加剧": ["竞争", "价格战", "市场份额", "内卷"],
    "市场竞争": ["竞争", "价格战"],
    "行业下游的需求变化": ["需求", "下游", "景气", "变化"],
    "大宗商品景气与价格波动加剧": ["大宗", "价格", "涨价", "降价", "景气", "波动"],
    "电力投资政策": ["电力", "电价", "政策", "电网", "投资"],
    "农业政策变动": ["农业政策", "补贴", "转基因", "种业", "政策"],
    "水产品价格变动": ["水产品", "水产", "价格", "鱼价"],
    "国内创新药支持政策": ["创新药", "政策", "医保", "审批"],
}

# 情感关键词
NEGATIVE_WORDS = [
    "亏损", "下滑", "下降", "减少", "风险", "退市", "处罚", "立案", "调查",
    "暴跌", "违约", "逾期", "展期", "爆雷", "造假", "裁员", "终止", "失败",
    "诉讼", "查封", "冻结", "巨亏", "危机", "警示", "问询", "违规", "减持",
    "减值", "预亏", "预减", "腰斩", "低迷", "疲软", "承压", "负债", "下调",
]
POSITIVE_WORDS = [
    "增长", "突破", "中标", "签约", "新高", "提升", "上涨", "获批", "上市",
    "盈利", "预增", "合作", "发布", "创新", "领先", "交付", "超预期", "扩产",
    "投产", "利好", "扭亏", "增持", "回购", "大涨", "订单", "签约", "达成",
    "突破", "首款", "首创", "第一", "领跑", "刷新",
]


class EventExtractor:
    """事件识别与要素抽取器。"""

    def __init__(
        self,
        schema: Optional[EventSchema] = None,
        llm: Optional[LLMClient] = None,
        use_exemplars: bool = True,
    ):
        self.schema = schema or EventSchema()
        self.llm = llm if llm is not None else LLMClient()
        self._exemplars: Optional[Dict] = None  # 事件类型 -> 训练集示例（懒加载）
        self.use_exemplars = use_exemplars

    # ================================================================
    # 主入口
    # ================================================================
    def extract(self, article: Dict) -> Dict:
        """
        抽取单篇文章的事件与要素（LLM 优先，失败回退规则）。

        article 至少含：article_file_id, article_title, content；
        可选：article_source, article_publish_time。
        """
        return self._extract_full(article, force_rule=False)

    def extract_for_batch(self, article: Dict, rule_only: bool = False) -> Dict:
        """
        批量抽取专用（路线 C 混合策略）。

        - rule_only=True：跳过 LLM，纯规则抽取（最快、免费）；
        - rule_only=False：先规则预筛，无事件信号的文章直接规则免调 LLM，
          有信号的走 LLM（失败再回退规则）。
        """
        if rule_only:
            return self._extract_full(article, force_rule=True)
        text = f"{article.get('article_title','')} {article.get('content','')}"
        if not self.has_event_signal(text):
            return self._extract_full(article, force_rule=True)
        return self._extract_full(article, force_rule=False)

    def _extract_full(self, article: Dict, force_rule: bool = False) -> Dict:
        """完整抽取流程（force_rule=True 时跳过 LLM，直接用规则）。"""
        text = f"{article.get('article_title','')} {article.get('content','')}"

        # 1. 预检测实体（主体/行业）—— 规则，稳定可靠
        companies = self.schema.find_companies(text)
        industries = self.schema.find_industries(text)
        company_abbrs = [c["secu_abbr"] for c in companies]
        industry_names = [i["induname"] for i in industries]

        # 2. 来源类型（多源统一理解：新闻/公告/研报/政策文件）
        source_type = self._detect_source_type(article)

        # 3. 候选事件类型（缩小范围）
        candidate_events = self._candidate_events(companies, industries)

        # 4. LLM 抽取（主）
        raw = None
        if not force_rule:
            raw = self._llm_extract(article, candidate_events, company_abbrs, source_type)
        method = "llm" if raw is not None else "rule"

        # 5. 规则抽取（回退）
        if raw is None:
            raw = self._rule_extract(text, companies, industries, candidate_events)

        # 6. 后处理标准化
        result = standardize_result(raw, self.schema, article)

        # 7. 跨文本补全（主体/时间缺失时用规则结果补全）
        completion_notes = []
        if not result["subjects"] and companies:
            result["subjects"] = companies
            completion_notes.append("主体由规则预检测补全")
        if not result["event_time"]:
            t = self._extract_time(text, publish_time=article.get("article_publish_time", "") or "")
            if t:
                result["event_time"] = t
                completion_notes.append("事件时间由正文正则补全")
        if not result["impact_objects"] and companies:
            result["impact_objects"] = [
                {"name": c["secu_abbr"], "type": "公司"} for c in companies
            ]
            completion_notes.append("影响对象由主体补全")

        # 8. 组装最终输出（事件结构化 JSON）
        return {
            "article_file_id": str(article.get("article_file_id", "")),
            "article_title": str(article.get("article_title", "")),
            "article_source": str(article.get("article_source", "")),
            "source_type": source_type,
            "publish_time": str(article.get("article_publish_time", "") or ""),
            "event_name": result["event_name"],
            "event_emotion": result["event_emotion"],
            "impact_direction": result["impact_direction"],
            "subjects": result["subjects"],
            "impact_objects": result["impact_objects"],
            "industry": result["industry"],
            "event_time": result["event_time"],
            "confidence": result["confidence"],
            "method": method,
            "completion": {
                "notes": completion_notes,
                "subject_completed": bool(completion_notes and any("主体" in n for n in completion_notes)),
                "time_completed": bool(completion_notes and any("时间" in n for n in completion_notes)),
            },
            "standardized": {
                "event_name_exact_match": result["event_name_exact_match"],
                "reason": result["reason"],
            },
        }

    def extract_batch(self, articles: List[Dict]) -> List[Dict]:
        return [self.extract(a) for a in articles]

    def has_event_signal(self, text: str) -> bool:
        """判断文本是否含事件信号（命中公司/行业/事件关键词）。"""
        if not text:
            return False
        if self.schema.find_companies(text):
            return True
        if self.schema.find_industries(text):
            return True
        for kws in EVENT_KEYWORDS.values():
            for kw in kws:
                if kw in text:
                    return True
        return False

    # ================================================================
    # 候选事件
    # ================================================================
    def _candidate_events(self, companies: List[Dict], industries: List[Dict]) -> List[str]:
        events = []
        for c in companies:
            events.extend(self.schema.get_company_candidate_events(c["secu_abbr"]))
        for ind in industries:
            events.extend(self.schema.get_industry_candidate_events(ind["induname"]))
        # 去重保序
        seen, unique = set(), []
        for e in events:
            if e and e not in seen:
                seen.add(e)
                unique.append(e)
        return unique

    # ================================================================
    # 少样本示例（Few-shot exemplars，从训练集懒加载）
    # ================================================================
    def _get_exemplars(self) -> Dict:
        """从带标签训练集为每个事件类型取一个示例，用于少样本提示。"""
        if self._exemplars is not None:
            return self._exemplars
        self._exemplars = {}
        if not self.use_exemplars:
            return self._exemplars
        import os
        import pandas as pd
        from .paths import cleaned_dir
        path = os.path.join(cleaned_dir(), "train_tagged_cleaned.xlsx")
        if os.path.exists(path):
            try:
                df = pd.read_excel(path)
                for _, row in df.iterrows():
                    ev = str(row.get("event_name", "")).strip()
                    if ev and ev not in self._exemplars:
                        self._exemplars[ev] = {
                            "title": str(row.get("article_title", "")).strip()[:60],
                            "snippet": str(row.get("content", "")).strip()[:100],
                        }
            except Exception:
                self._exemplars = {}
        return self._exemplars

    # ================================================================
    # 来源类型识别（多源统一理解）
    # ================================================================
    @staticmethod
    def _detect_source_type(article: Dict) -> str:
        """按标题/正文/来源关键词粗分类来源类型，辅助 LLM 理解语义。"""
        head = " ".join([
            str(article.get("article_title", "")),
            str(article.get("content", ""))[:800],
            str(article.get("article_source", "")),
        ])
        if re.search(r"公告|董事会|股东大会|年报|半年报|季报|招股|说明会|问询函|回复函|定期报告|业绩快报", head):
            return "公告"
        if re.search(r"研报|研究报告|分析师|目标价|评级|盈利预测|首次覆盖|维持买入|给予买入|深度报告", head):
            return "研报"
        if re.search(r"国务院|发改委|财政部|证监会|央行|人民银行|银保监会|部委|指导意见|政策措施|条例|规划纲要", head):
            return "政策文件"
        return "新闻"

    # ================================================================
    # LLM 抽取
    # ================================================================
    def _llm_extract(self, article, candidate_events, company_abbrs, source_type: Optional[str] = None) -> Optional[Dict]:
        if not self.llm.available:
            return None
        # 若规则预检测没命中任何公司/行业，则回退为全量事件类型，
        # 避免 LLM 自行杜撰出不在事件体系中的事件名。
        events = candidate_events if candidate_events else self.schema.event_names
        # 候选较少时启用少样本示例（候选过多时示例会撑爆上下文，仅用定义）
        exemplars = self._get_exemplars() if len(events) <= 10 else None
        messages = build_messages(
            article,
            candidate_events=events,
            event_definitions=self.schema.event_definitions,
            candidate_companies=company_abbrs,
            source_type=source_type,
            exemplars=exemplars,
        )
        reply = self.llm.chat(messages, json_mode=True)
        return parse_json(reply) if reply else None

    # ================================================================
    # 规则抽取（回退）
    # ================================================================
    def _rule_extract(self, text, companies, industries, candidate_events) -> Dict:
        # 事件类型：关键词匹配
        event_name = self._rule_event(text, candidate_events)

        # 情感：正负面关键词计数
        emotion = self._rule_emotion(text)

        return {
            "event_name": event_name,
            "event_emotion": emotion,
            "impact_direction": emotion,
            "subjects": companies,
            "impact_objects": [{"name": c["secu_abbr"], "type": "公司"} for c in companies],
            "industry": industries[0] if industries else {},
            "event_time": self._extract_time(text),
            "confidence": 0.5,
            "reason": "规则回退（未配置 LLM 或 LLM 调用失败）",
        }

    def _rule_event(self, text: str, candidate_events: List[str]) -> str:
        # 只在候选事件里匹配；若候选为空则用全量事件
        pool = candidate_events if candidate_events else list(EVENT_KEYWORDS.keys())
        best_event, best_score = "", 0
        for ev in pool:
            kws = EVENT_KEYWORDS.get(ev, [])
            if not kws:
                continue
            score = sum(1 for kw in kws if kw in text)
            if score > best_score:
                best_event, best_score = ev, score
        return best_event or ""

    def _rule_emotion(self, text: str) -> str:
        neg = sum(1 for w in NEGATIVE_WORDS if w in text)
        pos = sum(1 for w in POSITIVE_WORDS if w in text)
        # 标题权重更高：标题命中计 2 分
        return "负面" if neg > pos else ("正面" if pos > neg else "中性")

    @staticmethod
    def _extract_time(text: str, publish_time: Optional[str] = None) -> str:
        from .postprocess import _extract_time
        return _extract_time(text, publish_time=publish_time)
