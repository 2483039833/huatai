# 三方接口契约对齐文档

> 角色 C（系统工程）整理 · 2026-08-11
>
> **规则：接口以代码为准** —— 契约定义在 `src/api/schemas/`，任何字段变更先改代码、再同步本文档，避免口头约定。

## 1. 数据流总览

```
[A: 事件识别]                    [C: 同源聚类]                    [B: 可信度/影响力]
 原始新闻 ──EventSchema──▶ 事件库 ──▶ 聚类 ──EventGroupSchema──▶ 评分
    (NLP/LLM抽取)              (BGE+FAISS图聚类)    (ML)          │
                                                                 ▼
                                                           ScoredEventGroupSchema
                                                              (C: 时间线/图谱/预警)
```

| 接口 | 方向 | 定义位置 |
|------|------|----------|
| `EventSchema` | **A → C**（A 的输出） | [src/api/schemas/event.py](src/api/schemas/event.py) |
| `EventGroupSchema` | **C → B**（B 的输入） | [src/api/schemas/cluster.py](src/api/schemas/cluster.py) |
| `ScoredEventGroupSchema` | **B → C**（B 的输出） | [src/api/schemas/scoring.py](src/api/schemas/scoring.py) |

---

## 2. 接口定义与 JSON 示例

### 2.1 EventSchema（A → C）

**A 队友负责填写的字段**：原始新闻字段（A 透传）+ 事件抽取字段（A 新增）。C 的聚类主要依赖 `content`，其余供下游展示与特征。

```json
{
  "article_file_id": "20240101_600519_001",
  "article_title": "贵州茅台2023年度业绩说明会实录",
  "article_publish_time": "2024-01-15T09:30:00",
  "article_source": "新浪财经",
  "content": "贵州茅台发布2023年度业绩说明会公告，全年营收同比增长18.2%……",

  "trading_code": "600519",
  "secu_abbr": "贵州茅台",
  "industry_code": null,
  "industry_name": "白酒",
  "event_name": "业绩预告/快报",
  "event_time": "2024-01-15T09:30:00",
  "event_entity": "贵州茅台",
  "event_impact_direction": "正面"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `article_file_id` | str | ✅ | 原文唯一 ID（与原始 xlsx 一致） |
| `article_title` | str | | 新闻标题 |
| `article_publish_time` | datetime | | 新闻发布时间 |
| `article_source` | str | | 来源媒体 |
| `content` | str | ✅ | **新闻正文（C 聚类主依据）** |
| `trading_code` | str | | 标的代码 |
| `secu_abbr` | str | | 标的简称 |
| `industry_code` | str | | 行业代码 |
| `industry_name` | str | | 行业名称 |
| `event_name` | str | | 事件类型名称（**建议沿用 46 类事件体系**） |
| `event_time` | datetime | | 事件发生时间（区别于发布时间） |
| `event_entity` | str | | 事件主体实体 |
| `event_impact_direction` | str | | 正面 / 负面 / 中性 |

### 2.2 EventGroupSchema（C → B）

**B 队友拿到一组同源事件**，需要基于组内的 `events` 列表 + 组级特征进行可信度/影响力评估。

```json
{
  "group_id": "G000123",
  "events": [
    { "article_file_id": "20240115_600519_001", "article_title": "贵州茅台业绩说明会……", "event_impact_direction": "正面" },
    { "article_file_id": "20240116_600519_005", "article_title": "茅台业绩说明会实录全文……", "event_impact_direction": "正面" }
  ],
  "cluster_confidence": 0.9986,
  "timeline_start": "2024-01-15T09:30:00",
  "timeline_end": "2024-01-16T15:20:00",
  "merged_title": "贵州茅台2023年度业绩说明会",
  "merged_content": "茅台发布2023年度业绩说明会公告，营收同比增长18.2%……（2篇报道合并）",
  "article_count": 2,
  "source_diversity": 0.5,
  "event_name": "业绩预告/快报",
  "trading_codes": ["600519"],
  "secu_names": ["贵州茅台"]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `group_id` | str | 聚类组 ID |
| `events` | list[EventSchema] | 组内全部事件（B 可看原文抽取字段） |
| `cluster_confidence` | float | 聚类置信度（组内平均相似度），0~1 |
| `timeline_start/end` | datetime | 事件组时间跨度 |
| `merged_title/merged_content` | str | C 合并的标题/摘要 |
| `article_count` | int | 报道数（**可用于评估影响力**） |
| `source_diversity` | float | 来源多样性 = 不同来源数/报道数 |
| `event_name` / `trading_codes` / `secu_names` | | 组级事件类型与标的汇总 |

### 2.3 ScoredEventGroupSchema（B → C）

**B 队友在此 schema 上填充评分**，C 基于评分产出 `alert_level`（预警）。

```json
{
  "group_id": "G000123",
  "credibility_score": 0.92,
  "impact_level": "moderate",
  "impact_score": 0.76,
  "sentiment_score": 0.85,
  "alert_level": "orange",
  "alert_reason": "可信度0.92 + 影响力0.76，涉及核心标的，触发橙色预警"
}
```

| 字段 | 类型 | 取值 | 说明 |
|------|------|------|------|
| `credibility_score` | float | 0~1 | 可信度（来源权威性 + 多源一致性 + 语言特征） |
| `impact_level` | str | high / moderate / low | 影响等级 |
| `impact_score` | float | 0~1 | 影响力（报道量、来源广度、标的权重） |
| `sentiment_score` | float | -1~1 | 情绪：>0 正面，<0 负面 |
| `alert_level` | str | red / orange / yellow / none | **C 根据 B 评分 + 规则产出** |
| `alert_reason` | str | | 预警原因（供前端展示） |

---

## 3. 需要队友确认的问题

### 🔴 A 侧（NLP/LLM 事件识别）

1. **`event_name` 来源**：直接用原始标注的 46 类事件体系，还是开放式抽取？建议 **46 类体系**（下游聚类/展示需要离散标签，开放标签会碎片化）。
2. **一条新闻多标的/多事件**：建议输出**多条 EventSchema**（每条一个「标的 × 事件」对），`article_file_id` 相同、抽取字段不同。C 聚类按 content 聚合后再关联。
3. **`event_impact_direction` 与原始 `event_emotion` 的关系**：口径统一为「正面/负面/中性」即可。
4. **A 的输出落地格式与路径**：建议统一写 **JSONL**，字段名与本 schema 完全一致（pydantic 校验可直接过），放 `data/processed/events.jsonl`。

### 🟠 B 侧（ML 可信度/影响力）

5. **`credibility_score` / `impact_score` 的** 0~1 连续值**是否确认？还是分类？建议连续值（C 的预警规则需要阈值化）。
6. **`impact_level` 枚举**：high/moderate/low 是否可调整？
7. **B 的调用方式**：C 是一次性给一批 `EventGroupSchema`（批量打分），还是逐组？建议**批量**（REST 批量接口或 JSONL 落地）。

---

## 4. 数据分片与联调方案

- **大文件**：`news_without_tags_train.xlsx`（223MB）按 **Sheet 拆成公司/行业两大部分**，再由 `article_file_id` 前缀或行号分片；三人都读**同一份副本**，不各自搬运。
- **Phase 4 联调**（本周目标）：C 用 `data/processed/events.jsonl`（A 先手动抽 100 条样例）跑通全链路 → B 用 C 的 `data/results/event_groups.json` 打分 → C 消费评分出预警。**先用文件落地联调，Kafka 后期接入。**

## 5. 通信约定（建议）

| 阶段 | 方式 |
|------|------|
| Phase 4 联调 | JSONL/JSON 文件落地 + REST 调试 |
| 后期 | Kafka 流式（`raw-events` → `clustered-events` 两个 topic，见 `config.yaml`） |
