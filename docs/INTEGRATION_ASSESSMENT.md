# 队友交付物对接评估

> 角色 C（系统工程）整理 · 2026-09-03
> 目的：摸清当前仓库里"谁交付了什么、实际覆盖赛题哪个环节、怎么接入主系统"，供全队对齐后做对接决策。
> 结论基于对代码与产物文件的核查，非空谈。

---

## 0. 一句话现状

仓库里目前已经形成主系统 `src/` 作为统一入口：成员 A 的 `event_extract(1)/event_extract` 负责事件抽取，成员 B 的分类与评分算法已移植为 `BEventClassifier + BAlgorithmScorer`，成员 C 的聚类、时间线、图谱、预警和前端负责串联完整链路。

---

## 1. 交付物地图

| # | 交付物 | 作者自称定位 | 代码实际覆盖 | 关键产物 | 接入主系统状态 |
|---|--------|--------------|--------------|----------|----------------|
| 1 | 主系统 `src/`（我们 C） | C：聚类/脉络/系统 | 同源聚类 + 脉络追踪 + 预警 + **移植版 B 分类/评分** | `data/results/event_groups.json`、`alerts.json`（P0-P3） | —（本体） |
| 2 | `event_extract(1)/event_extract/` | A：事件识别与要素抽取 | `event_name`、`event_emotion`、主体、行业、事件时间抽取 | `outputs_event/extracted_*.json` | ✅ 已通过 `AEventAdapter` 接入端到端流水线 |
| 3 | `lyt/event_extract/event_extract/` | 自称"可信评估与影响评分"（文档 §0 图） | **B：谣言/低质量识别（真 GBDT 模型）+ 影响评分公式 + FAISS 向量检索** | `models/*.pkl`、`outputs/ranking.json`、rumor_list | ❌ 零接入，独立 FastAPI（同用 8000 端口，冲突） |
| 4 | `financial_event_radar_share(1)/` | 成员 B 交付 | **可信评估 + 影响评分 + 事件分类资产** | `web/data/dashboard_data.json`、`models/w2v` | ✅ 已移植分类与评分算法；不再依赖其生成好的结果文件 |

> 当前主链路优先消费 `event_extract(1)/event_extract/outputs_event/extracted_unlabeled_test.json`；旧 Excel 入口仅保留为回退基线。

---

## 2. 赛题环节 × 实现来源

| 赛题环节 | 主系统(自产) | lyt/ | radar/ | 状态判断 |
|----------|--------------|------|--------|----------|
| 事件识别/分类（A） | `AEventAdapter` 消费 A 的抽取 JSON；BEventClassifier 可校准事件类别 | ❌ 无 | 可作历史参考 | **A 输出已正式接入；B 分类用于评分前校准** |
| 同源聚合/聚类（C） | ✅ 图聚类 0.94（purity .964） | 简易 faiss 检索 | `deduplication_baseline.py` | 主系统最强，已定稿 |
| 谣言/低质量识别（B·辨伪） | 仅传闻词惩罚（0.07×命中数） | ✅ **GBDT 真模型 AUC .956** | credibility 分量 rumour_penalty | **lyt 唯一有监督辨伪** |
| 可信度融合 | BAlgorithmScorer 按 B 思路移植并输出 0-1 | model.5+source.2+cross.2+evid.1 | dashboard credibility 0-100 | 主系统统一 0-1 输出 |
| 影响分级/评分 | impact 0-1，level high/moderate/low | 0-100 公式 + 弱监督模型(acc .43) | impact 0-100 | 主系统统一 0-1 输出 |
| 脉络/预警（C） | ✅ P0-P3 AlertEngine + timeline | push_reason 文案 | P0-P3（旧语义） | 主系统已统一为 P0 最紧急 |

---

## 3. 评分路径：三条并存（本次评估核心）

| 维度 | 主系统 BAlgorithmScorer | lyt trust_assess | radar dashboard_data |
|------|------------------------|------------------|----------------------|
| 方法 | **手写规则权重**（来源分层 3 级、官方线索词、正则数字密度、materiality 关键词档） | **GBDT 谣言模型 + 透明 0-100 公式** | 规则+分类+模型综合，0-100 分 |
| 是否训练/验证 | 无训练、无指标 | 5 折 CV：**AUC 0.956**（5.2 万无标签，290 谣言 vs 5.1 万正常，正类 0.56%） | 有 reports/，自称有指标 |
| 谣言识别 | 无 rumor_prob，仅扣分 | `rumor_prob` + `is_rumor` + `push_reason` | rumour_penalty 分量 |
| 特征丰富度 | 词表小、无时间/交易时段 | **最全**：来源 5 层 0-5 分、交易时段、标题党、多源前缀计数 | 未深核 |
| 输入依赖 | 事件组（Cluster 产物） | **单篇文章** ArticleIn | 单条告警 |
| 输出量纲 | impact 0-1，level 英文档 | impact **0-100**，level 高中低 | impact 0-100 |
| 预警分级 | P0=最紧急 | 无 P 级（靠 push_reason 分级文案） | P0=最弱（**与主系统倒置**） |
| 对接主系统契约 | 原生 `ScoredEventGroupSchema` | 需适配（0-100→0-1、字段名映射） | 需适配（P 倒置、0-100→0-1、已建 43/532 低匹配） |
| 当前状态 | ✅ 驱动 alerts.json | 独立未接入 | 只读资产，匹配 43/532 |

**判定**：
- 主系统现在以统一契约驱动前端和预警：A 输出进 `EventSchema`，C 聚类成 `EventGroupSchema`，B 分类/评分后由 C 生成 P0-P3。
- lyt 那套仍可作为后续“谣言概率/低质量识别”增强项；它是文章级输出，若接入还需要单独适配。
- `financial_event_radar_share(1)` 的价值已经从“结果文件对齐”转为“算法和模型资产复用”，避免了 cluster_id/group_id 不一致造成的匹配问题。

---

## 4. 契约冲突（需要全队裁决）

1. **角色定义三份表不一致**：README 分工表 / `docs/INTERFACE_CONTRACT.md` / `lyt/INTEGRATION.md` 对 A/B/C 各做什么描述不同（尤其 B：README=评分、lyt 文档里 B=同源聚合）。
2. **P 级语义已统一**：主系统统一为 P0=最紧急、P3=最低优先级；旧结果文件若再导入必须做映射。
3. **量纲已在主系统统一**：impact/credibility 使用 0-1；外部 0-100 输出若接入必须归一化。
4. **EventGroupSchema 契约落后代码**：接口文档 §2.2 未含新增的 evidence_edges/timeline_stages/cluster_reason。
5. **A→C 已有真实入口**：`AEventAdapter` 可直接读取 A 的 `extracted_*.json`；`EventAdapter` 仅作为旧 Excel 回退入口。

---

## 5. 建议（供决策，非定论）

### 短期（1–2 天，可立即做）
1. **统一量纲与 P 语义约定**：写入一份全队共享的《字段规范 v2》，裁定 impact 用 0-1 还是 0-100、P0/P1/P2/P3 语义，所有人对齐后再动代码。
2. **保留主系统 BAlgorithmScorer + AlertEngine 为基座**（已兼容前端），但**在文档里如实标注它是"本地规则实现、未验证"，不是任何队友的模型输出**——答辩口径不能把它说成 B 队友的成果。
3. **完成 A 全量正式运行**：A 的 `event_ner` 已通过 `AEventAdapter` 接入；下一步是在清洗后的 106387 条无标签测试数据上生成全量 JSON，并用 BGE 重建最终事件组。

### 中期（看队友交付与意愿）
4. **接入 lyt 辨伪信号**：给主系统 `credibility` 增加一路 `rumor_prob`（lyt 模型可对任意文本 predict），融合进 b_score_components；这补上了主系统唯一缺的"谣言识别"能力（赛题任务2）。成本：跑通 lyt 依赖 + 一个 0-100→0-1 归一化适配器。
5. **radar 降级为"参考基线"**：其 429 条 dashboard 不直接进主系统；reports 指标（如事件抽取 acc 82%）作为 A/B 环节的答辩证据引用，w2v 模型已复用。

### 必须全队确认的 3 件事
- [ ] **lyt 目录作者到底做哪个角色**（代码是评分，不是目录名的事件抽取）？
- [ ] **radar 包的作者**是谁、是否就是正式 A/B 队友、是否愿意按统一契约出接口？
- [ ] 以哪份角色定义文档为准（建议以 `docs/INTERFACE_CONTRACT.md` + 本表裁决后的 v2 字段规范为唯一基准）？
