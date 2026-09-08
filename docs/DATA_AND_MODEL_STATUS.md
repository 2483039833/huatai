# 数据与模型状态

## 正式链路

```text
A 清洗数据 -> A 事件抽取 -> EventSchema
           -> C BGE/FAISS 图聚类 -> EventGroupSchema
           -> B 事件分类、可信度与影响评分
           -> C P0-P3 预警、时间线、图谱、API、Vue 前端
```

## 三方使用范围

| 成员 | 主系统采用的实现 | 当前边界 |
|---|---|---|
| A | `event_extract(1)/event_extract/event_ner` 的清洗、LLM/规则事件抽取、主体/行业/时间标准化；主系统由 `AEventAdapter` 接收 JSON | 正式输入应为 `cleaned_data/*.xlsx`；已有 1000 条输出只是联调样本 |
| B | `BEventClassifier` 的字符 NB、标题近邻、Word2Vec 语义检索，以及 `BAlgorithmScorer` 的可信度/影响评分 | 公司分类训练只用清洗训练集；测试集只用于评测；评分仍需进一步标定 |
| C | BGE 向量、FAISS、相似度图聚类、增量聚类、时间线、图谱、P0-P3 预警、FastAPI、Vue | 聚类阈值需要在 A 全量抽取结果上重新验证 |

## 数据状态

网盘中的四份清洗文件已经放在项目根目录 `cleaned_data/`：

- `train_tagged_cleaned.xlsx`：1498 条，标签完整，ID 无重复。
- `test_tagged_cleaned.xlsx`：749 条，标签完整，ID 无重复。
- `unlabeled_train_cleaned.xlsx`：197577 条，ID 无重复，正文均不少于 30 字。
- `unlabeled_test_cleaned.xlsx`：106387 条，ID 无重复，正文均不少于 30 字。

原始文件保留在项目根目录，不覆盖。清洗文件适合作为正式输入，但目前只保留了原 Excel 的第一个公司事件工作表；行业事件训练仍需从原始训练工作簿的 `行业新闻` 工作表读取，或由 A 补充对应的行业清洗文件。

`data/processed/events.jsonl` 与 `data/results/*.json` 当前由 1000 条 A 联调输出生成，不是最终提交结果。正式版本必须在全量 A 抽取后重新生成。

## 模型状态

- A：当前是兼容 OpenAI API 的大模型抽取，未配置接口或调用失败时退化为关键词规则。它不是已完成 LoRA 微调的固定模型。正式版本应先在 749 条清洗测试集上报告事件分类、方向、主体和行业指标，再决定是否 LoRA。
- B：当前分类器属于可复现的轻量监督/检索融合模型。主系统使用 `data/models/b_company_title_char_w2v.model` 的无泄漏重训练版本，语料为 1498 条清洗标注训练标题和 197577 条清洗无标签训练标题，官方测试集未参与。749 条独立测试结果为事件准确率 82.24%、宏 F1 78.72%、方向准确率 93.46%、覆盖率 100%、人工复核率 11.35%。旧资产准确率略高 0.67 个百分点，但训练来源无法完全审计，因此不作为正式模型。
- C：BGE `BAAI/bge-large-zh-v1.5` 是预训练向量模型，不需要从头训练。数据或 A 输出变化后必须重新计算 embedding，并在标注的重复/转载关系上重新选择聚类阈值。

## 正式化顺序

1. 运行 `python -X utf8 scripts/audit_data_quality.py`，数据门禁必须全部通过。
2. 用 A 对 `tagged_test` 抽取并做独立评测，不把测试标签用于训练或分类器拟合。
3. 运行 `python -X utf8 scripts/evaluate_b_classifier.py`，取得 B 分类器独立测试指标。
4. 指标达标后，用 A 对无标签全量数据抽取并保留原始 `content`。
5. 重建 `ml2026` 后用统一环境和 BGE 重算向量、聚类、B 评分及预警；重建前使用已验证的 Windows Python 3.11。
6. 保存运行清单、参数、模型版本和评测报告，再将结果视为正式版本。

## 环境说明

`ml2026` 已包含 Torch、BGE 及项目依赖，但当前 Conda Python 3.11 本身出现原生段错误，甚至 `pip` 的纯 Python 正则解析也异常，不能视为可靠运行环境。B Word2Vec 已临时在项目 `.venv`（Python 3.12.3）完成训练和评测；在提交前应重建而不是继续原位修补 `ml2026`，再运行全套测试和模型加载检查。
