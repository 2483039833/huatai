# 金融事件雷达

赛题二“面向上市公司的事件驱动智能识别、可信评估与脉络追踪”项目实现。系统将金融新闻转化为可追溯的事件：识别事件类型和主体、聚合同源报道、评估证据可信度、构建时间线，并生成分级预警。

## 功能链路

```text
新闻数据
  → 事件体系约束与事件识别
  → 同源事件聚合 / 去重
  → 证据可信度评估
  → 事件时间线追踪
  → 影响评分、人工复核与预警看板
```

## 项目结构

| 目录 | 用途 |
| --- | --- |
| `data/processed/` | 标准化后的压缩 JSONL 数据与事件体系 |
| `data/derived/` | 事件预测、聚合、可信度、时间线与告警结果 |
| `models/` | 本地训练的语义向量模型检查点 |
| `scripts/` | 数据处理、训练、验证和全链路运行脚本 |
| `reports/` | 实验结论、验证指标和方法说明 |
| `web/` | 本地预警看板页面与其数据 |

## 环境准备

项目的语义模型使用已有 Anaconda 环境。以下命令均假定当前目录为项目目录：

```powershell
Set-Location D:\huaitai\event_radar
D:\myanaconda\python.exe --version
```

已使用的主要包：`gensim`、`scikit-learn`、`PyTorch`、`numpy`。当前正式方案使用 `gensim` 训练本地标题语义向量，无需下载外部预训练模型。

## 快速运行

### 1. 数据准备

原始输入位于工作区根目录，包括带标签新闻、无标签新闻、公司事件体系和行业事件体系。

```powershell
Set-Location D:\huaitai\event_radar
D:\myanaconda\python.exe scripts\prepare_data.py --include-unlabeled
```

将生成以下主要文件：

- `data/processed/event_taxonomy.jsonl`：120 条公司与行业事件定义；
- `data/processed/news_labeled_*.jsonl.gz`：训练、验证诊断使用的带标签新闻；
- `data/processed/news_unlabeled_*.jsonl.gz`：无标签新闻；
- `data/processed/dataset_manifest.json`：数据字段与样本统计。

### 2. 训练本地语义模型

模型只使用“带标签训练新闻”和“无标签训练新闻”的标题学习语义，不使用带标签留出集进行训练。

```powershell
D:\myanaconda\python.exe scripts\gensim_semantic_retrieval_experiment.py --train-epoch
D:\myanaconda\python.exe scripts\gensim_semantic_retrieval_experiment.py --train-epoch
D:\myanaconda\python.exe scripts\gensim_semantic_retrieval_experiment.py --train-epoch
```

每条命令完成一个训练轮次并保存检查点，因此即使运行时长受限也可以继续训练。首次运行会创建 `models/company_title_char_w2v.model`；已有检查点时会继续训练。

### 3. 运行事件识别验证

```powershell
D:\myanaconda\python.exe scripts\gensim_semantic_retrieval_experiment.py --evaluate
```

验证结果会写入：

- `reports/gensim_semantic_retrieval_experiment.md`
- `reports/gensim_semantic_retrieval_experiment.json`

### 4. 生成正式事件预测

```powershell
D:\myanaconda\python.exe scripts\run_semantic_hybrid_pipeline.py
```

个股事件采用验证集选出的融合策略：

```text
0.1 × 同公司历史标题字符相似度 + 0.9 × 标题语义余弦相似度
```

行业事件保留字符 n-gram 朴素贝叶斯模型。两类事件都受公司/行业事件体系约束，避免预测到不属于该标的的事件类型。

### 5. 运行后续链路

```powershell
D:\myanaconda\python.exe scripts\deduplication_baseline.py
D:\myanaconda\python.exe scripts\credibility_baseline.py
D:\myanaconda\python.exe scripts\timeline_baseline.py
D:\myanaconda\python.exe scripts\alerting_baseline.py
D:\myanaconda\python.exe scripts\build_dashboard_data.py
```

各步骤含义：

1. `deduplication_baseline.py`：发现并聚合相同事件的多源报道。
2. `credibility_baseline.py`：依据来源权威性、官方线索、交叉印证、具体性与传闻风险输出证据可信度。
3. `timeline_baseline.py`：按时间排序，识别报道、跟进、确认、澄清等事件状态。
4. `alerting_baseline.py`：根据影响度、可信度、传播范围、状态和时效性输出 P0–P3 告警。
5. `build_dashboard_data.py`：生成看板使用的 `web/data/dashboard_data.json`。

### 6. 查看看板

```powershell
Set-Location D:\huaitai\event_radar
D:\myanaconda\python.exe -m http.server 8080
```

然后在浏览器打开：<http://localhost:8080/web/>。若服务已经启动，刷新页面即可加载新数据。

## 验证方法

### 模型选择原则

个股事件数据由带标签训练集进行分层切分：训练子集用于拟合，验证子集用于选择模型。优先比较 `Macro-F1`，再比较 Accuracy，避免样本量大的事件类别掩盖长尾类别的效果。

带标签留出集只用于最终诊断，不参与模型权重、阈值或特征选择。

### 当前核心指标

| 任务 | 数据范围 | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| 个股事件识别 | 带标签留出集诊断 | 82.33% | 80.09% |
| 行业事件识别 | 带标签留出集诊断 | 97.52% | 96.82% |
| 同源事件去重 | 带标签测试集诊断 | 97.37% Precision | 91.21% F1 |

个股语义融合相较原“标题字符近邻”模型的留出集诊断结果：

| 模型 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| 原字符近邻 | 79.32% | 77.26% |
| 语义融合（当前） | 82.33% | 80.09% |

详细实验记录见 [语义融合优化报告](reports/semantic_hybrid_optimization.md)、[综合评测与优化报告](reports/evaluation_optimization_report.md) 和 [高级模型尝试报告](reports/advanced_model_attempts.md)。

### 置信度与人工复核

个股分类采用第一、第二候选事件的得分差（边际分数）反映不确定性。边际分数低于 `0.01` 时，预测会标记为 `review_required`，供人工复核。该机制用于提高高优先级推送的查准率，而不是删除低置信度事件。

可信度评分衡量的是“新闻证据充分程度”，不是对新闻事实真假的自动裁定；比赛数据未提供事实真伪标签。

## 结果检查

运行完毕后，可检查以下文件是否存在并已刷新：

```text
data/derived/event_predictions_test_company.jsonl.gz
data/derived/event_predictions_test_industry.jsonl.gz
data/derived/alert_feed_test.jsonl.gz
reports/event_extraction_baseline_report.md
web/data/dashboard_data.json
```

网页服务健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8080/web/ | Select-Object StatusCode
```

返回 `200` 表示看板服务正常。

## 演示材料

- [全类型演示案例库](docs/demo_cases.md)：覆盖 44 个公司事件类别和 20 个行业事件类别；63 个案例来自留出集上的正确预测，1 个类别使用训练样例补充覆盖。
- [答辩演示路线](docs/demo_walkthrough.md)：提供 6–8 分钟的讲解顺序，以及多源去重、可信度等级、时间线状态、P0–P3 预警与人工复核的真实搜索案例。
- `data/derived/demo_cases.json`：案例库的结构化数据，可用于后续制作幻灯片或扩展网页展示。

## 答辩材料

- [赛题二答辩 PPT](deliverables/financial_event_radar_defense.pptx)：14 页可编辑宽屏演示稿，涵盖赛题映射、架构、模型、优化、评测、误差边界和在线演示。
- [逐页答辩讲稿](docs/答辩讲稿.md)：建议 10–12 分钟讲解节奏与每页口播要点。
- `scripts/generate_defense_ppt.ps1`：PPT 的可复现生成脚本；需在安装 Microsoft PowerPoint 的 Windows 环境中运行。

## 注意事项

- 若需要从零开始复现实验，请在新的工作副本中运行，或先妥善保存已有的 `models/company_title_char_w2v.model` 检查点；重复执行训练命令会继续增加训练轮次。
- 建模选择应只依据训练集内部验证结果；不得利用带标签留出集反复调参。
- 告警优先级的业务权重为可解释初始规则，生产环境应由投研与风控人员结合实际业务进一步校准。
