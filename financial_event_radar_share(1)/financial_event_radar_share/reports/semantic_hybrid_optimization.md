# 语义检索融合优化报告

## 结论

个股事件识别已从“同公司历史标题字符近邻”升级为“同公司历史标题的字符与语义混合检索”。模型选择仅依据有标签训练集的分层验证集，带标签留出集只用于最终诊断。

| 指标 | 原个股模型 | 语义融合模型 | 变化 |
| --- | ---: | ---: | ---: |
| 验证集 Accuracy | 81.06% | 83.85% | +2.79 个百分点 |
| 验证集 Macro-F1 | 76.00% | 79.75% | +3.75 个百分点 |
| 留出集 Accuracy（诊断） | 79.32% | 82.33% | +3.01 个百分点 |
| 留出集 Macro-F1（诊断） | 77.26% | 80.09% | +2.83 个百分点 |

行业事件分类不做替换，继续使用表现更稳定的字符 n-gram 朴素贝叶斯，留出集 Macro-F1 为 96.82%。

## 方法

1. 使用 1,525 条有标签个股训练新闻和 199,999 条无标签训练新闻的标题，训练 96 维字符 Word2Vec；不读取带标签留出集。
2. 对每篇待预测新闻，仅在该公司历史有标签标题中检索，并受事件体系候选标签约束。
3. 候选事件得分为 `0.1 × 字符重叠相似度 + 0.9 × 标题语义余弦相似度`。
4. 在验证集对融合权重搜索后选择上述权重；未通过验证的 SVM、TextCNN 和其他融合方案均未接入主流程。
5. 以预测第一、第二候选的边际分数校准置信度；边际分数低于 0.01 的个股事件进入人工复核标记。

## 置信度校准（验证集）

| 边际分数下限 | 覆盖率 | 对应准确率 |
| --- | ---: | ---: |
| 0.005 | 95.03% | 85.95% |
| 0.010 | 87.27% | 89.32% |
| 0.020 | 80.75% | 91.15% |
| 0.030 | 76.40% | 92.68% |
| 0.050 | 71.12% | 94.32% |
| 0.080 | 66.77% | 95.35% |

## 已更新的交付物

- `data/derived/event_predictions_test_company.jsonl.gz`：升级后的个股事件标签。
- `data/derived/event_predictions_duplicates_test_company.jsonl.gz`：供时间线使用的升级预测。
- `reports/event_extraction_baseline_report.md`：当前总评估结果。
- `web/data/dashboard_data.json`：已刷新为最新告警看板数据。

## 复现

在 `event_radar` 目录使用已有 Anaconda 解释器：

```powershell
D:\myanaconda\python.exe scripts\gensim_semantic_retrieval_experiment.py --train-epoch
# 重复三次
D:\myanaconda\python.exe scripts\gensim_semantic_retrieval_experiment.py --evaluate
D:\myanaconda\python.exe scripts\run_semantic_hybrid_pipeline.py
python scripts\credibility_baseline.py
python scripts\timeline_baseline.py
python scripts\alerting_baseline.py
python scripts\build_dashboard_data.py
```
