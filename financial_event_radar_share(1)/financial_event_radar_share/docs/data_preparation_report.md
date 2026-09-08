# 数据准备报告

生成时间：2026-07-27

## 处理结果

| 资源 | 有效记录数 | 输出文件 |
| --- | ---: | --- |
| 无标签训练新闻 | 199,999 | `news_unlabeled_train.jsonl.gz` |
| 无标签测试新闻 | 107,889 | `news_unlabeled_test.jsonl.gz` |
| 个股标注训练新闻 | 1,525 | `news_labeled_train_company.jsonl.gz` |
| 行业标注训练新闻 | 556 | `news_labeled_train_industry.jsonl.gz` |
| 个股标注测试新闻 | 764 | `news_labeled_test_company.jsonl.gz` |
| 行业标注测试新闻 | 282 | `news_labeled_test_industry.jsonl.gz` |
| 训练集重复新闻 | 1,499 | `duplicates_train_*.jsonl.gz` |
| 测试集重复新闻 | 753 | `duplicates_test_*.jsonl.gz` |

新闻库合计 311,015 条，重复新闻评测样本合计 2,252 条。所有输出均为 UTF-8 gzip 压缩 JSONL，可逐行读取，不需要一次性载入内存。

## 事件知识库

- 公司事件定义：100 条，对应 44 个去重后的事件类型；
- 行业事件定义：20 条，对应 20 个事件类型；
- 合计：120 条可检索的“主体—事件—定义”知识记录；
- 输出：`event_taxonomy.jsonl`。

每条新闻均统一为以下字段：`article_id`、`title`、`published_at`、`source`、`content`、`text`、公司/行业实体、事件标签、情感、影响分析、`duplication_id` 与数据集来源。

## 质量检查

- 8 个带标签/重复新闻表均未发现因缺少文章 ID 或正文而跳过的记录；
- 所有带标签新闻均已保留原始事件、情感和影响分析标签；
- Excel 序列日期已转换为 ISO-8601 UTC 时间；
- 正文中的换行、制表符、零宽字符已标准化，原始语义信息未被截断；
- 竞赛说明中的“20 万无标签训练集”与实际文件一致到表头差异：工作簿含 200,000 行，其中 1 行为表头，因此有效新闻为 199,999 条。

## 后续使用方式

1. 将 `event_taxonomy.jsonl` 建入向量检索或关键词检索索引，给事件识别模块提供候选标签和定义；
2. 用 `news_labeled_*.jsonl.gz` 训练、验证事件分类与要素抽取；
3. 用 `duplicates_*.jsonl.gz` 构造“是否同一事件”的二分类/聚类评测集；
4. 将无标签新闻库用于批量推理、向量索引和线上流式输入模拟。
