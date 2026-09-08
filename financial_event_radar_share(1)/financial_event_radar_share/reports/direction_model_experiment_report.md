# 影响方向模型交叉验证

选择规则：2 折分层交叉验证 Macro-F1 优先，其次 Accuracy。

## industry

| 模型 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| character_nb | 84.16% | 53.21% |
| idf_centroid | 69.61% | 42.33% |
| keyword_rules | 55.21% | 34.34% |
| voting_ensemble | 78.24% | 48.83% |

选中：`character_nb`。
