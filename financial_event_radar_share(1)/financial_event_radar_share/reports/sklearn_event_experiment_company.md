# company 事件分类 SVM 实验

方法：字符 TF-IDF（2—5 gram）+ 类别平衡 LinearSVC + 事件体系候选约束。

| 模型 | 验证 Accuracy | 验证 Macro-F1 | 提供测试集 Accuracy（仅诊断） | 提供测试集 Macro-F1（仅诊断） |
| --- | ---: | ---: | ---: | ---: |
| title_char_svm | 77.95% | 70.05% | 78.40% | 73.41% |
| headline_lead_char_svm | 79.19% | 74.94% | 81.94% | 76.58% |

验证集选中：`headline_lead_char_svm`。