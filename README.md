

## 上市公司事件识别、聚类与智能预警系统

### 项目结构

```
├── data/                    # 数据目录
│   ├── processed/           # 处理后数据
│   ├── embeddings/          # 向量缓存
│   └── results/             # 输出结果
├── cleaned_data/            # A 清洗后的正式输入数据
├── event_extract(1)/        # A：事件识别与要素抽取
├── financial_event_radar_share(1)/ # B：算法、模型与实验资产
├── src/                     # 核心代码
│   ├── config/              # 配置管理
│   ├── preprocessing/       # 数据预处理
│   ├── features/            # 特征工程（embedding）
│   ├── clustering/          # 同源事件聚类 
│   ├── timeline/            # 时间线构建
│   ├── graph/               # 关系图谱
│   ├── api/                 # FastAPI 后端
│   ├── streaming/           # Kafka 流处理
│   └── utils/               # 工具函数
├── frontend/                # Vue 3 前端
├── docker/                  # Docker 部署
├── notebooks/               # 探索性分析
└── tests/                   # 单元测试
```

### 快速开始

```bash
# 1. 安装依赖（当前已验证运行时为 Windows Python 3.11）
python -m pip install -r requirements.txt

# 2. 校验 A 清洗后的数据
python -X utf8 scripts/audit_data_quality.py

# 3. 运行已抽取事件的端到端链路
python -X utf8 scripts/pipeline_end_to_end.py --source a --embedding-backend bge

# 4. 启动 API
python -X utf8 -m uvicorn src.api.main:app --reload

# 5. 启动前端（新终端）
cd frontend && npm install && npm run dev
```

### 分工

| 角色         | 模块                         | 技术栈                                   |
| ------------ | ---------------------------- | ---------------------------------------- |
| A (NLP/LLM)  | 数据清洗、事件识别与要素抽取 | 兼容 OpenAI API 的 LLM、规则回退、标准化 |
| B (ML)       | 事件分类、可信度与影响力评估 | 字符 NB、标题近邻、Word2Vec、透明评分    |
| C (系统工程) | 聚类、时间线、预警与系统实现 | BGE、FAISS、图聚类、FastAPI、Vue         |

当前数据和模型的正式化状态见 [docs/DATA_AND_MODEL_STATUS.md](docs/DATA_AND_MODEL_STATUS.md)。

## 基线实验成果（2026-07-28）

技术路线：BGE embedding → FAISS 检索 → 相似度图 + 连通分量聚类

| 指标                          | 结果  |
| ----------------------------- | ----- |
| 转载去重 Recall@1             | 97.7% |
| 转载去重 F1（阈值0.80）       | 0.996 |
| 端到端聚类 purity（阈值0.94） | 0.964 |
| 端到端聚类 ARI                | 0.791 |
| 端到端聚类 NMI                | 0.989 |
| 同源事件组检测率              | 97.6% |

详细报告见 [docs/BASELINE_REPORT.md](docs/BASELINE_REPORT.md)

> 模型加载说明：本机网络无法直连 HuggingFace，BGE 模型通过 ModelScope 下载并缓存，`EmbeddingEngine` 会自动检测使用本地缓存。

> 环境说明：`ml2026` 当前解释器存在原生段错误，重建后可恢复为统一 WSL 环境；故障与处置记录见数据与模型状态文档。
