# 多源事件识别与要素抽取（赛题二 · 任务1）

> 本模块负责赛题二 **任务1：多源金融事件识别与要素抽取**。
> 对应分工：① 数据清洗与标注处理；② 基于 LLM 的事件识别；③ 抽取主体/时间/影响对象/影响方向；④ 跨文本补全与标准化。
>
> 📖 配套文档：
> - **`操作指南.md`** —— 你自己看，逐步操作手册
> - **`接口对接文档.md`** —— 发给队友看，接口说明与接入方式

## 一、目录结构

```
event_extract/
├── event_ner/                     # ★ 核心模块
│   ├── paths.py                   #   数据路径自动探测
│   ├── schema.py                  #   标准化事件知识库（加载事件体系 JSON）
│   ├── llm_client.py              #   LLM 客户端（Qwen/DeepSeek/GLM，OpenAI 兼容）
│   ├── prompts.py                 #   Prompt 模板
│   ├── extractor.py               #   事件识别 + 要素抽取 + 补全 + 标准化
│   ├── postprocess.py             #   输出后处理与标准化
│   ├── prepare_annotation.py      #   标注数据处理（转 LLM 微调 / NER 格式）
│   ├── evaluate.py                #   模型评测报告
│   ├── batch.py                   #   无标签批量抽取（混合策略+续跑+并行）
│   └── run.py                     #   一键运行入口
├── api_event.py                   # 对外对接接口（FastAPI）
├── main.py                        # 服务入口
├── clean_data.py                  # 数据清洗
├── structure.py                   # 数据探查
├── monitor.py                     # 批量进度监控（Windows 友好）
├── run_full.sh                    # 全量抽取脚本（测试集→训练集）
├── config.example.json            # LLM 配置示例
├── config.json                    # LLM 配置（含密钥，勿提交）
├── outputs_event/                 # 运行输出（JSONL / 评测报告 / 知识库）
└── requirements.txt
```

> 数据（`data/`、`cleaned_data/`）为团队共享，位于**父目录**；模块会自动探测父目录与当前目录。

## 二、快速开始

### 1. 安装依赖
```bash
cd event_extract
pip install -r requirements.txt
```

### 2. 配置 LLM（可选，不配置则走规则回退）
```bash
cp config.example.json config.json
# 编辑 config.json，填入 Qwen / DeepSeek / GLM 的 api_key 和 base_url
```

支持的厂商预设（OpenAI 兼容接口）：
| provider | base_url | 默认模型 |
|----------|----------|----------|
| qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-plus |
| deepseek | `https://api.deepseek.com/v1` | deepseek-chat |
| glm | `https://open.bigmodel.cn/api/paas/v4` | glm-4-flash |

### 3. 运行

```bash
# ① 事件识别 + 要素抽取（带标签测试集，同时输出评测报告）
python -m event_ner.run

# ② 对无标签数据批量抽取（混合策略 + 断点续跑 + 并行，输出 JSONL）
python -m event_ner.run --source unlabeled_test --workers 8
#    其他参数：--limit N（前N条）/ --rule-only（纯规则）/ --redo-rule（补抽规则兜底）
#    进度监控：python monitor.py

# ③ 导出标准化事件知识
python -m event_ner.run --export-schema

# ④ 标注数据处理（转 LLM 微调 JSONL + NER BIO）
python -m event_ner.run --prepare-annotation
```

### 4. 启动 Web 服务（供其他组员对接）
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
接口文档：`http://localhost:8000/docs`

## 三、输出（最终产出）

| 产出 | 文件 | 说明 |
|------|------|------|
| ② 事件结构化 JSON | 带标签 `outputs_event/extracted_*.json`；无标签 `outputs_event/extracted_*.jsonl` | 每篇文章的事件+要素结构化结果 |
| ③ 标准化事件知识 | `outputs_event/event_knowledge.json` | 公司/行业/事件类型/定义 |
| ④ 模型评测报告 | `outputs_event/evaluation_report.json` + `evaluation_errors.csv` | 准确率 + 错误明细 |
| 微调标注数据 | `outputs_event/llm_sft.jsonl` + `ner_bio.jsonl` | LLM 微调 / NER 训练 |

## 四、对外对接接口

其他组员通过 HTTP 接口调用本模块：

```bash
# 单篇抽取
curl -X POST http://localhost:8000/api/event/extract \
  -H "Content-Type: application/json" \
  -d '{
    "article": {
      "article_file_id": "NEWS_MULTISOURCE_D-1",
      "article_title": "问界2025年累计交付新车突破42万辆",
      "content": "赛力斯公告显示，问界2025全年累计交付突破42万辆。",
      "article_source": "上海证券报·中国证券网",
      "article_publish_time": "2026-01-02 10:47:42"
    }
  }'
```

输出 JSON 字段（下游模块直接消费）：

| 字段 | 含义 | 下游用途 |
|------|------|----------|
| `event_name` | 标准化事件类型 | 影响评分（严重程度） |
| `event_emotion` / `impact_direction` | 情感/影响方向（正面/中性/负面） | 影响评分 |
| `source_type` | 来源类型（新闻/公告/研报/政策文件） | 多源理解辅助 |
| `subjects` | 主体（证券简称+代码） | 影响对象 |
| `impact_objects` | 影响对象 | 预警推送 |
| `industry` | 行业 | 行业事件聚合 |
| `event_time` | 事件时间 | 脉络追踪 |
| `confidence` | 置信度 | 过滤/排序 |

## 五、技术路线说明

1. **LLM 事件识别**：Qwen/DeepSeek/GLM（OpenAI 兼容），两段式 Prompt（先预检测公司缩小候选事件，再让 LLM 分类），见 `llm_client.py` + `prompts.py`。
2. **规则回退**：未配置 LLM 时，用"事件关键词 + 情感关键词 + 公司简称匹配"兜底，保证离线可跑通。
3. **跨文本补全**：主体/时间/影响对象缺失时，用规则预检测结果与正则补全，并在 `completion` 字段记录补全来源。
4. **标准化**：事件类型模糊匹配到事件体系（`schema.py`）、情感三分类、证券简称→代码。
5. **LoRA 微调（可选）**：`prepare_annotation.py` 已生成微调数据，可按需接入。
