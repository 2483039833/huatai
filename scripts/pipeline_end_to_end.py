"""
端到端接口链路演示：A 事件抽取 → C 聚类 → B 分类与评分 → C 预警
=============================================================
验证三方接口契约（docs/INTERFACE_CONTRACT.md）端到端跑通：

    A: 事件抽取输出/原始标注数据 → EventSchema → data/processed/events.jsonl
    C: 图聚类 + GroupBuilder → EventGroupSchema → data/results/event_groups.json
    B: BEventClassifier + BAlgorithmScorer → ScoredEventGroupSchema → data/results/scored_groups.json
    C: AlertEngine → 预警                      → data/results/alerts.json

B 模块接入 B 的事件分类模型，并使用 B 风格可信度/影响评分。

用法：
    PYTHONIOENCODING=utf-8 python scripts/pipeline_end_to_end.py
    PYTHONIOENCODING=utf-8 python scripts/pipeline_end_to_end.py --source a
    PYTHONIOENCODING=utf-8 python scripts/pipeline_end_to_end.py --source excel
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from importlib.util import find_spec

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.loader import DataLoader
from src.clustering.cluster import EventClusterer
from src.features.embedding import EmbeddingEngine
from src.services.a_event_adapter import AEventAdapter
from src.services.event_adapter import EventAdapter
from src.services.group_builder import GroupBuilder
from src.services.b_algorithm_scorer import BAlgorithmScorer
from src.services.alert_engine import AlertEngine

DEFAULT_A_OUTPUT = Path("event_extract(1)") / "event_extract" / "outputs_event" / "extracted_unlabeled_test.json"


def load_embeddings(df1_len: int, df3_len: int) -> np.ndarray:
    """加载/拼接缓存的 embedding（原文+转载）"""
    cache_full = Path("data/embeddings/full_merged.npy")
    cache_s1 = Path("data/embeddings/sheet1_orig.npy")
    cache_s3 = Path("data/embeddings/sheet3_dup.npy")
    if cache_full.exists():
        emb = np.load(str(cache_full))
        print(f"[缓存] 加载 {cache_full}  {emb.shape}")
        return emb
    if cache_s1.exists() and cache_s3.exists():
        emb = np.vstack([np.load(str(cache_s1)), np.load(str(cache_s3))])
        np.save(str(cache_full), emb)
        print(f"[缓存] 拼接 s1+s3 → {emb.shape}")
        return emb
    raise FileNotFoundError(
        "未找到 embedding 缓存。请先运行 scripts/pipeline_full_eval.py 生成缓存。"
    )


def event_text(event) -> str:
    parts = [
        event.article_title,
        event.content,
        event.event_name,
        event.event_impact_direction,
        event.secu_abbr,
        event.industry_name,
    ]
    return " ".join(str(part) for part in parts if part)


def load_or_encode_event_embeddings(events, cache_key: str, force_recompute: bool = False) -> np.ndarray:
    cache_path = Path("data/embeddings") / f"{cache_key}.npy"
    if cache_path.exists() and not force_recompute:
        emb = np.load(str(cache_path))
        if emb.shape[0] == len(events):
            print(f"[缓存] 加载 {cache_path}  {emb.shape}")
            return emb
        print(f"[缓存] {cache_path} 行数 {emb.shape[0]} 与事件数 {len(events)} 不一致，重新计算")

    texts = [event_text(event) for event in events]
    engine = EmbeddingEngine()
    return engine.encode_with_cache(texts, cache_key=cache_key, force_recompute=True)


def load_or_encode_tfidf_embeddings(events) -> np.ndarray:
    texts = [event_text(event) for event in events]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    emb = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-9)
    print(f"[TF-IDF] 生成轻量验证向量 {emb.shape}")
    return emb


def choose_embedding_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if find_spec("torch") and find_spec("sentence_transformers"):
        return "bge"
    return "tfidf"


def load_events_from_a(a_output: Path) -> tuple[list, int]:
    source_records = AEventAdapter.load_records(a_output)
    events = AEventAdapter.from_records(source_records)
    original_content_count = sum(1 for record in source_records if str(record.get("content") or "").strip())
    n_written = AEventAdapter.to_jsonl(events, "data/processed/events.jsonl")
    events_read = AEventAdapter.read_jsonl("data/processed/events.jsonl")
    assert len(events_read) == n_written, "JSONL 往返条数不一致"
    print(f"  读取 A 输出: {a_output}")
    print(f"  A 输出 {n_written} 条 EventSchema | pydantic 校验通过 ✓")
    return events_read, original_content_count


def load_events_from_excel() -> tuple[list, int, int]:
    df1 = DataLoader.load_tagged_company_data()
    df3 = DataLoader.load_company_duplication()
    events = EventAdapter.from_dataframe(df1) + EventAdapter.from_dataframe(df3)
    n_written = EventAdapter.to_jsonl(events, "data/processed/events.jsonl")
    events_read = EventAdapter.read_jsonl("data/processed/events.jsonl")
    assert len(events_read) == n_written, "JSONL 往返条数不一致"
    print(f"  旧数据入口输出 {n_written} 条 EventSchema（原文{len(df1)} + 转载{len(df3)}）")
    print("  pydantic 校验通过 ✓")
    return events_read, len(df1), len(df3)


def save_json(data: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"  写入 {path}  ({len(data)} 条)")


def build_run_manifest(
    *,
    source_type: str,
    source_path: str,
    event_count: int,
    embedding_backend: str,
    with_content: int,
) -> dict:
    warnings = []
    if source_type != "a_extracted_json":
        warnings.append("当前使用旧 Excel 回退入口，未经过 A 的事件抽取。")
    if event_count < 100_000:
        warnings.append("事件数不足 10 万，当前结果按联调/评测样本处理。")
    if embedding_backend != "bge":
        warnings.append("当前未使用正式 BGE 向量，聚类结果仅用于链路验证。")
    if with_content < event_count:
        warnings.append("部分事件没有保留原始正文，可能降低聚类、分类与评分效果。")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "source_path": source_path,
        "event_count": event_count,
        "events_with_content": with_content,
        "embedding_backend": embedding_backend,
        "classifier_training_policy": "official train split only; cleaned company train preferred; official test excluded",
        "production_ready": not warnings,
        "warnings": warnings,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="A→C→B→C 端到端流水线")
    parser.add_argument(
        "--source",
        choices=["auto", "a", "excel"],
        default="auto",
        help="事件入口：auto 优先使用 A 输出，excel 使用旧标注数据入口",
    )
    parser.add_argument(
        "--a-output",
        type=Path,
        default=DEFAULT_A_OUTPUT,
        help="成员 A 事件抽取输出 JSON/JSONL 路径",
    )
    parser.add_argument(
        "--force-embedding",
        action="store_true",
        help="重新计算当前事件入口的 embedding",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "bge", "tfidf"],
        default="auto",
        help="A 输出的向量化方式：auto 优先 BGE，缺少深度学习依赖时用 TF-IDF 验证链路",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    print("=" * 64)
    print("端到端接口链路演示：A 事件抽取 → C 聚类 → B 分类与评分 → C 预警")
    print("=" * 64)

    # ── Step 1: A 模块输出 EventSchema ──
    print("\n[1/5] A: 事件抽取输出 → EventSchema (events.jsonl)")
    use_a = args.source == "a" or (args.source == "auto" and args.a_output.exists())
    if use_a:
        events, original_content_count = load_events_from_a(args.a_output)
        embedding_key = f"a_event_extract_{args.a_output.stem}"
        df1_len = df3_len = None
        source_type = "a_extracted_json"
        source_path = str(args.a_output)
    else:
        events, df1_len, df3_len = load_events_from_excel()
        original_content_count = sum(1 for event in events if event.content)
        embedding_key = None
        source_type = "legacy_excel"
        source_path = str(Path("news_with_tags_train.xlsx"))
    with_polarity = sum(1 for e in events if e.event_impact_direction)
    with_subject = sum(1 for e in events if e.trading_code or e.secu_abbr)
    print(f"  含影响方向字段: {with_polarity} 条 | 含主体字段: {with_subject} 条")

    # ── Step 2: C 聚类 ──
    print("\n[2/5] C: 图聚类 (threshold=0.94)")
    if use_a:
        embedding_backend = choose_embedding_backend(args.embedding_backend)
        print(f"  向量化后端: {embedding_backend}")
        if embedding_backend == "bge":
            emb = load_or_encode_event_embeddings(events, embedding_key, args.force_embedding)
        else:
            emb = load_or_encode_tfidf_embeddings(events)
    else:
        embedding_backend = "bge"
        emb = load_embeddings(df1_len, df3_len)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-9)
    clusterer = EventClusterer(method="graph")
    labels = clusterer.fit_predict(emb)

    # ── Step 3: C → B（构建 EventGroupSchema） ──
    print("\n[3/5] C: 构建 EventGroupSchema → event_groups.json")
    groups = GroupBuilder.build_groups(events, labels, embeddings=emb)
    save_json([g.model_dump(mode="json") for g in groups], "data/results/event_groups.json")
    n_linked = sum(1 for e, l in zip(events, labels) if l != -1)
    print(f"  {len(groups)} 个事件组，覆盖 {n_linked}/{len(events)} 篇文章")

    # ── Step 4: B 分类 + 评分 ──
    print("\n[4/5] B: BEventClassifier + BAlgorithmScorer 评分 → scored_groups.json")
    scored = BAlgorithmScorer.score_batch(groups)
    print(f"  BAlgorithmScorer scored {len(scored)}/{len(groups)} groups")
    save_json([g.model_dump(mode="json") for g in scored], "data/results/scored_groups.json")

    # ── Step 5: C 预警 ──
    print("\n[5/5] C: AlertEngine 预警 → alerts.json")
    final = AlertEngine.apply_batch(scored)
    save_json([g.model_dump(mode="json") for g in final], "data/results/alerts.json")
    manifest = build_run_manifest(
        source_type=source_type,
        source_path=source_path,
        event_count=len(events),
        embedding_backend=embedding_backend,
        with_content=original_content_count,
    )
    Path("data/results/run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "  运行状态: "
        + ("正式结果" if manifest["production_ready"] else "联调/评测结果")
        + " -> data/results/run_manifest.json"
    )

    # ── 效果总结 ──
    print("\n" + "=" * 64)
    print("效果总结")
    print("=" * 64)
    cred = np.array([g.credibility_score or 0 for g in final])
    imp = np.array([g.impact_score or 0 for g in final])
    sent = np.array([g.sentiment_score or 0 for g in final])
    print(f"事件组: {len(final)} 个 | 组均报道 {np.mean([g.article_count for g in final]):.1f} 篇")
    print(f"可信度: mean={cred.mean():.3f}  min={cred.min():.3f}  max={cred.max():.3f}")
    print(f"影响力: mean={imp.mean():.3f}  影响等级: "
          + ", ".join(f"{lvl}={sum(1 for g in final if g.impact_level == lvl)}"
                      for lvl in ["high", "moderate", "low"]))
    print(f"情绪  : 正面={np.sum(sent > 0.3)}, 负面={np.sum(sent < -0.3)}, 中性={np.sum(abs(sent) <= 0.3)}")

    from collections import Counter
    alert_dist = Counter(g.alert_level or "P3" for g in final)
    print("预警分布: " + ", ".join(f"{k}={v}" for k, v in sorted(alert_dist.items())))
    print(f"\n总耗时 {time.time()-t0:.1f}s")
    print("\n输出文件:")
    for p in ["data/processed/events.jsonl", "data/results/event_groups.json",
              "data/results/scored_groups.json", "data/results/alerts.json",
              "data/results/run_manifest.json"]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
