"""
同源去重能力评估
================
场景：给定转载/谣言文章（Sheet3），能否从原文库（Sheet1）中找回它对应的原文？

评估指标：
- Recall@1 / Recall@5 / Recall@10（原文是否在Top-K中）
- 阈值法的 precision / recall / F1
- 相似度分布可视化数据

用法：
    PYTHONIOENCODING=utf-8 python scripts/evaluate_dedup.py
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict

from src.preprocessing.loader import DataLoader
from src.preprocessing.cleaner import TextCleaner
from src.features.embedding import EmbeddingEngine

CACHE_PATH = Path("data/embeddings")


def main():
    t0 = time.time()
    print("=" * 60)
    print("同源去重能力评估 (Sheet3转载 → Sheet1原文)")

    # 1. 加载数据
    df1 = DataLoader.load_tagged_company_data()
    df3 = DataLoader.load_company_duplication()
    print(f"原文库: {len(df1)} 篇, 转载文章: {len(df3)} 篇")

    # 构造映射: 转载文章ID → 原文ID
    df1_ids = set(df1["article_file_id"].astype(str))
    dup_map = {}  # dup_article_id -> original_id
    for _, row in df3.iterrows():
        target = str(row["duplication_id"]).strip()
        if target in df1_ids:
            dup_map[str(row["article_file_id"])] = target

    print(f"有效同源对: {len(dup_map)} (转载文章有对应的原文)")

    # 2. 清洗文本
    df1["clean"] = df1["content"].apply(TextCleaner.clean_text)
    df3["clean"] = df3["content"].apply(TextCleaner.clean_text)

    # 3. 向量化（缓存）
    engine = EmbeddingEngine()
    orig_texts = df1["clean"].fillna("").tolist()
    dup_texts = df3["clean"].fillna("").tolist()

    orig_emb = _embed_with_cache(engine, orig_texts, "sheet1_orig")
    dup_emb = _embed_with_cache(engine, dup_texts, "sheet3_dup")

    # 4. FAISS 检索：每篇转载 → 在原文库中找 Top-K
    from src.clustering.indexer import VectorIndexer
    indexer = VectorIndexer(dimension=orig_emb.shape[1])
    indexer.build(orig_emb, index_type="flat")

    idx_to_orig_id = {i: str(aid) for i, aid in enumerate(df1["article_file_id"])}
    orig_id_to_idx = {str(aid): i for i, aid in enumerate(df1["article_file_id"])}

    # 对每篇转载文章检索
    valid_indices = []   # 转载文章在 df3 中的行号（有有效映射）
    query_embs = []
    for i, (_, row) in enumerate(df3.iterrows()):
        if str(row["article_file_id"]) in dup_map:
            valid_indices.append(i)
            query_embs.append(dup_emb[i])

    query_embs = np.array(query_embs)
    print(f"参与检索的转载文章: {len(query_embs)}")

    # 5. 计算 Recall@K
    k_values = [1, 3, 5, 10, 20]
    recall_at_k = {k: 0 for k in k_values}
    all_sims = []  # 记录每对 (转载, 正确原文) 的相似度
    all_neg_sims = []  # 记录 (转载, 随机原文) 的相似度

    batch_size = 128
    for start in range(0, len(query_embs), batch_size):
        end = min(start + batch_size, len(query_embs))
        scores, indices = indexer.search(query_embs[start:end], k=20)
        for row_idx, (score_row, idx_row) in enumerate(zip(scores, indices)):
            i = valid_indices[start + row_idx]
            dup_id = str(df3.iloc[i]["article_file_id"])
            true_orig = dup_map[dup_id]
            true_idx = orig_id_to_idx[true_orig]

            # 正确原文的相似度
            true_sim = float(np.dot(dup_emb[i], orig_emb[true_idx]))
            all_sims.append(true_sim)

            # 负例：随机取一个非正确原文
            neg_candidates = np.setdiff1d(np.arange(len(orig_emb)), [true_idx])
            neg_idx = neg_candidates[np.random.default_rng(int(time.time()) % 1000 + i).integers(len(neg_candidates))]
            all_neg_sims.append(float(np.dot(dup_emb[i], orig_emb[neg_idx])))

            # Recall@K
            for k in k_values:
                if true_idx in idx_row[:k]:
                    recall_at_k[k] += 1

    n = len(query_embs)
    print("\n--- Recall@K ---")
    for k in k_values:
        print(f"  Recall@{k}: {recall_at_k[k]/n:.4f} ({recall_at_k[k]}/{n})")

    # 6. 阈值法 P/R
    pos_sims = np.array(all_sims)
    neg_sims = np.array(all_neg_sims)
    print("\n--- 相似度分布 ---")
    print(f"  同源对相似度: median={np.median(pos_sims):.4f}, min={pos_sims.min():.4f}")
    print(f"  非同源对相似度: median={np.median(neg_sims):.4f}, max={neg_sims.max():.4f}")

    print("\n--- 阈值法 (转载vs原文) ---")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'F1':>10}")
    best_f1 = (0, 0)
    for threshold in np.arange(0.80, 0.995, 0.01):
        tp = int(np.sum(pos_sims >= threshold))
        fp = int(np.sum(neg_sims >= threshold))
        fn = int(np.sum(pos_sims < threshold))
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        if f1 > best_f1[1]:
            best_f1 = (threshold, f1)
        if threshold in [0.90, 0.93, 0.95, 0.97, 0.99]:
            print(f"  {threshold:>10.2f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

    print(f"\n最佳F1阈值: {best_f1[0]:.2f} → F1={best_f1[1]:.4f}")
    print(f"\n总耗时 {time.time()-t0:.1f}s")


def _embed_with_cache(engine, texts, key):
    cache = CACHE_PATH / f"{key}.npy"
    if cache.exists():
        emb = np.load(str(cache))
        print(f"[缓存] {cache} ({emb.shape})")
        return emb
    emb = engine.encode(texts, show_progress=True)
    np.save(str(cache), emb)
    print(f"[保存] {cache}")
    return emb


if __name__ == "__main__":
    main()
