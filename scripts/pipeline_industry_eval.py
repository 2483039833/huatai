"""
行业数据（Sheet2/Sheet4）端到端验证
====================================
公司维度（Sheet1+Sheet3）已验证通过（purity=0.964），本脚本验证技术路线
能否泛化到行业维度：Sheet2 行业原文(556) + Sheet4 行业转载(500) 合并聚类评估。

用法：
    PYTHONIOENCODING=utf-8 python scripts/pipeline_industry_eval.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from src.preprocessing.loader import DataLoader
from src.preprocessing.cleaner import TextCleaner
from src.features.embedding import EmbeddingEngine
from src.clustering.indexer import VectorIndexer
from src.clustering.evaluator import ClusterEvaluator


def build_ground_truth(df2, df4):
    """行业版 ground truth：转载→原文 + 原文间共享 duplication_id"""
    df2_ids = set(df2["article_file_id"].astype(str))
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, row in df4.iterrows():
        t = str(row["duplication_id"]).strip()
        if t in df2_ids:
            union(f"d:{row['article_file_id']}", f"o:{t}")
    for _, row in df2.iterrows():
        ids = DataLoader.parse_duplication_id(row["duplication_id"])
        if len(ids) >= 2:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    union(f"o:{ids[i]}", f"o:{ids[j]}")

    labels = np.full(len(df2) + len(df4), -1, dtype=int)
    counter, root_map = 0, {}
    for idx, aid in enumerate(df2["article_file_id"].astype(str)):
        root = find(f"o:{aid}")
        if root not in root_map:
            root_map[root] = counter
            counter += 1
        if any(x.startswith("d:") and find(x) == root for x in parent):
            labels[idx] = root_map[root]
    for idx, aid in enumerate(df4["article_file_id"].astype(str)):
        if f"d:{aid}" in parent:
            root = find(f"d:{aid}")
            if root in root_map:
                labels[len(df2) + idx] = root_map[root]
    return labels


def main():
    t0 = time.time()
    print("=" * 60)
    print("行业数据（Sheet2 + Sheet4）端到端验证")

    # 1. 加载
    df2 = DataLoader.load_tagged_industry_data()
    df4 = DataLoader.load_industry_duplication()
    print(f"Sheet2 行业原文: {len(df2)}, Sheet4 转载: {len(df4)}, 合计: {len(df2)+len(df4)}")

    # 2. ground truth
    gt = build_ground_truth(df2, df4)
    n_groups = len(set(gt[gt != -1]))
    print(f"ground truth: {n_groups} 组, 覆盖 {np.sum(gt != -1)} 篇")

    # 3. 向量化（优先复用缓存）
    cache = Path("data/embeddings/industry_full.npy")
    if cache.exists():
        emb = np.load(str(cache))
        print(f"[缓存] 加载 {cache}  {emb.shape}")
    else:
        texts = (df2["content"].fillna("").apply(TextCleaner.clean_text).tolist()
                 + df4["content"].fillna("").apply(TextCleaner.clean_text).tolist())
        engine = EmbeddingEngine()
        emb = engine.encode(texts, show_progress=True)
        np.save(str(cache), emb)
        print(f"[缓存] 保存 {cache}  {emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-9)

    # 4. FAISS + 相似度图 + 阈值搜索
    indexer = VectorIndexer(dimension=emb.shape[1])
    indexer.build(emb, index_type="flat")
    scores, indices = indexer.search(emb, 20)

    print(f"\n{'threshold':>10} {'purity':>8} {'ARI':>8} {'NMI':>8} {'groups':>7} {'cov_gt':>8}")
    best = None
    for threshold in [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96]:
        G = nx.Graph()
        G.add_nodes_from(range(len(emb)))
        for i in range(len(emb)):
            for sim, j in zip(scores[i], indices[i]):
                if j == -1 or j == i:
                    continue
                if sim >= threshold:
                    G.add_edge(i, int(j), weight=float(sim))
        comps = [c for c in nx.connected_components(G) if len(c) >= 2]
        pred = np.full(len(emb), -1, dtype=int)
        for cid, c in enumerate(comps):
            for node in c:
                pred[node] = cid

        mask = gt != -1
        gt_v, pred_v = gt[mask], pred[mask]
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        purity = ClusterEvaluator.purity(gt_v, pred_v)
        ari = adjusted_rand_score(gt_v, pred_v)
        nmi = normalized_mutual_info_score(gt_v, pred_v)

        gt_groups = defaultdict(list)
        for idx, g in enumerate(gt):
            if g != -1:
                gt_groups[int(g)].append(idx)
        detected = sum(1 for members in gt_groups.values()
                       if len(set(pred[members]) - {-1}) == 1 and len(members) >= 2)
        rate = detected / len(gt_groups) if gt_groups else 0

        print(f"  {threshold:>10.2f} {purity:>8.4f} {ari:>8.4f} {nmi:>8.4f} "
              f"{len(comps):>7} {rate:>8.2%}")
        if best is None or ari > best[1]:
            best = (threshold, ari, {"purity": purity, "nmi": nmi, "rate": rate})

    print(f"\n最佳 ARI 阈值: {best[0]} → {best[2]}")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
