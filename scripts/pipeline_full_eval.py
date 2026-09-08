"""
端到端综合评估：完整去重+聚类 pipeline
======================================
合并 Sheet1(原文) + Sheet3(转载/谣言)，统一向量化 → 相似度图聚类，
用 ground truth（转载→原文映射 + 原文间共享duplication_id）评估。

生产级 pipeline 设计：
1. 新文章来 → BGE 向量化
2. FAISS 在事件库中检索 Top-K
3. 相似度 ≥ 阈值 → 归入既有事件组；否则新建组
4. 事件组 = 同源事件（多篇报道一个事件）

用法：
    PYTHONIOENCODING=utf-8 python scripts/pipeline_full_eval.py
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


def build_ground_truth(df1, df3):
    """
    构建综合 ground truth：
    - Sheet3 转载文章与其指向的 Sheet1 原文同组
    - Sheet1 内部共享 duplication_id 的原文同组
    - 指向同一原文的多篇转载同组

    Returns:
        gt_labels: 每篇文章的组标签（-1 无组）
        group_info: {label: {articles: [...], source: ...}}
    """
    # 转载 → 原文映射
    df1_ids = set(df1["article_file_id"].astype(str))
    dup_to_orig = {}
    for _, row in df3.iterrows():
        target = str(row["duplication_id"]).strip()
        if target in df1_ids:
            dup_to_orig[str(row["article_file_id"])] = target

    # 使用并查集合并同组
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

    # 1. 转载与原文关联
    for dup_id, orig_id in dup_to_orig.items():
        union(f"dup:{dup_id}", f"orig:{orig_id}")

    # 2. Sheet1 内部共享 duplication_id 的原文关联
    id_to_articles = defaultdict(list)
    for idx, aid in enumerate(df1["article_file_id"].astype(str)):
        id_to_articles[aid].append(idx)
    for _, row in df1.iterrows():
        ids = DataLoader.parse_duplication_id(row["duplication_id"])
        if len(ids) >= 2:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    union(f"orig:{ids[i]}", f"orig:{ids[j]}")

    # 3. 为每篇文章分配标签
    labels = np.full(len(df1) + len(df3), -1, dtype=int)
    label_counter = 0
    root_to_label = {}

    # 原文（df1）
    for idx, aid in enumerate(df1["article_file_id"].astype(str)):
        root = find(f"orig:{aid}")
        # 该组是否有转载
        group_size = sum(1 for x in parent if find(x) == root and x.startswith("dup:"))
        if group_size > 0:
            if root not in root_to_label:
                root_to_label[root] = label_counter
                label_counter += 1
            labels[idx] = root_to_label[root]

    # 转载（df3）
    for idx, aid in enumerate(df3["article_file_id"].astype(str)):
        if f"dup:{aid}" in parent:
            root = find(f"dup:{aid}")
            labels[len(df1) + idx] = root_to_label.get(root, -1)

    return labels


def main():
    t0 = time.time()
    print("=" * 60)
    print("端到端综合评估：原文+转载 统一聚类")

    # 1. 加载
    df1 = DataLoader.load_tagged_company_data()
    df3 = DataLoader.load_company_duplication()
    print(f"原文: {len(df1)}, 转载: {len(df3)}, 合计: {len(df1)+len(df3)}")

    # 2. ground truth
    gt = build_ground_truth(df1, df3)
    n_gt = int(np.sum(gt != -1))
    n_groups = len(set(gt[gt != -1]))
    print(f"ground truth 组: {n_groups} 组, 覆盖 {n_gt} 篇文章")

    # 3. 合并文本并向量化（优先复用已缓存的 sheet1/sheet3 embedding）
    cache = Path("data/embeddings/full_merged.npy")
    cache_s1 = Path("data/embeddings/sheet1_orig.npy")
    cache_s3 = Path("data/embeddings/sheet3_dup.npy")
    if cache.exists():
        emb = np.load(str(cache))
        print(f"[缓存] 加载 {cache}")
    elif cache_s1.exists() and cache_s3.exists():
        emb_s1 = np.load(str(cache_s1))
        emb_s3 = np.load(str(cache_s3))
        emb = np.vstack([emb_s1, emb_s3])
        np.save(str(cache), emb)
        print(f"[缓存] 拼接 sheet1({emb_s1.shape}) + sheet3({emb_s3.shape}) → {emb.shape}")
    else:
        texts = (df1["content"].fillna("").apply(TextCleaner.clean_text).tolist()
                 + df3["content"].fillna("").apply(TextCleaner.clean_text).tolist())
        engine = EmbeddingEngine()
        emb = engine.encode(texts, show_progress=True)
        np.save(str(cache), emb)
        print(f"[缓存] 保存 {cache}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-9)
    print(f"向量: {emb.shape}")

    # 4. FAISS + 相似度图
    from src.clustering.indexer import VectorIndexer
    indexer = VectorIndexer(dimension=emb.shape[1])
    indexer.build(emb, index_type="flat")
    top_k = 20
    scores, indices = indexer.search(emb, top_k)

    # 5. 评估不同阈值
    print("\n" + "=" * 60)
    print(f"全量评估（{len(emb)} 篇文章）")
    print(f"{'threshold':>10} {'purity':>8} {'ARI':>8} {'NMI':>8} {'groups':>7} {'cov_gt':>8}")
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

        components = [c for c in nx.connected_components(G) if len(c) >= 2]
        pred = np.full(len(emb), -1, dtype=int)
        for cid, comp in enumerate(components):
            for node in comp:
                pred[node] = cid

        # 评估（只统计 ground truth 覆盖的样本）
        mask = gt != -1
        gt_v, pred_v = gt[mask], pred[mask]
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        purity = _purity(gt_v, pred_v)
        ari = adjusted_rand_score(gt_v, pred_v)
        nmi = normalized_mutual_info_score(gt_v, pred_v)
        n_pred_groups = len([c for c in nx.connected_components(G) if len(c) >= 2])

        # 组级检测率：多少 ground truth 组被"正确"检测（组内≥2篇聚到一起）
        gt_groups = defaultdict(list)
        for idx, g in enumerate(gt):
            if g != -1:
                gt_groups[int(g)].append(idx)
        detected = 0
        for members in gt_groups.values():
            preds = set(pred[members])
            preds.discard(-1)
            if len(preds) == 1 and len(members) >= 2:
                detected += 1
        group_detect_rate = detected / len(gt_groups) if gt_groups else 0

        print(f"  {threshold:>10.2f} {purity:>8.4f} {ari:>8.4f} {nmi:>8.4f} "
              f"{n_pred_groups:>7} {group_detect_rate:>8.2%}")

        if best is None or ari > best[1]:
            best = (threshold, ari, {"purity": purity, "nmi": nmi})

    print(f"\n最佳 ARI 阈值: {best[0]} (ARI={best[1]:.4f})")
    print(f"总耗时 {time.time()-t0:.1f}s")


def _purity(true, pred):
    from sklearn.metrics.cluster import contingency_matrix
    cm = contingency_matrix(true, pred)
    return float(np.sum(np.max(cm, axis=0))) / np.sum(cm)


if __name__ == "__main__":
    main()
