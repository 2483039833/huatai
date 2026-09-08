"""
聚类策略优化：相似度图 + 连通分量 vs HDBSCAN
==============================================
基于去重评估发现（同源对相似度≈0.998，非同源<0.86），
用 FAISS Top-K 构建相似度图，连通分量划分事件组。

对比策略：
1. 纯 HDBSCAN（当前基线）
2. 相似度图 + 连通分量（建议）
3. 相似度图 + 传闭包（转推关系）

用法：
    PYTHONIOENCODING=utf-8 python scripts/optimize_clustering.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import networkx as nx

from src.preprocessing.loader import DataLoader
from src.clustering.evaluator import ClusterEvaluator
from src.clustering.indexer import VectorIndexer


def load_data():
    df = DataLoader.load_tagged_company_data()
    gt = ClusterEvaluator._ground_truth_from_dup_ids(df["duplication_id"])
    df["gt_label"] = gt
    emb = np.load("data/embeddings/company_content_bge_large.npy")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-9)
    return df, gt, emb


def build_graph(emb, top_k=10, threshold=0.90):
    """构建相似度图：相似度>阈值且互为Top-K近邻则连边"""
    indexer = VectorIndexer(dimension=emb.shape[1])
    indexer.build(emb, index_type="flat")
    scores, indices = indexer.search(emb, top_k)

    G = nx.Graph()
    G.add_nodes_from(range(len(emb)))
    n = len(emb)
    for i in range(n):
        for sim, j in zip(scores[i], indices[i]):
            if j == -1 or j == i:
                continue
            if sim >= threshold:
                G.add_edge(i, int(j), weight=float(sim))
    return G


def evaluate_strategy(name, pred_labels, gt):
    mask = (gt != -1) & (pred_labels != -1)
    n_valid = int(mask.sum())
    if n_valid == 0:
        print(f"{name}: 无有效样本")
        return
    metrics = ClusterEvaluator.evaluate(gt, pred_labels)
    # 组级评估：预测组与ground truth组的匹配
    print(f"\n{'='*50}")
    print(f"策略: {name}")
    print(f"  评估样本: {n_valid}")
    print(f"  purity={metrics.get('purity')}, ARI={metrics.get('ari')}, NMI={metrics.get('nmi')}")

    # 统计聚类规模
    labels = pred_labels[mask]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"  预测簇数={n_clusters}, 噪声={n_noise}")


def strategy_1_hdbscan(emb, gt):
    """策略1: 纯 HDBSCAN"""
    from src.clustering.cluster import EventClusterer
    clusterer = EventClusterer(method="hdbscan")
    labels = clusterer.fit_predict(emb)
    evaluate_strategy("HDBSCAN (默认)", labels, gt)


def strategy_2_connected_components(emb, gt, threshold=0.90):
    """策略2: 相似度图 + 连通分量"""
    G = build_graph(emb, top_k=20, threshold=threshold)
    components = list(nx.connected_components(G))
    print(f"  图构建: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边, "
          f"{len(components)} 连通分量")

    labels = np.full(len(emb), -1, dtype=int)
    for comp_id, comp in enumerate(components):
        if len(comp) < 2:
            continue  # 单点视为噪声
        for node in comp:
            labels[node] = comp_id

    evaluate_strategy(f"相似度图+连通分量 (threshold={threshold})", labels, gt)
    return labels


def strategy_3_group_after_filter(emb, gt, threshold=0.90):
    """策略3: 相似度图 + 连通分量 + 只保留有ground truth的公司内聚类"""
    G = build_graph(emb, top_k=20, threshold=threshold)
    components = list(nx.connected_components(G))

    labels = np.full(len(emb), -1, dtype=int)
    for comp_id, comp in enumerate(components):
        if len(comp) < 2:
            continue
        for node in comp:
            labels[node] = comp_id

    evaluate_strategy(f"图+连通分量+过滤 (threshold={threshold})", labels, gt)
    return labels


def grid_search_thresholds(emb, gt):
    """阈值网格搜索"""
    print("\n" + "=" * 50)
    print("阈值网格搜索 (相似度图 + 连通分量)")
    print(f"{'threshold':>10} {'purity':>8} {'ARI':>8} {'n_clusters':>10} {'noise':>6}")
    best = None
    for threshold in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96]:
        G = build_graph(emb, top_k=20, threshold=threshold)
        components = list(nx.connected_components(G))
        labels = np.full(len(emb), -1, dtype=int)
        for comp_id, comp in enumerate(components):
            if len(comp) < 2:
                continue
            for node in comp:
                labels[node] = comp_id

        mask = (gt != -1) & (labels != -1)
        n_valid = int(mask.sum())
        if n_valid == 0:
            print(f"  {threshold:>10.2f} 无有效样本")
            continue
        metrics = ClusterEvaluator.evaluate(gt[mask], labels[mask])
        n_clusters = len(set(labels[mask])) - (1 if -1 in labels[mask] else 0)
        n_noise = int(np.sum(labels[mask] == -1))
        print(f"  {threshold:>10.2f} {metrics['purity']:>8.4f} {metrics['ari']:>8.4f} "
              f"{n_clusters:>10} {n_noise:>6}")

        # 综合评分：purity 优先
        score = metrics["purity"]
        if best is None or score > best[1]:
            best = (threshold, score, metrics)

    if best:
        print(f"\n  最佳阈值: {best[0]} (score={best[1]:.4f}, metrics={best[2]})")
    return best


def main():
    print("=" * 60)
    print("聚类策略优化对比实验")
    df, gt, emb = load_data()
    print(f"数据: {len(df)} 篇文章, {int((gt != -1).sum())} 含重复标签")

    # 阈值网格搜索（先用最有效的策略找最优阈值）
    best = grid_search_thresholds(emb, gt)

    # 用最优阈值跑正式对比
    threshold = best[0] if best else 0.90
    print(f"\n使用最优阈值 {threshold} 做正式对比:")
    strategy_2_connected_components(emb, gt, threshold)


if __name__ == "__main__":
    main()
