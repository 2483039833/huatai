"""
同源事件聚类基线实验
========================
流程：加载标注数据 → BGE embedding → FAISS召回验证 → HDBSCAN聚类 → 评估

用法：
    PYTHONIOENCODING=utf-8 python scripts/baseline_clustering.py
"""
import os
import sys
import time
from pathlib import Path

# 支持 HuggingFace 国内镜像（可选）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).parent.parent))
PYTHONIOENCODING = "utf-8"

import numpy as np
import pandas as pd

from src.preprocessing.loader import DataLoader
from src.preprocessing.cleaner import TextCleaner
from src.features.embedding import EmbeddingEngine
from src.clustering.indexer import VectorIndexer
from src.clustering.cluster import EventClusterer
from src.clustering.evaluator import ClusterEvaluator


def load_and_prepare():
    """加载数据并构造 ground truth"""
    print("=" * 60)
    print("Step 1: 加载有标签公司数据")
    df = DataLoader.load_tagged_company_data()
    print(f"  总文章数: {len(df)}")

    # 构造 ground truth：从 duplication_id 生成组标签
    gt = ClusterEvaluator._ground_truth_from_dup_ids(df["duplication_id"])
    df["gt_label"] = gt

    # 清洗文本
    df["clean_content"] = df["content"].apply(TextCleaner.clean_text)

    # 只取有重复标签的样本作为评估集
    eval_mask = df["gt_label"] != -1
    eval_df = df[eval_mask].copy()
    n_dup_samples = eval_mask.sum()
    n_dup_groups = eval_df["gt_label"].nunique()
    print(f"  含重复标签样本: {n_dup_samples} (占 {n_dup_samples/len(df)*100:.1f}%)")
    print(f"  重复组数量: {n_dup_groups}")
    return df


EMBEDDING_CACHE = Path("data/embeddings/company_content_bge_large.npy")


def step2_embedding(df, model_name=None, use_cache=True):
    """Step 2: BGE 向量化（带缓存，避免重复计算）"""
    print("\n" + "=" * 60)
    print(f"Step 2: BGE embedding 向量化 (model={model_name or 'default'})")

    if use_cache and EMBEDDING_CACHE.exists():
        embeddings = np.load(str(EMBEDDING_CACHE))
        print(f"  从缓存加载: {EMBEDDING_CACHE} ({embeddings.shape})")
        return embeddings

    engine = EmbeddingEngine(model_name=model_name)
    texts = df["clean_content"].fillna("").tolist()
    embeddings = engine.encode(texts, show_progress=True)
    print(f"  向量维度: {embeddings.shape}")

    if use_cache:
        EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(EMBEDDING_CACHE), embeddings)
        print(f"  缓存已保存: {EMBEDDING_CACHE}")
    return embeddings


def step3_faiss_recall(embeddings, df, top_k=20):
    """Step 3: FAISS 召回验证"""
    print("\n" + "=" * 60)
    print(f"Step 3: FAISS Top-{top_k} 召回率验证")

    # 构建索引（每篇文章检索其余所有文章会自匹配，跳过自身）
    indexer = VectorIndexer()
    indexer.build(embeddings, index_type="flat")

    # 记录每篇文章在 ground truth 中的组成员
    gt = df["gt_label"].values
    n = len(df)

    # Recall@K：对每篇文章，检查其前K个最近邻中是否有同组文章
    recall_hits = 0
    recall_total = 0
    # 对每个有重复标签的样本，统计其组内其他成员在Top-K中的比例
    recall_at_k_denominator = 0
    recall_at_k_numerator = 0

    # 用批量检索（分块避免内存爆炸）
    batch_size = 256
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores, indices = indexer.search(embeddings[start:end], top_k)
        for row_idx, idx_row in enumerate(indices):
            global_idx = start + row_idx
            my_gt = gt[global_idx]
            if my_gt == -1:
                continue
            # 组内其他成员集合
            members = np.where(gt == my_gt)[0]
            members = set(members) - {global_idx}
            if not members:
                continue
            recall_total += len(members)
            # 统计Top-K中包含多少组内成员
            top_k_members = set(idx_row) & members
            recall_at_k_numerator += len(top_k_members)

    recall_at_k = recall_at_k_numerator / recall_total if recall_total else 0
    print(f"  Recall@{top_k}: {recall_at_k:.4f} "
          f"({recall_at_k_numerator}/{recall_total})")
    return indexer


def step4_cluster(embeddings, df):
    """Step 4: HDBSCAN 聚类 + 评估"""
    print("\n" + "=" * 60)
    print("Step 4: HDBSCAN 聚类与评估")

    clusterer = EventClusterer(method="hdbscan")
    pred_labels = clusterer.fit_predict(embeddings)
    stats = clusterer.get_cluster_stats()
    print(f"  聚类结果: {stats['n_clusters']} 簇, {stats['n_noise']} 噪声")

    # 评估（仅在有ground truth且被聚类的样本上）
    gt = df["gt_label"].values
    metrics = ClusterEvaluator.evaluate(gt, pred_labels)
    print(f"  评估指标: {metrics}")

    return clusterer, metrics


def step5_grid_search(embeddings, df):
    """Step 5: HDBSCAN 参数网格搜索"""
    print("\n" + "=" * 60)
    print("Step 5: HDBSCAN 参数网格搜索")
    import hdbscan

    gt = df["gt_label"].values
    best = None
    best_score = -1

    for min_cluster_size in [2, 3, 4, 5]:
        for min_samples in [1, 2]:
            for eps in [0.0, 0.3, 0.5]:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    cluster_selection_epsilon=eps,
                )
                labels = clusterer.fit_predict(embeddings)
                metrics = ClusterEvaluator.evaluate(gt, labels)
                purity = metrics.get("purity", 0)
                score = purity
                print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
                      f"eps={eps} → purity={purity:.4f}, ARI={metrics.get('ari', 0):.4f}")

                if score > best_score:
                    best_score = score
                    best = (min_cluster_size, min_samples, eps, metrics)

    print(f"\n  最佳参数: min_cluster_size={best[0]}, min_samples={best[1]}, eps={best[2]}")
    print(f"  最佳指标: {best[3]}")
    return best


def main():
    import argparse
    parser = argparse.ArgumentParser(description="同源事件聚类基线实验")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="本地模型路径（如 ModelScope 下载的目录），默认自动加载",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="强制重新计算 embedding，不使用缓存",
    )
    args = parser.parse_args()

    t0 = time.time()
    df = load_and_prepare()
    embeddings = step2_embedding(df, model_name=args.model_path, use_cache=not args.no_cache)
    indexer = step3_faiss_recall(embeddings, df)
    clusterer, metrics = step4_cluster(embeddings, df)
    best = step5_grid_search(embeddings, df)

    print("\n" + "=" * 60)
    print(f"基线实验完成，总耗时 {time.time() - t0:.1f}s")

    # 保存结果
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    df_result = clusterer.add_cluster_labels_to_df(df)
    df_result.to_csv(results_dir / "baseline_clusters.csv", index=False, encoding="utf-8-sig")
    print(f"聚类结果已保存: {results_dir / 'baseline_clusters.csv'}")


if __name__ == "__main__":
    main()
