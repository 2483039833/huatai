"""
快速冒烟测试：用 TF-IDF 特征验证整个聚类 pipeline 逻辑
=========================================================
无需下载 BGE 模型，30秒内验证：加载 → 向量化 → FAISS → HDBSCAN → 评估

用法：
    PYTHONIOENCODING=utf-8 python scripts/smoke_test_clustering.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.loader import DataLoader
from src.preprocessing.cleaner import TextCleaner
from src.clustering.indexer import VectorIndexer
from src.clustering.cluster import EventClusterer
from src.clustering.evaluator import ClusterEvaluator


def main():
    t0 = time.time()
    print("=" * 60)
    print("冒烟测试：TF-IDF → FAISS → HDBSCAN pipeline")

    # 1. 加载
    df = DataLoader.load_tagged_company_data()
    df["clean_content"] = df["content"].apply(TextCleaner.clean_text)
    gt = ClusterEvaluator._ground_truth_from_dup_ids(df["duplication_id"])
    df["gt_label"] = gt
    print(f"Step 1 加载: {len(df)} 篇文章, {int((gt != -1).sum())} 条含重复标签")

    # 2. TF-IDF 向量化（替代 embedding，仅验证逻辑）
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(df["clean_content"]).toarray().astype(np.float32)
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)
    print(f"Step 2 向量化: {embeddings.shape}")

    # 3. FAISS 索引
    indexer = VectorIndexer(dimension=embeddings.shape[1])
    indexer.build(embeddings, index_type="flat")
    print(f"Step 3 FAISS: {indexer.index.ntotal} 条向量")

    # 4. HDBSCAN 聚类
    clusterer = EventClusterer(method="hdbscan")
    labels = clusterer.fit_predict(embeddings)
    stats = clusterer.get_cluster_stats()
    print(f"Step 4 聚类: {stats['n_clusters']} 簇, {stats['n_noise']} 噪声")

    # 5. 评估
    metrics = ClusterEvaluator.evaluate(gt, labels)
    print(f"Step 5 评估: {metrics}")

    # 6. 事件合并
    from src.clustering.merger import EventMerger
    df_labelled = clusterer.add_cluster_labels_to_df(df)
    events = EventMerger.merge_all_groups(df_labelled)
    print(f"Step 6 合并: 生成 {len(events)} 个事件组")

    print(f"\n{'='*60}")
    print(f"✅ 冒烟测试通过，总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
