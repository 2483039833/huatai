"""
聚类主逻辑：GraphCluster（相似度图+连通分量，推荐）/ HDBSCAN / DBSCAN

GraphCluster 是基于基线实验验证的最优策略：
同源事件对相似度极高（median≈0.998），用 FAISS Top-K 建图 + 高阈值连通分量
划分事件组，purity=0.964 / ARI=0.791 / NMI=0.989（见 docs/BASELINE_REPORT.md）。
"""
from typing import Optional
import numpy as np
import pandas as pd

from src.config import settings


class EventClusterer:
    """事件聚类器"""

    def __init__(self, method: Optional[str] = None):
        self.method = method or settings.clustering_method
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None  # HDBSCAN 置信度
        self.threshold: float = settings.similarity_threshold

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        对向量执行聚类

        Args:
            embeddings: shape (n, d)，已归一化的向量

        Returns:
            labels: shape (n,)，-1 表示噪声点（未聚类）
        """
        if self.method in ("graph", "graph_cluster"):
            return self._graph_cluster(embeddings)
        elif self.method == "hdbscan":
            return self._hdbscan_cluster(embeddings)
        elif self.method == "dbscan":
            return self._dbscan_cluster(embeddings)
        else:
            raise ValueError(f"不支持的聚类方法: {self.method}")

    def _graph_cluster(self, embeddings: np.ndarray, top_k: int = 20) -> np.ndarray:
        """
        相似度图 + 连通分量聚类（推荐策略，基线验证最优）

        1. 用 FAISS 找每篇文章的 Top-K 近邻
        2. 相似度 ≥ 阈值 → 连边
        3. 连通分量 = 事件组

        Args:
            embeddings: shape (n, d) 归一化向量
            top_k: FAISS 检索近邻数

        Returns:
            labels: shape (n,)，-1 表示未归组（孤立点）
        """
        import networkx as nx
        from src.clustering.indexer import VectorIndexer

        indexer = VectorIndexer(dimension=embeddings.shape[1])
        indexer.build(embeddings, index_type="flat")
        scores, indices = indexer.search(embeddings, top_k)

        G = nx.Graph()
        G.add_nodes_from(range(len(embeddings)))
        threshold = self.threshold
        for i in range(len(embeddings)):
            for sim, j in zip(scores[i], indices[i]):
                if j == -1 or j == i:
                    continue
                if sim >= threshold:
                    G.add_edge(i, int(j), weight=float(sim))

        components = [c for c in nx.connected_components(G) if len(c) >= 2]
        labels = np.full(len(embeddings), -1, dtype=int)
        for comp_id, comp in enumerate(components):
            for node in comp:
                labels[node] = comp_id

        self.labels_ = labels
        n_noise = int(np.sum(labels == -1))
        print(f"[Cluster] 相似度图+连通分量完成: {len(components)} 个事件组, "
              f"{n_noise} 个孤立点 (threshold={threshold})")
        return labels

    def _hdbscan_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """HDBSCAN 层次聚类"""
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=settings.hdbscan_min_cluster_size,
            min_samples=settings.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_epsilon=settings.hdbscan_min_cluster_size,
            prediction_data=True,
        )
        self.labels_ = clusterer.fit_predict(embeddings)
        self.probabilities_ = clusterer.probabilities_

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = int(np.sum(self.labels_ == -1))
        print(f"[Cluster] HDBSCAN 完成: {n_clusters} 个簇, {n_noise} 个噪声点")

        return self.labels_

    def _dbscan_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """DBSCAN 密度聚类"""
        from sklearn.cluster import DBSCAN

        clusterer = DBSCAN(
            eps=settings.dbscan_eps,
            min_samples=settings.dbscan_min_samples,
            metric="cosine",
        )
        self.labels_ = clusterer.fit_predict(embeddings)

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = int(np.sum(self.labels_ == -1))
        print(f"[Cluster] DBSCAN 完成: {n_clusters} 个簇, {n_noise} 个噪声点")

        return self.labels_

    def get_cluster_stats(self) -> dict:
        """返回聚类统计信息"""
        if self.labels_ is None:
            return {}
        unique, counts = np.unique(self.labels_, return_counts=True)
        stats = {
            "n_clusters": int(np.sum(unique != -1)),
            "n_noise": int(counts[unique == -1]) if -1 in unique else 0,
            "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts) if k != -1},
            "method": self.method,
        }
        return stats

    def add_cluster_labels_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """将聚类标签添加到 DataFrame"""
        if self.labels_ is None:
            raise ValueError("尚未执行聚类")
        result = df.copy()
        result["cluster_label"] = self.labels_
        if self.probabilities_ is not None:
            result["cluster_probability"] = self.probabilities_
        return result
