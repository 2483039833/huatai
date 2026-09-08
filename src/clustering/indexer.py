"""
FAISS 向量索引构建与检索
"""
from typing import Optional
import numpy as np
import faiss

from src.config import settings


class VectorIndexer:
    """基于 FAISS 的向量索引"""

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension or settings.embedding_dimension
        self.index: Optional[faiss.Index] = None
        self.id_map: Optional[np.ndarray] = None  # FAISS index ID → 原始数据索引

    def build(self, embeddings: np.ndarray, index_type: str = "flat") -> None:
        """
        构建 FAISS 索引

        Args:
            embeddings: shape (n, d) 归一化向量
            index_type: flat（精确检索）/ ivf（近似检索）
        """
        # 动态推断维度（兼容不同 embedding 模型的输出维度）
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
        assert embeddings.shape[1] == self.dimension, \
            f"向量维度不匹配: 输入{embeddings.shape[1]}, 期望{self.dimension}"

        if index_type == "flat":
            # IP (内积) 索引，适用于 L2 归一化后的余弦相似度
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "ivf":
            # IVF 倒排索引，适合大规模
            nlist = max(1, int(np.sqrt(embeddings.shape[0])))
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")

        self.index.add(embeddings)
        self.id_map = np.arange(len(embeddings))
        print(f"[Indexer] FAISS 索引构建完成: {self.index.ntotal} 条向量，类型={index_type}")

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        检索 Top-K 相似向量

        Args:
            query: shape (nq, d) 查询向量
            k: 返回数

        Returns:
            (scores, indices) shape (nq, k)，scores 为余弦相似度
        """
        assert self.index is not None, "索引未构建，请先调用 build()"
        return self.index.search(query, k)

    def search_single(self, query_vec: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """检索单个向量的 Top-K"""
        return self.search(query_vec.reshape(1, -1), k)

    def batch_search_and_filter(
        self,
        queries: np.ndarray,
        threshold: float = 0.0,
        k: int = 20,
    ) -> list[list[tuple[int, float]]]:
        """
        批量检索并过滤：对每个 query 返回相似度 > threshold 的结果

        Args:
            queries: shape (n, d)
            threshold: 相似度阈值
            k: 最多返回 k 个

        Returns:
            list of list of (index_in_original, similarity)
        """
        scores, indices = self.search(queries, k)
        results = []
        for score_row, idx_row in zip(scores, indices):
            matches = []
            for sim, idx in zip(score_row, idx_row):
                if idx == -1:
                    continue
                if sim >= threshold:
                    matches.append((int(idx), float(sim)))
            results.append(matches)
        return results

    def save(self, path: str) -> None:
        """保存索引到磁盘"""
        assert self.index is not None
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, dimension: Optional[int] = None) -> "VectorIndexer":
        """从磁盘加载索引"""
        indexer = cls(dimension)
        indexer.index = faiss.read_index(path)
        return indexer
