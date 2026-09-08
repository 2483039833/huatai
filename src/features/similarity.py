"""
相似度计算工具：余弦相似度、pairwise对比
"""
import numpy as np


class SimilarityCalculator:
    """相似度计算工具"""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        if norm == 0:
            return 0.0
        return dot / norm

    @staticmethod
    def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
        """
        计算所有向量两两之间的余弦相似度矩阵

        Args:
            embeddings: shape (n, d)

        Returns:
            shape (n, n) 的相似度矩阵
        """
        # 假设已经 L2 归一化，内积 = 余弦相似度
        return np.dot(embeddings, embeddings.T)

    @staticmethod
    def top_k_similar(
        query_vec: np.ndarray,
        index_embeddings: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        对单个 query 向量，在索引中找到 Top-K 最相似的

        Args:
            query_vec: shape (d,)
            index_embeddings: shape (n, d)
            k: 返回 top-k

        Returns:
            (similarities, indices) 均按相似度降序排列
        """
        similarities = np.dot(index_embeddings, query_vec)
        top_k = np.argsort(similarities)[::-1][:k]
        return similarities[top_k], top_k
