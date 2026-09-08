"""
流处理管道编排
"""
from typing import Optional
import numpy as np
from src.features.embedding import EmbeddingEngine
from src.clustering.cluster import EventClusterer
from src.config import settings


class StreamPipeline:
    """流处理管道"""

    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.clusterer = EventClusterer()
        self.buffer = []  # 暂存待聚类的事件

    def process_event(self, event: dict) -> Optional[dict]:
        """
        处理单条事件

        1. 向量化
        2. 与缓冲区已有事件比较相似度
        3. 如果相似度超过阈值则归入同一组

        Args:
            event: 结构化事件 dict

        Returns:
            如果形成新组/组有更新则返回结果，否则 None
        """
        self.buffer.append(event)

        if len(self.buffer) < 2:
            return None

        texts = [str(e.get("content", "")) for e in self.buffer]
        embeddings = self.embedder.encode(texts)

        # 计算与最新事件的相似度
        latest_vec = embeddings[-1:]
        index_embeddings = embeddings[:-1]

        if len(index_embeddings) == 0:
            return None

        from src.features.similarity import SimilarityCalculator
        sim_matrix = SimilarityCalculator.pairwise_cosine(
            np.vstack([index_embeddings, latest_vec])[-2:]
        )

        max_sim = sim_matrix[0, 1] if sim_matrix.shape[0] > 1 else 0

        if max_sim >= settings.similarity_threshold:
            return {
                "is_duplicate": True,
                "max_similarity": float(max_sim),
                "new_event": event,
                "buffer_size": len(self.buffer),
            }
        return None
