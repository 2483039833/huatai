"""Incremental event assignment for streaming or near-real-time updates."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.api.schemas.cluster import EventGroupSchema
from src.api.schemas.event import EventSchema
from src.config import settings


@dataclass
class IncrementalAssignment:
    """Result of assigning one new event to existing groups."""
    article_id: str
    matched_group_id: Optional[str]
    similarity: float
    is_new_group: bool
    reason: str

    def model_dump(self) -> dict:
        return {
            "article_id": self.article_id,
            "matched_group_id": self.matched_group_id,
            "similarity": round(self.similarity, 4),
            "is_new_group": self.is_new_group,
            "reason": self.reason,
        }


class IncrementalClusterer:
    """Assign a new event to an existing event group using representative vectors."""

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold or settings.similarity_threshold

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def _representative_indices(groups: list[EventGroupSchema], article_ids: list[str]) -> list[Optional[int]]:
        article_pos = {aid: idx for idx, aid in enumerate(article_ids)}
        indices: list[Optional[int]] = []
        for group in groups:
            idx = None
            if group.representative_article_id:
                idx = article_pos.get(group.representative_article_id)
            if idx is None:
                for event in group.events:
                    idx = article_pos.get(event.article_file_id)
                    if idx is not None:
                        break
            indices.append(idx)
        return indices

    def assign(
        self,
        event: EventSchema,
        event_embedding: np.ndarray,
        groups: list[EventGroupSchema],
        article_ids: list[str],
        article_embeddings: np.ndarray,
    ) -> IncrementalAssignment:
        """Assign one event to the best matching existing group, or mark as new."""
        if not groups or len(article_ids) == 0:
            return IncrementalAssignment(
                article_id=event.article_file_id,
                matched_group_id=None,
                similarity=0.0,
                is_new_group=True,
                reason="no existing groups available",
            )

        rep_indices = self._representative_indices(groups, article_ids)
        candidates = [(group, idx) for group, idx in zip(groups, rep_indices) if idx is not None]
        if not candidates:
            return IncrementalAssignment(
                article_id=event.article_file_id,
                matched_group_id=None,
                similarity=0.0,
                is_new_group=True,
                reason="existing groups have no representative article vectors",
            )

        reps = np.vstack([article_embeddings[idx] for _, idx in candidates])
        reps = self._normalize(reps)
        query = self._normalize(event_embedding)
        sims = (query @ reps.T).ravel()
        best_pos = int(np.argmax(sims))
        best_group = candidates[best_pos][0]
        best_sim = float(sims[best_pos])

        if best_sim >= self.threshold:
            return IncrementalAssignment(
                article_id=event.article_file_id,
                matched_group_id=best_group.group_id,
                similarity=best_sim,
                is_new_group=False,
                reason=f"matched representative article above threshold {self.threshold:.2f}",
            )

        return IncrementalAssignment(
            article_id=event.article_file_id,
            matched_group_id=None,
            similarity=best_sim,
            is_new_group=True,
            reason=f"best similarity below threshold {self.threshold:.2f}",
        )
