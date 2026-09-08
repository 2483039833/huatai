"""
Build structured event groups from clustering labels.

This service is the C-module bridge from raw clustered articles to the
EventGroupSchema consumed by scoring, alerting, timeline and frontend views.
"""
from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np

from src.api.schemas.cluster import EvidenceEdge, EventGroupSchema, TimelineStage
from src.api.schemas.event import EventSchema


class GroupBuilder:
    """Build explainable same-source event groups."""

    @staticmethod
    def _group_confidence(embeddings: np.ndarray, member_idx: np.ndarray) -> float:
        """Average nearest-neighbor similarity inside a group."""
        if len(member_idx) <= 1:
            return 0.9
        sub = embeddings[member_idx]
        norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
        sims = norm @ norm.T
        np.fill_diagonal(sims, -1.0)
        return float(np.mean(np.max(sims, axis=1)))

    @staticmethod
    def _representative_idx(events: list[EventSchema], member_idx: list[int]) -> int:
        """Pick the earliest article as the representative evidence item."""
        return min(
            member_idx,
            key=lambda i: events[i].article_publish_time or datetime(1970, 1, 1),
        )

    @staticmethod
    def _evidence_edges(
        events: list[EventSchema],
        embeddings: Optional[np.ndarray],
        member_idx: list[int],
        threshold: float = 0.94,
        max_edges: int = 12,
    ) -> list[EvidenceEdge]:
        """Return the strongest article-pair similarities inside the group."""
        if embeddings is None or len(member_idx) <= 1:
            return []

        sub = embeddings[np.array(member_idx)]
        sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
        sims = sub @ sub.T
        pairs: list[tuple[float, int, int]] = []
        for i in range(len(member_idx)):
            for j in range(i + 1, len(member_idx)):
                sim = float(sims[i, j])
                if sim >= threshold:
                    pairs.append((sim, member_idx[i], member_idx[j]))
        pairs.sort(reverse=True, key=lambda x: x[0])

        return [
            EvidenceEdge(
                source_article_id=events[i].article_file_id,
                target_article_id=events[j].article_file_id,
                similarity=round(sim, 4),
            )
            for sim, i, j in pairs[:max_edges]
        ]

    @staticmethod
    def _stage_for_event(event: EventSchema, is_first: bool) -> tuple[str, str]:
        """Rule-based timeline stage labeling for event tracking."""
        title = event.article_title or ""
        content = event.content or ""
        source = event.article_source or ""
        text = f"{title} {content} {source}"

        official_terms = ("公告", "官方", "交易所", "证监", "上交所", "深交所")
        clarify_terms = ("澄清", "辟谣", "不实", "否认", "回应")
        update_terms = ("进展", "后续", "最新", "扩大", "升级", "影响")

        if any(term in text for term in clarify_terms):
            return "clarification", "contains clarification or rumor-refutation signals"
        if any(term in text for term in official_terms):
            return "official_confirmation", "contains official confirmation signals"
        if not is_first and any(term in text for term in update_terms):
            return "impact_update", "contains follow-up or impact-update signals"
        if is_first:
            return "initial_report", "earliest article in the event group"
        return "follow_up", "later report in the same event group"

    @classmethod
    def _timeline_stages(cls, members: list[EventSchema]) -> list[TimelineStage]:
        ordered = sorted(
            members,
            key=lambda e: e.article_publish_time or datetime(1970, 1, 1),
        )
        stages: list[TimelineStage] = []
        for idx, event in enumerate(ordered):
            stage, reason = cls._stage_for_event(event, idx == 0)
            stages.append(TimelineStage(
                stage=stage,
                article_id=event.article_file_id,
                title=event.article_title or "",
                time=event.article_publish_time,
                source=event.article_source or "",
                reason=reason,
            ))
        return stages

    @staticmethod
    def _cluster_reason(
        article_count: int,
        source_diversity: float,
        confidence: float,
        edge_count: int,
    ) -> str:
        parts = [
            f"{article_count} articles grouped",
            f"cluster confidence {confidence:.2f}",
            f"source diversity {source_diversity:.0%}",
        ]
        if edge_count:
            parts.append(f"{edge_count} high-similarity evidence edges")
        return "; ".join(parts)

    @classmethod
    def build_groups(
        cls,
        events: list[EventSchema],
        labels: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> list[EventGroupSchema]:
        """Map cluster labels to EventGroupSchema objects."""
        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            if lab == -1:
                continue
            groups.setdefault(int(lab), []).append(idx)

        result: list[EventGroupSchema] = []
        for lab, member_idx in groups.items():
            members = [events[i] for i in member_idx]

            times = [e.article_publish_time for e in members if e.article_publish_time]
            timeline_start = min(times) if times else None
            timeline_end = max(times) if times else None

            sources = [e.article_source for e in members if e.article_source]
            source_diversity = len(set(sources)) / len(members) if members else 0.0

            event_names = [e.event_name for e in members if e.event_name]
            event_name = Counter(event_names).most_common(1)[0][0] if event_names else None
            trading_codes = sorted({e.trading_code for e in members if e.trading_code})
            secu_names = sorted({e.secu_abbr for e in members if e.secu_abbr})

            titles = [e.article_title for e in members if e.article_title]
            rep_idx = cls._representative_idx(events, member_idx)
            representative = events[rep_idx]
            merged_title = max(titles, key=len) if titles else representative.article_title or ""
            merged_content = (representative.content or "")[:300]

            confidence = 0.9
            if embeddings is not None:
                confidence = cls._group_confidence(embeddings, np.array(member_idx))
            confidence = round(confidence, 4)

            edges = cls._evidence_edges(events, embeddings, member_idx)
            stages = cls._timeline_stages(members)

            result.append(EventGroupSchema(
                group_id=f"G{lab:06d}",
                events=members,
                cluster_confidence=confidence,
                timeline_start=timeline_start,
                timeline_end=timeline_end,
                merged_title=merged_title,
                merged_content=merged_content,
                article_count=len(members),
                source_diversity=round(source_diversity, 4),
                event_name=event_name,
                trading_codes=trading_codes,
                secu_names=secu_names,
                representative_article_id=representative.article_file_id,
                evidence_edges=edges,
                cluster_reason=cls._cluster_reason(len(members), source_diversity, confidence, len(edges)),
                timeline_stages=stages,
            ))

        result.sort(key=lambda g: g.article_count, reverse=True)
        return result

