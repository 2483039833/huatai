"""Integration helpers for applying member B scores to main-system groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.b_score_adapter import BScoreAdapter


def enrich_event_groups_with_b_scores(
    groups: list[dict[str, Any]],
    artifact_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Merge member B's credibility and impact scores into event groups.

    The function is intentionally non-invasive: unmatched groups are returned
    unchanged, and the main system still owns final alert-level calculation.
    """

    adapter = BScoreAdapter(artifact_path)
    by_id, by_title = adapter.load_score_indexes()
    enriched_groups: list[dict[str, Any]] = []

    for group in groups:
        score = adapter.find_score(group, by_id=by_id, by_title=by_title)
        enriched_groups.append(adapter.enrich_group(group, score))

    return enriched_groups
