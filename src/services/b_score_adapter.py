"""Adapter for member B's credibility and impact scoring outputs.

The B package currently produces an offline alert/dashboard format.  This
module converts the useful scoring fields from that format into the flatter
schema used by the main system.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PRIORITY_TO_ALERT_LEVEL = {
    "P3": "P0",
    "P2": "P1",
    "P1": "P2",
    "P0": "P3",
}

B_OVERLAY_FIELDS = {
    "b_score_components",
    "b_score_reason",
    "b_alert_level_hint",
    "b_match_method",
    "b_match_score",
}


@dataclass(slots=True)
class BScore:
    """Normalized scoring signal imported from member B."""

    group_id: str
    credibility_score: float | None = None
    impact_score: float | None = None
    impact_level: str | None = None
    event_name: str | None = None
    sentiment_score: float | None = None
    alert_level_hint: str | None = None
    score_components: dict[str, Any] = field(default_factory=dict)
    score_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class BScoreAdapter:
    """Load and normalize member B's offline scoring artifacts."""

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self.artifact_path = Path(artifact_path) if artifact_path else self.default_artifact_path()

    @staticmethod
    def default_artifact_path() -> Path:
        root = Path(__file__).resolve().parents[2]
        standard_path = (
            root
            / "financial_event_radar_share(1)"
            / "financial_event_radar_share"
            / "web"
            / "data"
            / "main_system_scores.json"
        )
        if standard_path.exists():
            return standard_path
        dashboard_path = (
            root
            / "financial_event_radar_share(1)"
            / "financial_event_radar_share"
            / "web"
            / "data"
            / "dashboard_data.json"
        )
        if dashboard_path.exists():
            return dashboard_path
        return (
            root
            / "financial_event_radar_share(1)"
            / "financial_event_radar_share"
            / "data"
            / "derived"
            / "alert_feed_test.jsonl.gz"
        )

    def load_scores(self) -> dict[str, BScore]:
        if not self.artifact_path.exists():
            return {}

        records = list(self._read_records(self.artifact_path))
        scores: dict[str, BScore] = {}
        for record in records:
            score = self.normalize_record(record)
            if score:
                scores[score.group_id] = score
        return scores

    def load_score_indexes(self) -> tuple[dict[str, BScore], dict[str, BScore]]:
        """Build indexes by id and by title for short-term integration.

        Member B's current offline artifacts use their own cluster ids, so title
        matching gives us a bridge until B emits the main system group_id.
        """

        by_id = self.load_scores()
        by_title: dict[str, BScore] = {}
        for score in by_id.values():
            title = (
                score.raw.get("canonical_title")
                or score.raw.get("title")
                or score.raw.get("b_title")
                or score.raw.get("merged_title")
            )
            title_key = self._title_key(title)
            if title_key:
                by_title[title_key] = score
        return by_id, by_title

    def find_score(
        self,
        group: dict[str, Any],
        by_id: dict[str, BScore] | None = None,
        by_title: dict[str, BScore] | None = None,
    ) -> BScore | None:
        if by_id is None or by_title is None:
            by_id, by_title = self.load_score_indexes()

        group_id = str(group.get("group_id") or group.get("cluster_id") or "").strip()
        if group_id and group_id in by_id:
            by_id[group_id].raw["_match_method"] = "id"
            by_id[group_id].raw["_match_score"] = 1.0
            return by_id[group_id]

        for field_name in ("merged_title", "canonical_title", "title"):
            title_key = self._title_key(group.get(field_name))
            if title_key and title_key in by_title:
                by_title[title_key].raw["_match_method"] = "title_exact"
                by_title[title_key].raw["_match_score"] = 1.0
                return by_title[title_key]

        best_score = 0.0
        best_match: BScore | None = None
        for candidate in by_id.values():
            candidate_group_id = str(candidate.raw.get("group_id") or "").strip()
            if candidate_group_id and candidate_group_id != group_id:
                continue
            match_score, accepted = self._candidate_match_score(group, candidate)
            if accepted and match_score > best_score:
                best_score = match_score
                best_match = candidate

        if best_match:
            best_match.raw["_match_method"] = "fuzzy"
            best_match.raw["_match_score"] = round(best_score, 4)
            return best_match
        return None

    def enrich_group(self, group: dict[str, Any], score: BScore | None) -> dict[str, Any]:
        """Return a copy of a group with B's normalized scores applied."""

        enriched = dict(group)
        for field_name in B_OVERLAY_FIELDS:
            enriched.pop(field_name, None)
        if not score:
            return enriched

        if score.credibility_score is not None:
            enriched["credibility_score"] = score.credibility_score
        if score.impact_score is not None:
            enriched["impact_score"] = score.impact_score
        if score.impact_level:
            enriched["impact_level"] = score.impact_level
        if score.event_name and self._should_replace_event_name(enriched.get("event_name")):
            enriched["event_name"] = score.event_name
        if score.sentiment_score is not None:
            enriched["sentiment_score"] = score.sentiment_score
        if score.score_components:
            enriched["b_score_components"] = score.score_components
        if score.score_reason:
            enriched["b_score_reason"] = score.score_reason
        if score.alert_level_hint:
            enriched["b_alert_level_hint"] = score.alert_level_hint
        if score.raw.get("_match_method"):
            enriched["b_match_method"] = score.raw["_match_method"]
        if score.raw.get("_match_score") is not None:
            enriched["b_match_score"] = score.raw["_match_score"]
        return enriched

    @classmethod
    def normalize_record(cls, record: dict[str, Any]) -> BScore | None:
        group_id = str(
            record.get("group_id")
            or record.get("cluster_id")
            or record.get("b_cluster_id")
            or record.get("alert_id")
            or ""
        ).strip()
        if not group_id:
            return None

        credibility = record.get("credibility")
        credibility_score = None
        impact_level = record.get("impact_level")
        score_components: dict[str, Any] = {}
        score_reason = cls._normalize_reason(record.get("score_reason") or record.get("push_reason"))

        if isinstance(credibility, dict):
            credibility_score = cls._normalize_unit_score(credibility.get("score"))
            impact_level = impact_level or credibility.get("level")
            components = credibility.get("components")
            if isinstance(components, dict):
                score_components.update(components)
            reasons = credibility.get("reasons")
            if not score_reason and isinstance(reasons, list):
                score_reason = cls._normalize_reason(reasons)
        else:
            credibility_score = cls._normalize_unit_score(record.get("credibility_score"))

        raw_components = record.get("score_components")
        if isinstance(raw_components, dict):
            score_components.update(raw_components)

        return BScore(
            group_id=group_id,
            credibility_score=credibility_score,
            impact_score=cls._normalize_unit_score(record.get("impact_score")),
            impact_level=cls._normalize_impact_level(impact_level),
            event_name=record.get("b_event_type") or record.get("event_type") or record.get("event_name"),
            sentiment_score=cls._normalize_sentiment(record.get("sentiment_score")),
            alert_level_hint=PRIORITY_TO_ALERT_LEVEL.get(str(record.get("priority", "")).upper()),
            score_components=score_components,
            score_reason=score_reason,
            raw=record,
        )

    @classmethod
    def _read_records(cls, path: Path) -> Iterable[dict[str, Any]]:
        suffixes = "".join(path.suffixes)
        if suffixes.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            yield from (item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(records, list):
                yield from (item for item in records if isinstance(item, dict))
                return
            alerts = payload.get("alerts")
            if isinstance(alerts, list):
                yield from (item for item in alerts if isinstance(item, dict))
            else:
                yield payload

    @staticmethod
    def _normalize_unit_score(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number > 1.0:
            number = number / 100.0
        return max(0.0, min(1.0, number))

    @staticmethod
    def _normalize_sentiment(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number < -1.0 or number > 1.0:
            number = number / 100.0
        return max(-1.0, min(1.0, number))

    @staticmethod
    def _normalize_impact_level(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        mapping = {
            "very_high": "high",
            "high": "high",
            "medium": "moderate",
            "moderate": "moderate",
            "low": "low",
            "very_low": "low",
        }
        return mapping.get(normalized, normalized or None)

    @staticmethod
    def _title_key(value: Any) -> str | None:
        if value is None:
            return None
        key = "".join(str(value).split()).lower()
        return key or None

    @classmethod
    def _title_similarity(cls, left: Any, right: Any) -> float:
        left_grams = cls._char_ngrams(cls._title_key(left) or "")
        right_grams = cls._char_ngrams(cls._title_key(right) or "")
        if not left_grams or not right_grams:
            return 0.0
        return len(left_grams & right_grams) / len(left_grams | right_grams)

    @staticmethod
    def _char_ngrams(value: str, n: int = 2) -> set[str]:
        if not value:
            return set()
        if len(value) < n:
            return {value}
        return {value[index : index + n] for index in range(len(value) - n + 1)}

    @classmethod
    def _candidate_match_score(cls, group: dict[str, Any], score: BScore) -> tuple[float, bool]:
        title_similarity = cls._title_similarity(
            group.get("merged_title") or group.get("title") or group.get("canonical_title"),
            score.raw.get("title")
            or score.raw.get("canonical_title")
            or score.raw.get("merged_title")
            or score.raw.get("b_title"),
        )
        code_overlap = bool(cls._group_codes(group) & cls._b_codes(score.raw))
        event_match = bool(group.get("event_name") and group.get("event_name") == score.event_name)
        days_delta = cls._days_delta(cls._group_date(group), cls._b_date(score.raw))
        time_match = days_delta is not None and days_delta <= 120

        match_score = (
            0.60 * title_similarity
            + (0.25 if code_overlap else 0.0)
            + (0.20 if event_match else 0.0)
            + (0.08 if time_match else 0.0)
        )
        match_score = min(match_score, 1.0)
        accepted = (
            title_similarity >= 0.72
            or (title_similarity >= 0.45 and (code_overlap or event_match))
            or (title_similarity >= 0.28 and code_overlap and event_match)
        )
        return match_score, accepted

    @staticmethod
    def _group_codes(group: dict[str, Any]) -> set[str]:
        return {str(code) for code in group.get("trading_codes") or [] if code}

    @staticmethod
    def _b_codes(record: dict[str, Any]) -> set[str]:
        subjects = record.get("subjects") or []
        codes = {
            str(subject.get("code"))
            for subject in subjects
            if isinstance(subject, dict) and subject.get("code")
        }
        codes.update(str(code) for code in record.get("trading_codes") or [] if code)
        return codes

    @staticmethod
    def _group_date(group: dict[str, Any]) -> str | None:
        value = group.get("timeline_start") or group.get("timeline_end")
        return str(value)[:10] if value else None

    @staticmethod
    def _b_date(record: dict[str, Any]) -> str | None:
        value = record.get("event_time") or record.get("last_updated_at")
        if isinstance(value, dict):
            value = value.get("value")
        return str(value)[:10] if value else None

    @staticmethod
    def _days_delta(left: str | None, right: str | None) -> int | None:
        if not left or not right:
            return None
        try:
            return abs((datetime.fromisoformat(left) - datetime.fromisoformat(right)).days)
        except ValueError:
            return None

    @staticmethod
    def _should_replace_event_name(value: Any) -> bool:
        if value is None:
            return True
        normalized = str(value).strip().lower()
        return normalized in {"", "unknown", "\u672a\u77e5", "\u93c8\ue047\u7151"}

    @staticmethod
    def _normalize_reason(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            parts = [str(item) for item in value if item]
            return "; ".join(parts) if parts else None
        text = str(value).strip()
        return text or None
