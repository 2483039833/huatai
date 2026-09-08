"""Export member B scores in the main-system interface format.

This does not mutate B's original dashboard artifact.  It creates a standard
B-to-C handoff file keyed by the main system's group_id while preserving B's
cluster_id for traceability.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.b_score_adapter import BScoreAdapter


DEFAULT_OUTPUT = (
    ROOT
    / "financial_event_radar_share(1)"
    / "financial_event_radar_share"
    / "web"
    / "data"
    / "main_system_scores.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="data/results/event_groups.json")
    parser.add_argument("--b-artifact", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-unmatched", action="store_true")
    return parser.parse_args()


def load_groups(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        groups = payload.get("groups") or payload.get("event_groups") or payload.get("data")
        if isinstance(groups, list):
            return [item for item in groups if isinstance(item, dict)]
    raise ValueError(f"Cannot find event groups in {path}")


def article_ids(group: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for event in group.get("events") or []:
        if isinstance(event, dict) and event.get("article_file_id"):
            values.append(str(event["article_file_id"]))
    return values


def export_record(group: dict[str, Any], score: Any | None) -> dict[str, Any]:
    base = {
        "group_id": group.get("group_id"),
        "article_ids": article_ids(group),
        "representative_article_id": group.get("representative_article_id"),
        "merged_title": group.get("merged_title"),
        "event_name": group.get("event_name"),
        "trading_codes": group.get("trading_codes") or [],
        "secu_names": group.get("secu_names") or [],
        "matched": bool(score),
    }
    if not score:
        return base

    raw = score.raw
    base.update(
        {
            "matched": True,
            "b_cluster_id": raw.get("cluster_id"),
            "b_alert_id": raw.get("alert_id"),
            "b_title": raw.get("title") or raw.get("canonical_title"),
            "b_event_type": score.event_name,
            "credibility_score": score.credibility_score,
            "impact_score": score.impact_score,
            "impact_level": score.impact_level,
            "sentiment_score": score.sentiment_score,
            "b_alert_level_hint": score.alert_level_hint,
            "score_components": score.score_components,
            "score_reason": score.score_reason,
            "match_method": raw.get("_match_method"),
            "match_score": raw.get("_match_score"),
        }
    )
    return base


def main() -> None:
    args = parse_args()
    groups = load_groups(Path(args.groups))
    adapter = BScoreAdapter(args.b_artifact)
    by_id, by_title = adapter.load_score_indexes()

    records: list[dict[str, Any]] = []
    for group in groups:
        score = adapter.find_score(group, by_id=by_id, by_title=by_title)
        if score or args.include_unmatched:
            records.append(export_record(group, score))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "BScoreForMainSystem/v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_artifact": str(adapter.artifact_path),
        "total_groups": len(groups),
        "matched_groups": sum(1 for item in records if item["matched"]),
        "records": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {payload['matched_groups']}/{payload['total_groups']} B score records.")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
