"""Regenerate scored groups and alerts after applying member B scores.

This script starts from the existing main-system event groups, scores every
group with the member-B style algorithm, and lets the C alert engine recompute
the final alert level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.alert_engine import AlertEngine
from src.services.b_algorithm_scorer import BAlgorithmScorer
from src.api.schemas.cluster import EventGroupSchema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/event_groups.json")
    parser.add_argument("--scored-output", default="data/results/scored_groups.json")
    parser.add_argument("--alerts-output", default="data/results/alerts.json")
    parser.add_argument("--no-b-classifier", action="store_true")
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in payload if isinstance(item, dict)]


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)


def main() -> None:
    args = parse_args()
    groups = load_json_list(Path(args.input))
    scored = BAlgorithmScorer.score_batch(
        [EventGroupSchema(**item) for item in groups],
        use_event_classifier=not args.no_b_classifier,
    )
    alerted = AlertEngine.apply_batch(scored)

    write_json(Path(args.scored_output), [item.model_dump(mode="json") for item in scored])
    write_json(Path(args.alerts_output), [item.model_dump(mode="json") for item in alerted])

    mode = "with B event classifier" if not args.no_b_classifier else "without B event classifier"
    print(f"Applied B algorithm scores to {len(scored)}/{len(scored)} scored groups ({mode}).")
    print(f"Wrote {args.scored_output}")
    print(f"Wrote {args.alerts_output}")


if __name__ == "__main__":
    main()
