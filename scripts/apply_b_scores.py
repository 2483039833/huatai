"""Apply member B's credibility and impact scores to main event groups.

Usage:
    python scripts/apply_b_scores.py \
        --input data/results/event_groups.json \
        --output data/results/event_groups_with_b_scores.json
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

from src.services.b_score_integration import enrich_event_groups_with_b_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/event_groups.json")
    parser.add_argument("--output", default="data/results/event_groups_with_b_scores.json")
    parser.add_argument("--b-artifact", default=None)
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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    groups = load_groups(input_path)
    enriched = enrich_event_groups_with_b_scores(groups, args.b_artifact)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(enriched, handle, ensure_ascii=False, indent=2)

    matched = sum(
        1
        for item in enriched
        if item.get("b_match_method")
        or item.get("b_score_components")
        or item.get("b_score_reason")
        or item.get("b_alert_level_hint")
    )
    print(f"Applied B scores to {matched}/{len(enriched)} event groups.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
