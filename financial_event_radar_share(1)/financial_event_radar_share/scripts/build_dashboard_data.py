"""Create the browser-ready data bundle for the event-radar dashboard."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "event_radar" / "data" / "derived"
WEB_DATA_DIR = ROOT / "event_radar" / "web" / "data"
ALERT_PATH = DERIVED_DIR / "alert_feed_test.jsonl.gz"
TIMELINE_PATH = DERIVED_DIR / "event_timelines_test.jsonl.gz"
OUTPUT_PATH = WEB_DATA_DIR / "dashboard_data.json"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def main() -> None:
    if not ALERT_PATH.exists() or not TIMELINE_PATH.exists():
        raise FileNotFoundError("Run alerting_baseline.py and timeline_baseline.py before building dashboard data")
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timelines = {row["cluster_id"]: row for row in iter_jsonl(TIMELINE_PATH)}
    alerts = []
    for alert in iter_jsonl(ALERT_PATH):
        timeline = timelines.get(alert["cluster_id"], {})
        alert["timeline_summary"] = timeline.get("timeline_summary", "")
        alert["article_count"] = timeline.get("article_count", 0)
        alert["source_count"] = timeline.get("source_count", 0)
        alert["full_timeline"] = timeline.get("milestones", [])
        alerts.append(alert)
    priorities = Counter(alert["priority"] for alert in alerts)
    statuses = Counter(alert["timeline_status"] for alert in alerts)
    credibility = Counter(alert["credibility"].get("level", "unknown") for alert in alerts)
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "title": "金融事件雷达",
        "summary": {
            "total_events": len(alerts),
            "priority_distribution": dict(priorities),
            "status_distribution": dict(statuses),
            "credibility_distribution": dict(credibility),
            "latest_update": max((alert.get("last_updated_at") for alert in alerts if alert.get("last_updated_at")), default=None),
        },
        "alerts": alerts,
    }
    OUTPUT_PATH.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    print(f"Dashboard data written to {OUTPUT_PATH} ({len(alerts)} events)")


if __name__ == "__main__":
    main()
