from pathlib import Path

import pytest

from src.services.b_score_adapter import BScoreAdapter
from src.services.b_score_integration import enrich_event_groups_with_b_scores


def test_b_score_adapter_normalizes_dashboard_record():
    record = {
        "cluster_id": "EVENT-0010",
        "title": "A company event",
        "event_type": "earnings_forecast_up",
        "credibility": {
            "score": 74.79,
            "level": "medium",
            "components": {"source_reliability": 82.0},
            "reasons": ["confirmed by multiple sources"],
        },
        "impact_score": 83.79,
        "priority": "P3",
        "push_reason": "large expected impact",
    }

    score = BScoreAdapter.normalize_record(record)

    assert score is not None
    assert score.group_id == "EVENT-0010"
    assert score.event_name == "earnings_forecast_up"
    assert score.credibility_score == pytest.approx(0.7479)
    assert score.impact_score == pytest.approx(0.8379)
    assert score.impact_level == "moderate"
    assert score.alert_level_hint == "P0"
    assert score.score_components["source_reliability"] == 82.0
    assert score.score_reason == "large expected impact"


def test_b_score_integration_matches_by_title_when_ids_differ(tmp_path: Path):
    artifact_path = tmp_path / "dashboard_data.json"
    artifact_path.write_text(
        """
        {
          "alerts": [
            {
              "cluster_id": "EVENT-0010",
              "title": "Company issues earnings forecast upgrade",
              "event_type": "earnings_forecast_up",
              "credibility": {"score": 88, "level": "high"},
              "impact_score": 76,
              "priority": "P2",
              "score_components": {"market_relevance": 0.9}
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    groups = [
        {
            "group_id": "G000001",
            "merged_title": "Company issues earnings forecast upgrade",
            "event_name": "unknown",
        }
    ]

    enriched = enrich_event_groups_with_b_scores(groups, artifact_path)

    assert enriched[0]["group_id"] == "G000001"
    assert enriched[0]["event_name"] == "earnings_forecast_up"
    assert enriched[0]["credibility_score"] == pytest.approx(0.88)
    assert enriched[0]["impact_score"] == pytest.approx(0.76)
    assert enriched[0]["b_alert_level_hint"] == "P1"
    assert enriched[0]["b_score_components"]["market_relevance"] == 0.9


def test_b_score_integration_uses_conservative_fuzzy_match(tmp_path: Path):
    artifact_path = tmp_path / "dashboard_data.json"
    artifact_path.write_text(
        """
        {
          "alerts": [
            {
              "cluster_id": "EVENT-0020",
              "title": "Company A wins battery storage contract",
              "event_type": "new_contract",
              "subjects": [{"code": "600001", "name": "Company A"}],
              "event_time": {"value": "2026-05-20"},
              "credibility": {"score": 90, "level": "high"},
              "impact_score": 80,
              "priority": "P2"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    groups = [
        {
            "group_id": "G000020",
            "merged_title": "Company A wins large battery storage project contract",
            "event_name": "new_contract",
            "trading_codes": ["600001"],
            "timeline_start": "2026-05-21T09:30:00",
        },
        {
            "group_id": "G000021",
            "merged_title": "Company A announces unrelated board appointment",
            "event_name": "new_contract",
            "trading_codes": ["600001"],
            "timeline_start": "2026-05-21T09:30:00",
        },
    ]

    enriched = enrich_event_groups_with_b_scores(groups, artifact_path)

    assert enriched[0]["credibility_score"] == pytest.approx(0.9)
    assert enriched[0]["impact_score"] == pytest.approx(0.8)
    assert enriched[0]["b_match_method"] == "fuzzy"
    assert "credibility_score" not in enriched[1]
