from datetime import datetime

import numpy as np

from src.api.schemas.event import EventSchema
from src.api.schemas.scoring import ScoredEventGroupSchema
from src.services.alert_engine import AlertEngine
from src.services.b_algorithm_scorer import BAlgorithmScorer
from src.services.b_event_classifier import BEventClassifier
from src.services.group_builder import GroupBuilder
from src.services.incremental_clusterer import IncrementalClusterer
from scripts.pipeline_end_to_end import build_run_manifest


def _event(article_id: str, title: str, source: str = "source") -> EventSchema:
    return EventSchema(
        article_file_id=article_id,
        article_title=title,
        article_publish_time=datetime(2026, 1, 1, 9, 0),
        article_source=source,
        content=title,
        trading_code="600000",
        secu_abbr="浦发银行",
        event_name="业绩预告",
        event_impact_direction="正面",
    )


def test_group_builder_adds_explainability_fields():
    events = [_event("a1", "公司发布业绩预告"), _event("a2", "公司发布业绩预告 官方公告")]
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01]], dtype="float32")
    labels = np.array([0, 0])

    groups = GroupBuilder.build_groups(events, labels, embeddings)

    assert len(groups) == 1
    group = groups[0]
    assert group.representative_article_id == "a1"
    assert group.evidence_edges
    assert group.cluster_reason
    assert group.timeline_stages


def test_alert_engine_adds_structured_factors():
    base = GroupBuilder.build_groups(
        [_event("a1", "公司发布重大事项"), _event("a2", "公司重大事项获官方确认")],
        np.array([0, 0]),
        np.array([[1.0, 0.0], [0.99, 0.01]], dtype="float32"),
    )[0]
    scored = ScoredEventGroupSchema(
        **base.model_dump(),
        credibility_score=0.9,
        impact_score=0.9,
        impact_level="high",
        sentiment_score=0.5,
    )

    alerted = AlertEngine.apply(scored)

    assert alerted.alert_level == "P0"
    assert alerted.alert_factors["credibility_score"] == 0.9
    assert "可信度" in alerted.alert_reason


def test_b_algorithm_scorer_scores_main_event_group():
    base = GroupBuilder.build_groups(
        [_event("a1", "公司发布重大项目中标公告", "证券时报"), _event("a2", "公司项目获官方确认")],
        np.array([0, 0]),
        np.array([[1.0, 0.0], [0.99, 0.01]], dtype="float32"),
    )[0]

    scored = BAlgorithmScorer.score(base, classifier=None)

    assert scored.credibility_score is not None
    assert scored.impact_score is not None
    assert scored.b_match_method == "algorithm"
    assert scored.b_match_score == 1.0
    assert "credibility_source_authority" in scored.b_score_components
    assert "impact_event_materiality" in scored.b_score_components


def test_b_event_classifier_can_update_group_without_cluster_id():
    records = [
        {
            "article_id": "train-1",
            "dataset": "unit",
            "title": "浦发银行发布业绩预告",
            "content": "公司业绩预告增长",
            "text": "浦发银行发布业绩预告 公司业绩预告增长",
            "security_code": "600000",
            "security_name": "浦发银行",
            "industry_code": "",
            "industry_name": "",
            "event_name": "业绩预告",
            "event_emotion": "正面",
        },
        {
            "article_id": "train-2",
            "dataset": "unit",
            "title": "浦发银行收到监管处罚",
            "content": "公司收到监管处罚",
            "text": "浦发银行收到监管处罚 公司收到监管处罚",
            "security_code": "600000",
            "security_name": "浦发银行",
            "industry_code": "",
            "industry_name": "",
            "event_name": "监管处罚",
            "event_emotion": "负面",
        },
    ]
    classifier = BEventClassifier(records, [])
    base = GroupBuilder.build_groups(
        [_event("a1", "浦发银行发布业绩预告增长")],
        np.array([0]),
        np.array([[1.0, 0.0]], dtype="float32"),
    )[0]

    scored = BAlgorithmScorer.score(base, classifier=classifier)

    assert scored.event_name == "业绩预告"
    assert scored.b_score_components["classification_classifier"] == "b_issuer_title_character_nn"
    assert scored.b_score_components["classification_confidence"] > 0


def test_incremental_clusterer_assigns_existing_group():
    events = [_event("a1", "公司发布业绩预告"), _event("a2", "公司发布业绩预告 官方公告")]
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01]], dtype="float32")
    group = GroupBuilder.build_groups(events, np.array([0, 0]), embeddings)[0]

    assignment = IncrementalClusterer(threshold=0.9).assign(
        _event("a3", "公司发布业绩预告 后续报道"),
        np.array([0.98, 0.02], dtype="float32"),
        [group],
        ["a1", "a2"],
        embeddings,
    )

    assert assignment.matched_group_id == group.group_id
    assert not assignment.is_new_group


def test_run_manifest_rejects_sample_or_fallback_embeddings():
    manifest = build_run_manifest(
        source_type="a_extracted_json",
        source_path="sample.json",
        event_count=1000,
        embedding_backend="tfidf",
        with_content=1000,
    )

    assert not manifest["production_ready"]
    assert len(manifest["warnings"]) == 2


def test_run_manifest_accepts_full_a_bge_run():
    manifest = build_run_manifest(
        source_type="a_extracted_json",
        source_path="extracted_unlabeled_test.json",
        event_count=106387,
        embedding_backend="bge",
        with_content=106387,
    )

    assert manifest["production_ready"]


def test_default_b_classifier_training_excludes_official_test(monkeypatch):
    import src.services.b_event_classifier as classifier_module

    calls = []

    def fake_records(path, sheet, scope):
        calls.append((path.name, sheet, scope))
        return [{"event_name": "测试事件"}]

    classifier_module._load_training_records.cache_clear()
    monkeypatch.setattr(classifier_module, "_records_from_excel", fake_records)
    classifier_module._load_training_records()
    classifier_module._load_training_records.cache_clear()

    assert calls
    assert all("test" not in filename.lower() for filename, _, _ in calls)
