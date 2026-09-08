"""Generate production artifacts using the validation-selected semantic hybrid.

Run this script with the workspace Anaconda interpreter.  It preserves the
existing output schema so credibility, timeline and alert scripts consume the
upgraded event labels without interface changes.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gensim.models import Word2Vec

import event_extraction_baseline as base
from gensim_semantic_retrieval_experiment import SemanticEntityNearestNeighbor


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "event_radar" / "models" / "company_title_char_w2v.model"


class SemanticHybridNearestNeighbor:
    """Validation-selected 0.1 character overlap + 0.9 semantic similarity."""

    classifier_name = "issuer_title_semantic_hybrid"

    def __init__(self, records: list[dict[str, Any]], model: Word2Vec) -> None:
        self.character = base.EntityTitleNearestNeighbor(records)
        self.semantic = SemanticEntityNearestNeighbor(records, model)

    def score(self, record: dict[str, Any], candidates: list[str]) -> dict[str, float]:
        character_scores = self.character.score(record, candidates)
        semantic_scores = self.semantic.score(record, candidates)
        return {
            label: 0.1 * character_scores[label] + 0.9 * semantic_scores[label]
            for label in candidates
        }

    def predict(self, record: dict[str, Any], candidates: list[str]) -> str:
        scores = self.score(record, candidates)
        return max(candidates, key=lambda label: scores[label])

    @staticmethod
    def confidence_for_margin(margin: float) -> float:
        """Conservative buckets derived from the held-out training validation split."""
        if margin >= 0.08:
            return 0.95
        if margin >= 0.05:
            return 0.94
        if margin >= 0.03:
            return 0.92
        if margin >= 0.02:
            return 0.90
        if margin >= 0.01:
            return 0.88
        if margin >= 0.005:
            return 0.80
        return 0.60

    @staticmethod
    def review_required_for_margin(margin: float) -> bool:
        return margin < 0.01


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as target:
        for prediction in predictions:
            target.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing semantic checkpoint: {MODEL_PATH}")
    base.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy, entities = base.load_taxonomy()
    semantic_vectors = Word2Vec.load(str(MODEL_PATH))
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "taxonomy-constrained event recognition; company labels use 0.1 title character retrieval + 0.9 locally pretrained semantic retrieval, industry labels use character n-gram NB",
        "semantic_embedding": {"vector_size": semantic_vectors.vector_size, "vocabulary": len(semantic_vectors.wv)},
        "scopes": {},
    }

    for scope in ("company", "industry"):
        print(f"Generating {scope} predictions...", flush=True)
        training = list(base.iter_jsonl(base.DATA_DIR / base.TRAIN_DATASETS[scope]))
        model = base.TaxonomyNaiveBayes()
        model.fit(training)
        direction_model = base.TaxonomyNaiveBayes()
        direction_training = []
        for record in training:
            labelled = dict(record)
            labelled["event_name"] = record["event_emotion"]
            direction_training.append(labelled)
        direction_model.fit(direction_training)
        nearest = SemanticHybridNearestNeighbor(training, semantic_vectors) if scope == "company" else None

        predictions = [
            base.predict_record(record, scope, model, model, direction_model, nearest, taxonomy, entities)
            for record in base.iter_jsonl(base.DATA_DIR / base.TEST_DATASETS[scope])
        ]
        prediction_path = base.DERIVED_DIR / f"event_predictions_test_{scope}.jsonl.gz"
        write_predictions(prediction_path, predictions)
        duplicate_predictions = [
            base.predict_record(record, scope, model, model, direction_model, nearest, taxonomy, entities)
            for record in base.iter_jsonl(base.DATA_DIR / base.DUPLICATE_TEST_DATASETS[scope])
        ]
        duplicate_path = base.DERIVED_DIR / f"event_predictions_duplicates_test_{scope}.jsonl.gz"
        write_predictions(duplicate_path, duplicate_predictions)
        metrics = base.evaluate(scope, predictions)
        metrics.update({
            "training_records": model.document_count,
            "trained_event_types": len(model.labels),
            "event_classifier": nearest.classifier_name if nearest else "body_character_nb",
            "direction_classifier": "supervised_character_nb",
            "prediction_file": prediction_path.name,
            "timeline_input_prediction_file": duplicate_path.name,
        })
        report["scopes"][scope] = metrics

    (base.REPORT_DIR / "event_extraction_baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 事件识别优化报告", "",
        "个股事件采用验证集选中的语义检索融合模型；行业事件保留表现稳定的字符 n-gram 朴素贝叶斯。", "",
        "| 范围 | 测试记录 | 事件准确率 | 事件 Macro-F1 | 方向准确率 | 实体覆盖率 |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, metrics in report["scopes"].items():
        direction = f"{metrics['direction_accuracy']:.2%}" if metrics["direction_accuracy"] is not None else "N/A"
        lines.append(f"| {scope} | {metrics['records']} | {metrics['event_accuracy']:.2%} | {metrics['event_macro_f1']:.2%} | {direction} | {metrics['entity_field_coverage']:.2%} |")
    lines.append("")
    lines.append("个股低边际预测（< 0.01）将进入人工复核队列。")
    (base.REPORT_DIR / "event_extraction_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
