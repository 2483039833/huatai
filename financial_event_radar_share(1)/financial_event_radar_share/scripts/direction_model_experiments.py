"""Cross-validate class-balanced impact-direction classifiers.

Direction labels are strongly imbalanced, so accuracy alone hides weak negative
and neutral recall.  This script compares a generative NB baseline with an
IDF-weighted class centroid classifier, lexical rules and a voting ensemble.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from event_extraction_baseline import (
    DATA_DIR,
    TaxonomyNaiveBayes,
    char_ngrams,
    extract_direction,
    iter_jsonl,
    macro_f1,
    model_text,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "event_radar" / "reports"
STANDARD_TO_LABEL = {"positive": "正面", "negative": "负面", "neutral": "中性"}


def direction_record(record: dict[str, Any]) -> dict[str, Any]:
    """Direction is usually expressed in the headline and lead paragraph."""
    compact = dict(record)
    compact["content"] = (record.get("content") or "")[:260]
    return compact


class IdfCentroidDirection:
    """Represent each direction by the average IDF-weighted article vector."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.centroids: dict[str, Counter[str]] = defaultdict(Counter)
        self.class_count: Counter[str] = Counter()

    def fit(self, records: list[dict[str, Any]]) -> None:
        document_frequency: Counter[str] = Counter()
        raw_features: list[tuple[str, Counter[str]]] = []
        for record in records:
            features = char_ngrams(model_text(record))
            raw_features.append((record["event_emotion"], features))
            document_frequency.update(features.keys())
        document_count = len(records)
        self.idf = {feature: math.log((document_count + 1) / (count + 1)) + 1 for feature, count in document_frequency.items()}
        for label, features in raw_features:
            total = sum(features.values()) or 1
            self.class_count[label] += 1
            for feature, count in features.items():
                self.centroids[label][feature] += count / total * self.idf[feature]
        for label in self.centroids:
            for feature in self.centroids[label]:
                self.centroids[label][feature] /= self.class_count[label]

    def score(self, record: dict[str, Any]) -> dict[str, float]:
        features = char_ngrams(model_text(record))
        total = sum(features.values()) or 1
        weighted = {feature: count / total * self.idf.get(feature, 0) for feature, count in features.items()}
        return {label: sum(weighted.get(feature, 0) * value for feature, value in centroid.items()) for label, centroid in self.centroids.items()}


def folds(records: list[dict[str, Any]], n_splits: int = 2) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["event_emotion"]].append(record)
    result = []
    for fold in range(n_splits):
        train, validation = [], []
        for group in buckets.values():
            for index, record in enumerate(group):
                (validation if index % n_splits == fold else train).append(record)
        result.append((train, validation))
    return result


def evaluate_fold(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, tuple[list[str], list[str]]]:
    compact_train = [direction_record(record) for record in train]
    compact_validation = [direction_record(record) for record in validation]
    nb = TaxonomyNaiveBayes()
    nb_records = []
    for record in compact_train:
        labelled = dict(record)
        labelled["event_name"] = record["event_emotion"]
        nb_records.append(labelled)
    nb.fit(nb_records)
    centroid = IdfCentroidDirection()
    centroid.fit(compact_train)
    expected = [record["event_emotion"] for record in validation]
    predictions = {"character_nb": [], "idf_centroid": [], "keyword_rules": [], "voting_ensemble": []}
    for record, compact in zip(validation, compact_validation):
        nb_label = max(nb.score(model_text(compact), nb.labels).items(), key=lambda item: item[1])[0]
        centroid_label = max(centroid.score(compact).items(), key=lambda item: item[1])[0]
        keyword_label = STANDARD_TO_LABEL[extract_direction(record.get("text") or "")[0]]
        votes = [nb_label, centroid_label, keyword_label]
        ensemble_label = next((label for label in (centroid_label, nb_label, keyword_label) if votes.count(label) >= 2), centroid_label)
        predictions["character_nb"].append(nb_label)
        predictions["idf_centroid"].append(centroid_label)
        predictions["keyword_rules"].append(keyword_label)
        predictions["voting_ensemble"].append(ensemble_label)
    return {name: (expected, values) for name, values in predictions.items()}


def aggregate(fold_results: list[dict[str, tuple[list[str], list[str]]]]) -> dict[str, dict[str, float]]:
    totals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for result in fold_results:
        for name, (expected, predicted) in result.items():
            totals[name].append((sum(left == right for left, right in zip(expected, predicted)) / len(expected), macro_f1(expected, predicted)))
    return {
        name: {
            "accuracy_mean": round(sum(item[0] for item in values) / len(values), 4),
            "macro_f1_mean": round(sum(item[1] for item in values) / len(values), 4),
        }
        for name, values in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate class-balanced direction classifiers.")
    parser.add_argument("--scope", choices=("company", "industry"), help="Run one scope to keep experiments lightweight.")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"selection_rule": "highest 2-fold macro F1, then accuracy", "scopes": {}}
    for scope in ((args.scope,) if args.scope else ("company", "industry")):
        records = list(iter_jsonl(DATA_DIR / f"news_labeled_train_{scope}.jsonl.gz"))
        scores = aggregate([evaluate_fold(train, validation) for train, validation in folds(records)])
        selected = max(scores, key=lambda name: (scores[name]["macro_f1_mean"], scores[name]["accuracy_mean"]))
        report["scopes"][scope] = {"records": len(records), "models": scores, "selected_model": selected}
    (REPORT_DIR / "direction_model_experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 影响方向模型交叉验证", "", "选择规则：2 折分层交叉验证 Macro-F1 优先，其次 Accuracy。", ""]
    for scope, result in report["scopes"].items():
        lines.extend([f"## {scope}", "", "| 模型 | Accuracy | Macro-F1 |", "| --- | ---: | ---: |"])
        for name, metric in result["models"].items():
            lines.append(f"| {name} | {metric['accuracy_mean']:.2%} | {metric['macro_f1_mean']:.2%} |")
        lines.extend(["", f"选中：`{result['selected_model']}`。", ""])
    (REPORT_DIR / "direction_model_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
