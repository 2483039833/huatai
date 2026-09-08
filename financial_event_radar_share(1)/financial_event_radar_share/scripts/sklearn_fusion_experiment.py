"""Validate confidence-aware fusion of LinearSVC and issuer-history retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from event_extraction_baseline import DATA_DIR, EntityTitleNearestNeighbor, allowed_labels, extract_entities, iter_jsonl, load_taxonomy, macro_f1
from event_model_experiments import stratified_split


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "event_radar" / "reports"


def text(record: dict[str, Any]) -> str:
    return f"{record.get('title') or ''} {record.get('title') or ''} {(record.get('content') or '')[:500]}"


def candidates(record: dict[str, Any], taxonomy: dict[str, dict[str, set[str]]], entities: dict[str, list[dict[str, str]]], classes: list[str]) -> list[str]:
    labels: set[str] = set()
    for subject in extract_entities(record, "company", entities):
        labels.update(taxonomy["company"].get(subject["code"], set()))
    return sorted(labels & set(classes)) or classes


def predictor(train: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> tuple[list[str], list[float], list[str], list[float]]:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=180_000, sublinear_tf=True, norm="l2")
    classifier = LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=20260727)
    classifier.fit(vectorizer.fit_transform([text(record) for record in train]), [record["event_name"] for record in train])
    raw_scores = classifier.decision_function(vectorizer.transform([text(record) for record in evaluation]))
    classes = [str(label) for label in classifier.classes_]
    class_index = {label: index for index, label in enumerate(classes)}
    taxonomy, entities = load_taxonomy()
    nearest = EntityTitleNearestNeighbor(train)
    svm_labels, svm_margins, nn_labels, nn_margins = [], [], [], []
    for record, scores in zip(evaluation, raw_scores):
        allowed = candidates(record, taxonomy, entities, classes)
        ordered = sorted(((float(scores[class_index[label]]), label) for label in allowed), reverse=True)
        svm_labels.append(ordered[0][1])
        svm_margins.append(ordered[0][0] - (ordered[1][0] if len(ordered) > 1 else ordered[0][0] - 1))
        nearest_scores = nearest.score(record, allowed)
        nn_labels.append(nearest.predict(record, allowed))
        ranking = sorted(nearest_scores.values(), reverse=True)
        nn_margins.append(ranking[0] - (ranking[1] if len(ranking) > 1 else 0.0))
    return svm_labels, svm_margins, nn_labels, nn_margins


def metrics(records: list[dict[str, Any]], labels: list[str]) -> dict[str, float]:
    actual = [record["event_name"] for record in records]
    return {"accuracy": round(sum(left == right for left, right in zip(actual, labels)) / len(actual), 4), "macro_f1": round(macro_f1(actual, labels), 4)}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = list(iter_jsonl(DATA_DIR / "news_labeled_train_company.jsonl.gz"))
    train, validation = stratified_split(records)
    svm_labels, svm_margins, nn_labels, _ = predictor(train, validation)
    thresholds = sorted({float(np.quantile(svm_margins, quantile)) for quantile in np.linspace(0, 1, 11)})
    validation_scores = {"svm": metrics(validation, svm_labels), "nearest_neighbor": metrics(validation, nn_labels)}
    for threshold in thresholds:
        validation_scores[f"svm_if_margin_ge_{threshold:.3f}"] = metrics(validation, [svm if margin >= threshold else nn for svm, nn, margin in zip(svm_labels, nn_labels, svm_margins)])
    selected = max(validation_scores, key=lambda name: (validation_scores[name]["macro_f1"], validation_scores[name]["accuracy"]))
    selected_threshold = None if selected in {"svm", "nearest_neighbor"} else float(selected.rsplit("_", 1)[1])

    test = list(iter_jsonl(DATA_DIR / "news_labeled_test_company.jsonl.gz"))
    test_svm, test_margins, test_nn, _ = predictor(records, test)
    test_labels = test_svm if selected == "svm" else test_nn if selected == "nearest_neighbor" else [svm if margin >= selected_threshold else nn for svm, nn, margin in zip(test_svm, test_nn, test_margins)]
    report = {
        "training_records": len(train),
        "validation_records": len(validation),
        "selection_rule": "highest validation Macro-F1, then accuracy",
        "selected_strategy": selected,
        "selected_svm_margin": selected_threshold,
        "validation": validation_scores,
        "provided_test_diagnostic": metrics(test, test_labels),
    }
    (REPORT_DIR / "sklearn_fusion_experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# SVM 与历史近邻置信度融合实验", "",
        "仅在训练集分层验证上选择 SVM 分数差阈值；提供测试集仅用于最终诊断。", "",
        f"选中策略：`{selected}`。", "",
        f"提供测试集诊断：Accuracy `{report['provided_test_diagnostic']['accuracy']:.2%}`，Macro-F1 `{report['provided_test_diagnostic']['macro_f1']:.2%}`。",
    ]
    (REPORT_DIR / "sklearn_fusion_experiment.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
