"""Evaluate sparse TF-IDF + LinearSVC event classifiers with taxonomy constraints.

Run with the workspace Anaconda interpreter, which provides scikit-learn:
    D:\\myanaconda\\python.exe event_radar/scripts/sklearn_event_experiments.py --scope company
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from event_extraction_baseline import DATA_DIR, allowed_labels, extract_entities, iter_jsonl, load_taxonomy, macro_f1
from event_model_experiments import stratified_split


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "event_radar" / "reports"

CONFIGURATIONS = {
    "title_char_svm": lambda record: record.get("title") or "",
    "headline_lead_char_svm": lambda record: f"{record.get('title') or ''} {record.get('title') or ''} {(record.get('content') or '')[:500]}",
}


def candidate_labels(record: dict[str, Any], scope: str, taxonomy: dict[str, dict[str, set[str]]], entities: dict[str, list[dict[str, str]]], classes: list[str]) -> list[str]:
    subjects = extract_entities(record, scope, entities)
    taxonomy_labels: set[str] = set()
    for subject in subjects:
        taxonomy_labels.update(taxonomy[scope].get(subject["code"], set()))
    constrained = sorted(taxonomy_labels & set(classes))
    return constrained or classes


def train_and_predict(train: list[dict[str, Any]], evaluation: list[dict[str, Any]], scope: str, text_builder: Any) -> list[str]:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=180_000, sublinear_tf=True, norm="l2")
    features = vectorizer.fit_transform([text_builder(record) for record in train])
    classifier = LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=20260727)
    classifier.fit(features, [record["event_name"] for record in train])
    scores = classifier.decision_function(vectorizer.transform([text_builder(record) for record in evaluation]))
    classes = [str(label) for label in classifier.classes_]
    taxonomy, entities = load_taxonomy()
    index_by_label = {label: index for index, label in enumerate(classes)}
    predictions = []
    for record, row in zip(evaluation, scores):
        allowed = candidate_labels(record, scope, taxonomy, entities, classes)
        predictions.append(max(allowed, key=lambda label: row[index_by_label[label]]))
    return predictions


def evaluate(records: list[dict[str, Any]], labels: list[str]) -> dict[str, float]:
    expected = [record["event_name"] for record in records]
    return {
        "accuracy": round(sum(left == right for left, right in zip(expected, labels)) / len(expected), 4),
        "macro_f1": round(macro_f1(expected, labels), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sparse SVM event-classification experiments.")
    parser.add_argument("--scope", choices=("company", "industry"), default="company")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = list(iter_jsonl(DATA_DIR / f"news_labeled_train_{args.scope}.jsonl.gz"))
    train, validation = stratified_split(records)
    validation_metrics = {}
    for name, builder in CONFIGURATIONS.items():
        validation_metrics[name] = evaluate(validation, train_and_predict(train, validation, args.scope, builder))
    selected = max(validation_metrics, key=lambda name: (validation_metrics[name]["macro_f1"], validation_metrics[name]["accuracy"]))
    labelled_test = list(iter_jsonl(DATA_DIR / f"news_labeled_test_{args.scope}.jsonl.gz"))
    diagnostic = {
        name: evaluate(labelled_test, train_and_predict(records, labelled_test, args.scope, builder))
        for name, builder in CONFIGURATIONS.items()
    }
    report = {
        "scope": args.scope,
        "training_records": len(train),
        "validation_records": len(validation),
        "method": "character TF-IDF (2-5 gram) + class-balanced LinearSVC + taxonomy-constrained candidate decoding",
        "selection_rule": "highest validation Macro-F1, then validation accuracy",
        "selected_model": selected,
        "validation": validation_metrics,
        "provided_test_diagnostic": diagnostic,
    }
    json_path = REPORT_DIR / f"sklearn_event_experiment_{args.scope}.json"
    markdown_path = REPORT_DIR / f"sklearn_event_experiment_{args.scope}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {args.scope} 事件分类 SVM 实验", "",
        "方法：字符 TF-IDF（2—5 gram）+ 类别平衡 LinearSVC + 事件体系候选约束。", "",
        "| 模型 | 验证 Accuracy | 验证 Macro-F1 | 提供测试集 Accuracy（仅诊断） | 提供测试集 Macro-F1（仅诊断） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in validation_metrics.items():
        test_metric = diagnostic[name]
        lines.append(f"| {name} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.2%} | {test_metric['accuracy']:.2%} | {test_metric['macro_f1']:.2%} |")
    lines.extend(["", f"验证集选中：`{selected}`。"])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
