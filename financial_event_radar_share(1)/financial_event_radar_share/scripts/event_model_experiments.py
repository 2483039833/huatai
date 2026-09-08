"""Compare event-classification strategies on a stratified validation split.

The selected model is based on a split of the provided training data.  The
labelled competition test set is reported only as a secondary diagnostic.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from event_extraction_baseline import (
    DATA_DIR,
    TaxonomyNaiveBayes,
    allowed_labels,
    char_ngrams,
    extract_entities,
    iter_jsonl,
    load_taxonomy,
    macro_f1,
    model_text,
    title_model_record,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "event_radar" / "reports"


def stratified_split(records: list[dict[str, Any]], validation_every: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["event_name"]].append(record)
    train, validation = [], []
    for label_records in by_label.values():
        for index, record in enumerate(label_records):
            (validation if len(label_records) > 2 and index % validation_every == 0 else train).append(record)
    return train, validation


class EntityNearestNeighbor:
    """A title-similarity fallback restricted to the same issuer when possible."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.by_entity: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
        self.global_samples: list[tuple[str, set[str]]] = []
        for record in records:
            features = set(char_ngrams(record.get("title") or ""))
            sample = (record["event_name"], features)
            self.global_samples.append(sample)
            if record.get("security_code"):
                self.by_entity[record["security_code"]].append(sample)

    def predict(self, record: dict[str, Any], candidates: list[str]) -> str:
        samples = self.by_entity.get(record.get("security_code") or "", self.global_samples)
        target = set(char_ngrams(record.get("title") or ""))
        best_label, best_score = candidates[0], -1.0
        for label, features in samples:
            if label not in candidates or not features or not target:
                continue
            score = len(features & target) / min(len(features), len(target))
            if score > best_score:
                best_label, best_score = label, score
        return best_label

    def score(self, record: dict[str, Any], candidates: list[str]) -> dict[str, float]:
        target = set(char_ngrams(record.get("title") or ""))
        samples = self.by_entity.get(record.get("security_code") or "", self.global_samples)
        scores = {label: 0.0 for label in candidates}
        for label, features in samples:
            if label in scores and target and features:
                scores[label] = max(scores[label], len(features & target) / min(len(features), len(target)))
        return scores


def high_precision_title_rule(title: str, candidates: list[str]) -> str | None:
    """Return a candidate only for explicit business-action wording in a headline."""
    rules = (
        (("战略调整", "战略转型", "战略布局", "发展规划", "全球化布局"), ("战略",)),
        (("新品发布", "新产品", "产品上市", "全新车型"), ("产品", "创新")),
        (("核心技术突破", "关键技术突破", "研发突破", "专利突破"), ("突破", "研发")),
        (("重大订单", "中标", "签署合同", "签约"), ("业务", "订单", "项目")),
    )
    for triggers, label_terms in rules:
        if any(trigger in title for trigger in triggers):
            for candidate in candidates:
                if any(term in candidate for term in label_terms):
                    return candidate
    return None


def predictions(train: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> dict[str, list[str]]:
    events_by_entity, entities = load_taxonomy()
    definition_features: dict[tuple[str, str, str], set[str]] = {}
    for line in (DATA_DIR / "event_taxonomy.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        definition_features[(row["scope"], row["entity_code"], row["event_name"])] = set(char_ngrams(f"{row['event_name']} {row['definition']}"))
    full_model = TaxonomyNaiveBayes()
    full_model.fit(train)
    title_model = TaxonomyNaiveBayes()
    title_model.fit(title_model_record(record) for record in train)
    nearest = EntityNearestNeighbor(train)
    outcomes = {"title_nb": [], "body_nb": [], "title_nn": [], "taxonomy_definition": [], "nn_definition_fallback": [], "nn_title_fallback_001": [], "nn_title_fallback_005": [], "nn_title_fallback_010": [], "nn_title_fallback_015": [], "nn_rule_low_conf": [], "nn_rule_always": [], "voting_ensemble": []}
    for record in evaluation:
        candidates = allowed_labels("company", extract_entities(record, "company", entities), events_by_entity, full_model)
        title_label = max(title_model.score(model_text(title_model_record(record)), candidates).items(), key=lambda item: item[1])[0]
        body_label = max(full_model.score(model_text(record), candidates).items(), key=lambda item: item[1])[0]
        nearest_label = nearest.predict(record, candidates)
        nearest_scores = nearest.score(record, candidates)
        title_features = set(char_ngrams(record.get("title") or ""))
        subjects = extract_entities(record, "company", entities)
        definition_scores = {
            label: max(
                (len(title_features & definition_features.get(("company", subject["code"], label), set())) / min(len(title_features), len(definition_features.get(("company", subject["code"], label), set())) or 1)
                 for subject in subjects),
                default=0.0,
            )
            for label in candidates
        }
        definition_label = max(definition_scores.items(), key=lambda item: item[1])[0]
        fallback_label = definition_label if max(nearest_scores.values(), default=0) < 0.08 else nearest_label
        nearest_confidence = max(nearest_scores.values(), default=0)
        rule_label = high_precision_title_rule(record.get("title") or "", candidates)
        votes = [title_label, body_label, nearest_label, definition_label]
        ensemble = next((label for label in (nearest_label, title_label, body_label, definition_label) if votes.count(label) >= 2), nearest_label)
        outcomes["title_nb"].append(title_label)
        outcomes["body_nb"].append(body_label)
        outcomes["title_nn"].append(nearest_label)
        outcomes["taxonomy_definition"].append(definition_label)
        outcomes["nn_definition_fallback"].append(fallback_label)
        outcomes["nn_title_fallback_001"].append(title_label if nearest_confidence < 0.01 else nearest_label)
        outcomes["nn_title_fallback_005"].append(title_label if nearest_confidence < 0.05 else nearest_label)
        outcomes["nn_title_fallback_010"].append(title_label if nearest_confidence < 0.10 else nearest_label)
        outcomes["nn_title_fallback_015"].append(title_label if nearest_confidence < 0.15 else nearest_label)
        outcomes["nn_rule_low_conf"].append(rule_label if rule_label and nearest_confidence < 0.12 else nearest_label)
        outcomes["nn_rule_always"].append(rule_label or nearest_label)
        outcomes["voting_ensemble"].append(ensemble)
    return outcomes


def score(records: list[dict[str, Any]], results: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    expected = [record["event_name"] for record in records]
    return {
        name: {
            "accuracy": round(sum(left == right for left, right in zip(expected, labels)) / len(expected), 4),
            "macro_f1": round(macro_f1(expected, labels), 4),
        }
        for name, labels in results.items()
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = list(iter_jsonl(DATA_DIR / "news_labeled_train_company.jsonl.gz"))
    train, validation = stratified_split(records)
    validation_scores = score(validation, predictions(train, validation))
    selected = max(validation_scores, key=lambda name: (validation_scores[name]["macro_f1"], validation_scores[name]["accuracy"]))
    # The supplied test labels are a diagnostic only, never used to select `selected`.
    labelled_test = list(iter_jsonl(DATA_DIR / "news_labeled_test_company.jsonl.gz"))
    test_scores = score(labelled_test, predictions(records, labelled_test))
    report = {
        "training_records": len(train),
        "validation_records": len(validation),
        "selection_rule": "highest validation Macro-F1, then validation accuracy",
        "selected_model": selected,
        "validation": validation_scores,
        "provided_test_diagnostic": test_scores,
    }
    (REPORT_DIR / "event_model_experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 个股事件分类模型实验", "",
        f"训练/验证样本：{len(train)} / {len(validation)}。模型选择规则：验证集 Macro-F1 优先，其次 Accuracy。", "",
        "| 模型 | 验证 Accuracy | 验证 Macro-F1 | 提供测试集 Accuracy（仅诊断） | 提供测试集 Macro-F1（仅诊断） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in validation_scores:
        lines.append(f"| {name} | {validation_scores[name]['accuracy']:.2%} | {validation_scores[name]['macro_f1']:.2%} | {test_scores[name]['accuracy']:.2%} | {test_scores[name]['macro_f1']:.2%} |")
    lines.extend(["", f"验证集选中模型：`{selected}`。"])
    (REPORT_DIR / "event_model_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
