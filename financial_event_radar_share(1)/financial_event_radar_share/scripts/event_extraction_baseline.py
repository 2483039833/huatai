"""Train and evaluate a dependency-free event-recognition baseline.

This baseline is intentionally transparent: taxonomy-constrained character
n-gram Naive Bayes predicts the event type, while deterministic extractors
produce entities, event time, sentiment direction and supporting evidence.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "event_radar" / "data" / "processed"
DERIVED_DIR = ROOT / "event_radar" / "data" / "derived"
REPORT_DIR = ROOT / "event_radar" / "reports"
TAXONOMY_PATH = DATA_DIR / "event_taxonomy.jsonl"

TRAIN_DATASETS = {
    "company": "news_labeled_train_company.jsonl.gz",
    "industry": "news_labeled_train_industry.jsonl.gz",
}
TEST_DATASETS = {
    "company": "news_labeled_test_company.jsonl.gz",
    "industry": "news_labeled_test_industry.jsonl.gz",
}
DUPLICATE_TEST_DATASETS = {
    "company": "duplicates_test_company.jsonl.gz",
    "industry": "duplicates_test_industry.jsonl.gz",
}

DATE_PATTERNS = (
    re.compile(r"(?P<year>20\d{2})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})"),
    re.compile(r"(?P<year>20\d{2})年(?P<month>\d{1,2})月(?:(?P<day>\d{1,2})日?)?"),
)
SENTENCE_SPLIT = re.compile(r"[。！？；\n]+")
POSITIVE_WORDS = ("增长", "突破", "中标", "签约", "获批", "盈利", "增持", "创新", "合作", "投产", "交付", "上调")
NEGATIVE_WORDS = ("下滑", "亏损", "减持", "处罚", "调查", "违约", "风险", "终止", "召回", "裁员", "下调", "澄清")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def model_text(record: dict[str, Any]) -> str:
    """Limit body length so a single long article cannot dominate the model."""
    title = record.get("title") or ""
    content = record.get("content") or ""
    return f"{title} {title} {content[:1200]}"


def title_model_record(record: dict[str, Any]) -> dict[str, Any]:
    """Create a copy for the title-focused classifier used on company events."""
    title_only = dict(record)
    title_only["content"] = ""
    return title_only


def char_ngrams(text: str) -> Counter[str]:
    compact = re.sub(r"\s+", "", text)
    features: Counter[str] = Counter()
    for size in (2, 3):
        features.update(compact[index : index + size] for index in range(len(compact) - size + 1))
    return features


class TaxonomyNaiveBayes:
    """Multinomial NB with a small, auditable Chinese character n-gram feature set."""

    def __init__(self) -> None:
        self.class_documents: Counter[str] = Counter()
        self.class_tokens: dict[str, Counter[str]] = defaultdict(Counter)
        self.class_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.document_count = 0

    def fit(self, records: Iterable[dict[str, Any]]) -> None:
        raw_counts: Counter[str] = Counter()
        cached: list[tuple[str, Counter[str]]] = []
        for record in records:
            label = record.get("event_name")
            if not label:
                continue
            features = char_ngrams(model_text(record))
            cached.append((label, features))
            self.class_documents[label] += 1
            self.document_count += 1
            raw_counts.update(features)
        self.vocabulary = {feature for feature, count in raw_counts.items() if count >= 2}
        for label, features in cached:
            filtered = Counter({feature: count for feature, count in features.items() if feature in self.vocabulary})
            self.class_tokens[label].update(filtered)
            self.class_totals[label] += sum(filtered.values())

    @property
    def labels(self) -> list[str]:
        return sorted(self.class_documents)

    def score(self, text: str, candidates: list[str]) -> dict[str, float]:
        features = Counter({key: value for key, value in char_ngrams(text).items() if key in self.vocabulary})
        vocabulary_size = max(len(self.vocabulary), 1)
        scores: dict[str, float] = {}
        for label in candidates:
            prior = (self.class_documents[label] + 1) / (self.document_count + len(self.labels))
            denominator = self.class_totals[label] + vocabulary_size
            score = math.log(prior)
            for feature, count in features.items():
                score += count * math.log((self.class_tokens[label][feature] + 1) / denominator)
            scores[label] = score
        return scores


class EntityTitleNearestNeighbor:
    """Use prior labelled titles for the same issuer as a fine-grained type signal."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.by_entity: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
        self.global_samples: list[tuple[str, set[str]]] = []
        for record in records:
            sample = (record["event_name"], set(char_ngrams(record.get("title") or "")))
            self.global_samples.append(sample)
            if record.get("security_code"):
                self.by_entity[record["security_code"]].append(sample)

    def score(self, record: dict[str, Any], candidates: list[str]) -> dict[str, float]:
        target = set(char_ngrams(record.get("title") or ""))
        samples = self.by_entity.get(record.get("security_code") or "", self.global_samples)
        scores = {label: 0.0 for label in candidates}
        for label, features in samples:
            if label in scores and target and features:
                scores[label] = max(scores[label], len(target & features) / min(len(target), len(features)))
        return scores

    def predict(self, record: dict[str, Any], candidates: list[str]) -> str:
        """Return the first best matching labelled title; preserves stable tie-breaking."""
        target = set(char_ngrams(record.get("title") or ""))
        samples = self.by_entity.get(record.get("security_code") or "", self.global_samples)
        best_label, best_score = candidates[0], -1.0
        for label, features in samples:
            if label not in candidates or not target or not features:
                continue
            score = len(target & features) / min(len(target), len(features))
            if score > best_score:
                best_label, best_score = label, score
        return best_label


def load_taxonomy() -> tuple[dict[str, dict[str, set[str]]], dict[str, list[dict[str, str]]]]:
    events_by_entity: dict[str, dict[str, set[str]]] = {"company": defaultdict(set), "industry": defaultdict(set)}
    entities: dict[str, list[dict[str, str]]] = {"company": [], "industry": []}
    for line in TAXONOMY_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        scope = row["scope"]
        events_by_entity[scope][row["entity_code"]].add(row["event_name"])
        entities[scope].append({"code": row["entity_code"], "name": row["entity_name"]})
    for scope in entities:
        entities[scope] = list({(entity["code"], entity["name"]): entity for entity in entities[scope]}.values())
    return events_by_entity, entities


def extract_entities(record: dict[str, Any], scope: str, entity_catalog: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    primary_code = record.get("security_code") if scope == "company" else record.get("industry_code")
    primary_name = record.get("security_name") if scope == "company" else record.get("industry_name")
    if primary_code or primary_name:
        found.append({"scope": scope, "code": primary_code or "", "name": primary_name or "", "match": "source_field"})
    text = record.get("text") or ""
    for entity in entity_catalog[scope]:
        name = entity["name"]
        if name and name in text and not any(item["code"] == entity["code"] for item in found):
            found.append({"scope": scope, "code": entity["code"], "name": name, "match": "text"})
        if len(found) >= 5:
            break
    return found


def extract_event_time(text: str, published_at: str | None) -> dict[str, str]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            parts = match.groupdict()
            day = parts.get("day") or "01"
            return {"value": f"{parts['year']}-{int(parts['month']):02d}-{int(day):02d}", "source": "article_text"}
    return {"value": (published_at or "")[:10], "source": "publication_time"}


def extract_direction(text: str) -> tuple[str, dict[str, int]]:
    positive = sum(text.count(word) for word in POSITIVE_WORDS)
    negative = sum(text.count(word) for word in NEGATIVE_WORDS)
    if positive > negative:
        return "positive", {"positive": positive, "negative": negative}
    if negative > positive:
        return "negative", {"positive": positive, "negative": negative}
    return "neutral", {"positive": positive, "negative": negative}


def extract_evidence(text: str, event_name: str, entities: list[dict[str, str]]) -> list[str]:
    sentences = [item.strip() for item in SENTENCE_SPLIT.split(text) if len(item.strip()) >= 12]
    event_terms = {event_name[index : index + 2] for index in range(max(len(event_name) - 1, 0))}
    entity_names = {item["name"] for item in entities if item["name"]}

    def score(sentence: str) -> int:
        return (
            3 * sum(term in sentence for term in event_terms)
            + 2 * sum(name in sentence for name in entity_names)
            + sum(word in sentence for word in POSITIVE_WORDS + NEGATIVE_WORDS)
        )

    ranked = sorted(((score(sentence), index, sentence) for index, sentence in enumerate(sentences)), reverse=True)
    evidence = [sentence[:300] for value, _, sentence in ranked if value > 0][:2]
    return evidence or ([sentences[0][:300]] if sentences else [])


def allowed_labels(scope: str, entities: list[dict[str, str]], events_by_entity: dict[str, dict[str, set[str]]], model: TaxonomyNaiveBayes) -> list[str]:
    taxonomy_labels: set[str] = set()
    for entity in entities:
        taxonomy_labels.update(events_by_entity[scope].get(entity["code"], set()))
    trained = set(model.labels)
    constrained = sorted(taxonomy_labels & trained)
    return constrained or model.labels


def predict_record(
    record: dict[str, Any], scope: str, model: TaxonomyNaiveBayes, title_model: TaxonomyNaiveBayes,
    direction_model: TaxonomyNaiveBayes, nearest_model: EntityTitleNearestNeighbor | None,
    events_by_entity: dict[str, dict[str, set[str]],],
    entity_catalog: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    entities = extract_entities(record, scope, entity_catalog)
    candidates = allowed_labels(scope, entities, events_by_entity, model)
    classifier_name = "body_character_nb"
    nearest_prediction: str | None = None
    if scope == "company" and nearest_model:
        scores = nearest_model.score(record, candidates)
        classifier_name = getattr(nearest_model, "classifier_name", "entity_title_nearest_neighbor")
        nearest_prediction = nearest_model.predict(record, candidates)
    else:
        scores = model.score(model_text(record), candidates)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    event_name, top_score = ranked[0]
    if nearest_prediction is not None:
        event_name = nearest_prediction
        top_score = scores[event_name]
    second_score = max((score for label, score in scores.items() if label != event_name), default=top_score - 2)
    margin = top_score - second_score
    if scope == "company":
        calibrated_confidence = getattr(nearest_model, "confidence_for_margin", None) if nearest_model else None
        confidence = calibrated_confidence(margin) if calibrated_confidence else (
            0.95 if margin >= 0.15 else 0.90 if margin >= 0.10 else 0.82 if margin >= 0.05
            else 0.75 if margin >= 0.03 else 0.55
        )
    else:
        confidence = round(max(0.35, min(0.98, 0.55 + 0.12 * margin)), 3)
    keyword_direction, direction_signals = extract_direction(record.get("text") or "")
    direction_scores = direction_model.score(model_text(record), direction_model.labels)
    raw_direction = max(direction_scores.items(), key=lambda item: item[1])[0]
    direction = {"正面": "positive", "负面": "negative", "中性": "neutral"}.get(raw_direction, keyword_direction)
    review_policy = getattr(nearest_model, "review_required_for_margin", None) if nearest_model else None
    return {
        "article_id": record["article_id"],
        "dataset": record["dataset"],
        "event": {
            "scope": scope,
            "event_name": event_name,
            "confidence": confidence,
            "subjects": entities,
            "event_time": extract_event_time(record.get("text") or "", record.get("published_at")),
            "impact_direction": direction,
            "evidence": extract_evidence(record.get("text") or "", event_name, entities),
        },
        "decision_trace": {
            "candidate_count": len(candidates),
            "candidate_source": "entity_taxonomy" if len(candidates) < len(model.labels) else "model_label_space",
            "event_classifier": classifier_name,
            "top_score_margin": round(margin, 3),
            "review_required": bool(review_policy(margin)) if review_policy else scope == "company" and margin < 0.03,
            "direction_signals": direction_signals,
            "direction_classifier": "supervised_character_nb",
        },
        "reference": {
            "event_name": record.get("event_name"),
            "impact_direction": record.get("event_emotion"),
        },
    }


def macro_f1(true_labels: list[str], predicted_labels: list[str]) -> float:
    labels = set(true_labels) | set(predicted_labels)
    scores = []
    for label in labels:
        tp = sum(true == label and predicted == label for true, predicted in zip(true_labels, predicted_labels))
        fp = sum(true != label and predicted == label for true, predicted in zip(true_labels, predicted_labels))
        fn = sum(true == label and predicted != label for true, predicted in zip(true_labels, predicted_labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def reference_direction(value: str | None) -> str | None:
    return {"正面": "positive", "负面": "negative", "中性": "neutral"}.get(value or "")


def evaluate(scope: str, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    true_events = [item["reference"]["event_name"] for item in predictions]
    predicted_events = [item["event"]["event_name"] for item in predictions]
    correct = sum(true == predicted for true, predicted in zip(true_events, predicted_events))
    direction_pairs = [
        (reference_direction(item["reference"]["impact_direction"]), item["event"]["impact_direction"])
        for item in predictions if reference_direction(item["reference"]["impact_direction"])
    ]
    return {
        "scope": scope,
        "records": len(predictions),
        "event_accuracy": round(correct / len(predictions), 4) if predictions else 0.0,
        "event_macro_f1": round(macro_f1(true_events, predicted_events), 4),
        "direction_accuracy": round(sum(true == predicted for true, predicted in direction_pairs) / len(direction_pairs), 4) if direction_pairs else None,
        "direction_evaluated_records": len(direction_pairs),
        "entity_field_coverage": round(sum(bool(item["event"]["subjects"]) for item in predictions) / len(predictions), 4) if predictions else 0.0,
    }


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    events_by_entity, entity_catalog = load_taxonomy()
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "taxonomy-constrained character n-gram Naive Bayes + deterministic slot extraction",
        "scopes": {},
    }

    for scope in ("company", "industry"):
        print(f"Training and evaluating {scope} events...", flush=True)
        training_records = list(iter_jsonl(DATA_DIR / TRAIN_DATASETS[scope]))
        model = TaxonomyNaiveBayes()
        model.fit(training_records)
        title_model = TaxonomyNaiveBayes()
        title_model.fit(title_model_record(record) for record in training_records)
        direction_model = TaxonomyNaiveBayes()
        direction_training = []
        for record in training_records:
            labelled = dict(record)
            labelled["event_name"] = record["event_emotion"]
            direction_training.append(labelled)
        direction_model.fit(direction_training)
        nearest_model = EntityTitleNearestNeighbor(training_records) if scope == "company" else None
        predictions = [
            predict_record(record, scope, model, title_model, direction_model, nearest_model, events_by_entity, entity_catalog)
            for record in iter_jsonl(DATA_DIR / TEST_DATASETS[scope])
        ]
        output_path = DERIVED_DIR / f"event_predictions_test_{scope}.jsonl.gz"
        with gzip.open(output_path, "wt", encoding="utf-8") as target:
            for prediction in predictions:
                target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        duplicate_predictions = [
            predict_record(record, scope, model, title_model, direction_model, nearest_model, events_by_entity, entity_catalog)
            for record in iter_jsonl(DATA_DIR / DUPLICATE_TEST_DATASETS[scope])
        ]
        duplicate_output_path = DERIVED_DIR / f"event_predictions_duplicates_test_{scope}.jsonl.gz"
        with gzip.open(duplicate_output_path, "wt", encoding="utf-8") as target:
            for prediction in duplicate_predictions:
                target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        metrics = evaluate(scope, predictions)
        metrics["training_records"] = model.document_count
        metrics["trained_event_types"] = len(model.labels)
        metrics["event_classifier"] = "entity_title_nearest_neighbor" if scope == "company" else "body_character_nb"
        metrics["direction_classifier"] = "supervised_character_nb"
        metrics["prediction_file"] = output_path.name
        metrics["timeline_input_prediction_file"] = duplicate_output_path.name
        report["scopes"][scope] = metrics

    (REPORT_DIR / "event_extraction_baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 事件识别基线报告", "",
        "方法：事件体系约束的字符 n-gram 朴素贝叶斯分类 + 规则化实体、时间、方向与证据抽取。", "",
        "| 范围 | 测试记录 | 事件准确率 | 事件 Macro-F1 | 方向准确率 | 实体覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, metrics in report["scopes"].items():
        direction = f"{metrics['direction_accuracy']:.2%}" if metrics["direction_accuracy"] is not None else "N/A"
        lines.append(f"| {scope} | {metrics['records']} | {metrics['event_accuracy']:.2%} | {metrics['event_macro_f1']:.2%} | {direction} | {metrics['entity_field_coverage']:.2%} |")
    lines.extend(["", "该基线用于建立可复现的离线对照；后续可用语义向量召回、重排序模型和大模型结构化抽取提升长尾事件与隐含语义的识别效果。"])
    (REPORT_DIR / "event_extraction_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
