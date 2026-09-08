"""Validate a locally-trained semantic retrieval model for company event types.

The embedding uses only the permitted labelled-training and unlabelled-training
news, never the labelled holdout set.  Training is checkpointed one epoch at a
time, which keeps it usable under the desktop execution-time limit:

    D:\\myanaconda\\python.exe scripts/gensim_semantic_retrieval_experiment.py --train-epoch
    D:\\myanaconda\\python.exe scripts/gensim_semantic_retrieval_experiment.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from gensim.models import Word2Vec

from event_extraction_baseline import (
    DATA_DIR,
    EntityTitleNearestNeighbor,
    char_ngrams,
    extract_entities,
    iter_jsonl,
    load_taxonomy,
    macro_f1,
)
from event_model_experiments import stratified_split


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "event_radar" / "models" / "company_title_char_w2v.model"
REPORT_DIR = ROOT / "event_radar" / "reports"
CORPUS_FILES = (
    DATA_DIR / "news_labeled_train_company.jsonl.gz",
    DATA_DIR / "news_unlabeled_train.jsonl.gz",
)


def title_chars(title: str) -> list[str]:
    """Keep Chinese characters, letters and digits; Word2Vec consumes characters."""
    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", title or "")


class TitleCorpus:
    def __iter__(self) -> Iterable[list[str]]:
        for path in CORPUS_FILES:
            for record in iter_jsonl(path):
                tokens = title_chars(record.get("title") or "")
                if len(tokens) >= 2:
                    yield tokens


def sentence_vector(model: Word2Vec, title: str) -> np.ndarray:
    tokens = [token for token in title_chars(title) if token in model.wv]
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    vector = np.mean(model.wv[tokens], axis=0)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class SemanticEntityNearestNeighbor:
    """Issuer-local maximum cosine retrieval over historical labelled headlines."""

    def __init__(self, records: list[dict[str, Any]], model: Word2Vec) -> None:
        self.model = model
        self.by_entity: dict[str, list[tuple[str, np.ndarray]]] = {}
        self.global_samples: list[tuple[str, np.ndarray]] = []
        for record in records:
            vector = sentence_vector(model, record.get("title") or "")
            sample = (record["event_name"], vector)
            self.global_samples.append(sample)
            code = record.get("security_code") or ""
            self.by_entity.setdefault(code, []).append(sample)

    def score(self, record: dict[str, Any], candidates: list[str]) -> dict[str, float]:
        target = sentence_vector(self.model, record.get("title") or "")
        samples = self.by_entity.get(record.get("security_code") or "", self.global_samples)
        scores = {label: 0.0 for label in candidates}
        if not np.any(target):
            return scores
        for label, vector in samples:
            if label in scores and np.any(vector):
                scores[label] = max(scores[label], float(np.dot(target, vector)))
        return scores

    @property
    def model(self) -> Word2Vec:
        return self._model

    @model.setter
    def model(self, value: Word2Vec) -> None:
        self._model = value


def candidate_labels(record: dict[str, Any], taxonomy: dict[str, dict[str, set[str]]], entities: dict[str, list[dict[str, str]]], all_labels: list[str]) -> list[str]:
    taxonomy_labels: set[str] = set()
    for subject in extract_entities(record, "company", entities):
        taxonomy_labels.update(taxonomy["company"].get(subject["code"], set()))
    constrained = sorted(taxonomy_labels & set(all_labels))
    return constrained or all_labels


def predict_all(train: list[dict[str, Any]], evaluation: list[dict[str, Any]], model: Word2Vec) -> dict[str, list[str]]:
    taxonomy, entities = load_taxonomy()
    char_model = EntityTitleNearestNeighbor(train)
    semantic_model = SemanticEntityNearestNeighbor(train, model)
    labels = sorted({record["event_name"] for record in train})

    outcomes = {"character_nn": [], "semantic_nn": []}
    outcomes.update({f"hybrid_char_{weight:.1f}": [] for weight in np.arange(0.1, 1.0, 0.1)})
    for record in evaluation:
        candidates = candidate_labels(record, taxonomy, entities, labels)
        character_scores = char_model.score(record, candidates)
        semantic_scores = semantic_model.score(record, candidates)
        outcomes["character_nn"].append(char_model.predict(record, candidates))
        outcomes["semantic_nn"].append(max(candidates, key=lambda label: semantic_scores[label]))
        for weight in np.arange(0.1, 1.0, 0.1):
            name = f"hybrid_char_{weight:.1f}"
            outcomes[name].append(max(candidates, key=lambda label: weight * character_scores[label] + (1.0 - weight) * semantic_scores[label]))
    return outcomes


def evaluate(records: list[dict[str, Any]], outcomes: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    expected = [record["event_name"] for record in records]
    return {
        name: {
            "accuracy": round(sum(actual == predicted for actual, predicted in zip(expected, labels)) / len(expected), 4),
            "macro_f1": round(macro_f1(expected, labels), 4),
        }
        for name, labels in outcomes.items()
    }


def hybrid_margins(train: list[dict[str, Any]], evaluation: list[dict[str, Any]], model: Word2Vec) -> tuple[list[str], list[float]]:
    """Return the selected 0.1-character / 0.9-semantic labels and margins."""
    taxonomy, entities = load_taxonomy()
    char_model = EntityTitleNearestNeighbor(train)
    semantic_model = SemanticEntityNearestNeighbor(train, model)
    labels = sorted({record["event_name"] for record in train})
    predictions, margins = [], []
    for record in evaluation:
        candidates = candidate_labels(record, taxonomy, entities, labels)
        char_scores = char_model.score(record, candidates)
        semantic_scores = semantic_model.score(record, candidates)
        ranking = sorted(((0.1 * char_scores[label] + 0.9 * semantic_scores[label], label) for label in candidates), reverse=True)
        predictions.append(ranking[0][1])
        margins.append(ranking[0][0] - (ranking[1][0] if len(ranking) > 1 else 0.0))
    return predictions, margins


def selective_diagnostics(records: list[dict[str, Any]], predictions: list[str], margins: list[float]) -> dict[str, dict[str, float]]:
    expected = [record["event_name"] for record in records]
    outcomes: dict[str, dict[str, float]] = {}
    for threshold in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        chosen = [(actual, predicted) for actual, predicted, margin in zip(expected, predictions, margins) if margin >= threshold]
        accuracy = sum(actual == predicted for actual, predicted in chosen) / len(chosen) if chosen else 0.0
        outcomes[f">={threshold:.3f}"] = {"coverage": round(len(chosen) / len(expected), 4), "accuracy": round(accuracy, 4)}
    return outcomes


def train_epoch() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus = TitleCorpus()
    if MODEL_PATH.exists():
        model = Word2Vec.load(str(MODEL_PATH))
    else:
        model = Word2Vec(vector_size=96, window=5, min_count=2, workers=4, sg=1, negative=8, seed=20260727)
        model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=1)
    model.save(str(MODEL_PATH))
    print(json.dumps({"status": "trained_one_epoch", "corpus_count": model.corpus_count, "vocabulary": len(model.wv), "epochs_completed": model.get_latest_training_loss()}, ensure_ascii=False))


def run_evaluation() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No checkpoint yet. Run --train-epoch first.")
    model = Word2Vec.load(str(MODEL_PATH))
    records = list(iter_jsonl(DATA_DIR / "news_labeled_train_company.jsonl.gz"))
    train, validation = stratified_split(records)
    validation_scores = evaluate(validation, predict_all(train, validation, model))
    selected = max(validation_scores, key=lambda name: (validation_scores[name]["macro_f1"], validation_scores[name]["accuracy"]))
    test = list(iter_jsonl(DATA_DIR / "news_labeled_test_company.jsonl.gz"))
    test_scores = evaluate(test, predict_all(records, test, model))
    validation_hybrid, validation_margins = hybrid_margins(train, validation, model)
    report = {
        "method": "character Word2Vec trained on labelled training and unlabelled training headlines, issuer-local semantic retrieval, and character/semantic score fusion",
        "embedding": {"vector_size": model.vector_size, "vocabulary": len(model.wv), "corpus_count": model.corpus_count},
        "selection_rule": "highest validation Macro-F1, then accuracy",
        "selected_model": selected,
        "validation": validation_scores,
        "provided_test_diagnostic": test_scores,
        "validation_hybrid_margin_diagnostics": selective_diagnostics(validation, validation_hybrid, validation_margins),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "gensim_semantic_retrieval_experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 本地语义向量检索实验", "",
        "语义向量只在有标签训练新闻与无标签训练新闻标题上无监督学习；标签模型选择严格以训练集的分层验证结果为准。", "",
        "| 模型 | 验证 Accuracy | 验证 Macro-F1 | 提供测试集 Accuracy（仅诊断） | 提供测试集 Macro-F1（仅诊断） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in validation_scores.items():
        diagnosis = test_scores[name]
        lines.append(f"| {name} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.2%} | {diagnosis['accuracy']:.2%} | {diagnosis['macro_f1']:.2%} |")
    lines.extend(["", f"验证集选中：`{selected}`。"])
    lines.extend(["", "融合模型的验证集置信度诊断："])
    for threshold, metrics in report["validation_hybrid_margin_diagnostics"].items():
        lines.append(f"- 边际分数 {threshold}：覆盖率 {metrics['coverage']:.2%}，准确率 {metrics['accuracy']:.2%}。")
    (REPORT_DIR / "gensim_semantic_retrieval_experiment.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-epoch", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.train_epoch:
        train_epoch()
    if args.evaluate:
        run_evaluation()
    if not args.train_epoch and not args.evaluate:
        parser.error("Choose --train-epoch or --evaluate")


if __name__ == "__main__":
    main()
