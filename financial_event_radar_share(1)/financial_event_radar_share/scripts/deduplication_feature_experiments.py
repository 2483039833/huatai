"""Test a learned multi-feature same-event classifier against the fixed threshold.

It remains dependency-free: a small logistic-regression implementation learns
how title overlap, body overlap, timing, source diversity and length patterns
jointly indicate repeated coverage.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from deduplication_baseline import (
    REPORT_DIR,
    TEST_FILES,
    TRAIN_FILES,
    Article,
    candidate_pairs,
    containment,
    make_articles,
    pair_metrics,
    positive_pairs,
)


def publication_gap(left: Article, right: Article) -> float:
    if not left.published_at or not right.published_at:
        return 0.0
    try:
        left_time = datetime.fromisoformat(left.published_at.replace("Z", "+00:00"))
        right_time = datetime.fromisoformat(right.published_at.replace("Z", "+00:00"))
        return math.exp(-abs((left_time - right_time).total_seconds()) / 86400 / 7)
    except ValueError:
        return 0.0


def features(left: Article, right: Article) -> list[float]:
    title = containment(left.title_features, right.title_features)
    body = containment(left.body_features, right.body_features)
    content_ratio = min(len(left.content), len(right.content)) / max(len(left.content), len(right.content), 1)
    return [
        title,
        body,
        publication_gap(left, right),
        float(bool(left.source and left.source == right.source)),
        float(left.title.strip() == right.title.strip()),
        content_ratio,
    ]


class LogisticRegression:
    def __init__(self) -> None:
        self.mean: list[float] = []
        self.scale: list[float] = []
        self.weights: list[float] = []

    def transform(self, values: list[float]) -> list[float]:
        return [1.0] + [(value - mean) / scale for value, mean, scale in zip(values, self.mean, self.scale)]

    def fit(self, rows: list[list[float]], labels: list[int], epochs: int = 260, learning_rate: float = 0.08) -> None:
        width = len(rows[0])
        self.mean = [sum(row[index] for row in rows) / len(rows) for index in range(width)]
        self.scale = [max(0.05, math.sqrt(sum((row[index] - self.mean[index]) ** 2 for row in rows) / len(rows))) for index in range(width)]
        self.weights = [0.0] * (width + 1)
        transformed = [self.transform(row) for row in rows]
        positive_weight = len(labels) / max(1, 2 * sum(labels))
        negative_weight = len(labels) / max(1, 2 * (len(labels) - sum(labels)))
        for _ in range(epochs):
            gradient = [0.0] * len(self.weights)
            for row, label in zip(transformed, labels):
                score = sum(weight * value for weight, value in zip(self.weights, row))
                probability = 1 / (1 + math.exp(-max(-30, min(30, score))))
                error = (probability - label) * (positive_weight if label else negative_weight)
                for index, value in enumerate(row):
                    gradient[index] += error * value
            for index in range(len(self.weights)):
                self.weights[index] -= learning_rate * gradient[index] / len(rows)

    def probability(self, row: list[float]) -> float:
        values = self.transform(row)
        score = sum(weight * value for weight, value in zip(self.weights, values))
        return 1 / (1 + math.exp(-max(-30, min(30, score))))


def choose_threshold(probabilities: list[tuple[tuple[int, int], float]], positives: set[tuple[int, int]]) -> tuple[float, dict[str, float]]:
    best_threshold, best_metrics = 1.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for index in range(10, 96, 2):
        threshold = index / 100
        predicted = {pair for pair, probability in probabilities if probability >= threshold}
        metrics = pair_metrics(predicted, positives)
        if metrics["precision"] >= 0.88 and (metrics["recall"], metrics["f1"]) > (best_metrics["recall"], best_metrics["f1"]):
            best_threshold, best_metrics = threshold, metrics
    return best_threshold, best_metrics


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(20260727)
    train_articles = make_articles(TRAIN_FILES)
    train_candidates = sorted(candidate_pairs(train_articles))
    train_positives = positive_pairs(train_articles)
    positive_examples = [pair for pair in train_candidates if pair in train_positives]
    negative_examples = [pair for pair in train_candidates if pair not in train_positives]
    random_generator.shuffle(negative_examples)
    sampled = positive_examples + negative_examples[: len(positive_examples) * 5]
    random_generator.shuffle(sampled)
    split = int(len(sampled) * 0.7)
    fit_pairs, calibration_pairs = sampled[:split], sampled[split:]
    model = LogisticRegression()
    model.fit([features(train_articles[left], train_articles[right]) for left, right in fit_pairs], [int(pair in train_positives) for pair in fit_pairs])
    calibration_probabilities = [(pair, model.probability(features(train_articles[left], train_articles[right]))) for pair in calibration_pairs for left, right in [pair]]
    calibration_positives = set(calibration_pairs) & train_positives
    threshold, calibration_metrics = choose_threshold(calibration_probabilities, calibration_positives)

    test_articles = make_articles(TEST_FILES)
    test_candidates = candidate_pairs(test_articles)
    test_positives = positive_pairs(test_articles)
    test_probabilities = [(pair, model.probability(features(test_articles[left], test_articles[right]))) for pair in test_candidates for left, right in [pair]]
    test_predicted = {pair for pair, probability in test_probabilities if probability >= threshold}
    test_metrics = pair_metrics(test_predicted, test_positives)
    report: dict[str, Any] = {
        "method": "class-balanced logistic regression on title/body containment, publication gap, same source, exact title and content length ratio",
        "training": {"candidate_pairs": len(train_candidates), "positive_pairs": len(train_positives), "fit_examples": len(fit_pairs), "calibration_examples": len(calibration_pairs)},
        "selected_threshold": threshold,
        "calibration_metrics": {key: round(value, 4) if isinstance(value, float) else value for key, value in calibration_metrics.items()},
        "test_diagnostic": {key: round(value, 4) if isinstance(value, float) else value for key, value in test_metrics.items()},
    }
    (REPORT_DIR / "deduplication_feature_experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# 多特征去重实验", "",
        "模型：类别平衡逻辑回归，特征包括标题/正文包含度、发布时间间隔、同来源、标题完全一致和正文长度比例。阈值只在训练集校准划分上选择。", "",
        f"- 选定阈值：`{threshold:.2f}`",
        f"- 提供测试集诊断 Precision / Recall / F1：`{test_metrics['precision']:.2%}` / `{test_metrics['recall']:.2%}` / `{test_metrics['f1']:.2%}`",
    ]
    (REPORT_DIR / "deduplication_feature_experiment_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
