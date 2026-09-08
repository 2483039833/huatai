"""Evaluate the integrated B classifier on the held-out cleaned company test set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.b_event_classifier import get_default_b_event_classifier  # noqa: E402
from src.services.cleaned_xlsx import iter_cleaned_rows  # noqa: E402
from src.services.event_adapter import EventAdapter  # noqa: E402
from src.services.group_builder import GroupBuilder  # noqa: E402


def evaluate(test_path: Path) -> dict:
    frame = pd.DataFrame(iter_cleaned_rows(test_path))
    events = EventAdapter.from_dataframe(frame)
    groups = GroupBuilder.build_groups(events, np.arange(len(events), dtype=np.int32))
    classifier = get_default_b_event_classifier()

    truth_by_id = {event.article_file_id: event for event in events}
    event_truth: list[str] = []
    event_pred: list[str] = []
    direction_truth: list[str] = []
    direction_pred: list[str] = []
    review_count = 0
    missing_count = 0
    errors: Counter[tuple[str, str]] = Counter()

    for group in groups:
        truth = truth_by_id[group.representative_article_id]
        prediction = classifier.classify_group(group)
        if prediction is None:
            missing_count += 1
            continue
        event_truth.append(truth.event_name or "")
        event_pred.append(prediction.event_name)
        direction_truth.append(truth.event_impact_direction or "中性")
        direction_pred.append(prediction.impact_direction)
        review_count += int(prediction.review_required)
        if event_truth[-1] != event_pred[-1]:
            errors[(event_truth[-1], event_pred[-1])] += 1

    evaluated = len(event_truth)
    return {
        "test_path": str(test_path.resolve()),
        "test_rows": len(events),
        "evaluated_rows": evaluated,
        "coverage": evaluated / len(events) if events else 0.0,
        "event_accuracy": accuracy_score(event_truth, event_pred) if evaluated else 0.0,
        "event_macro_f1": f1_score(event_truth, event_pred, average="macro", zero_division=0) if evaluated else 0.0,
        "direction_accuracy": accuracy_score(direction_truth, direction_pred) if evaluated else 0.0,
        "review_required_rate": review_count / evaluated if evaluated else 0.0,
        "missing_predictions": missing_count,
        "top_confusions": [
            {"actual": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in errors.most_common(20)
        ],
        "split_policy": "train_tagged_cleaned.xlsx for fitting; test_tagged_cleaned.xlsx for evaluation only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B classifier on held-out cleaned data")
    parser.add_argument(
        "--test-path",
        type=Path,
        default=PROJECT_ROOT / "cleaned_data" / "test_tagged_cleaned.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "results" / "b_classifier_evaluation.json",
    )
    args = parser.parse_args()
    report = evaluate(args.test_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
