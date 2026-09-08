"""Audit the official raw data and member-A cleaned data.

The report is intentionally independent from model code so it can be run before
training or a full A -> C -> B pipeline refresh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "train_tagged": "train_tagged_cleaned.xlsx",
    "test_tagged": "test_tagged_cleaned.xlsx",
    "unlabeled_train": "unlabeled_train_cleaned.xlsx",
    "unlabeled_test": "unlabeled_test_cleaned.xlsx",
}
TAGGED = {"train_tagged", "test_tagged"}


def _summarize_frame(frame: pd.DataFrame, tagged: bool) -> dict[str, Any]:
    ids = frame.get("article_file_id", pd.Series(dtype="object"))
    content = frame.get("content", pd.Series(dtype="object")).fillna("").astype(str)
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_article_id": int(ids.isna().sum()),
        "duplicate_article_id": int(ids.astype(str).duplicated().sum()),
        "empty_content": int((content.str.len() == 0).sum()),
        "content_shorter_than_30": int((content.str.len() < 30).sum()),
        "median_content_length": float(content.str.len().median()),
    }
    if tagged:
        for column in ("event_name", "event_emotion"):
            summary[f"missing_{column}"] = (
                int(frame[column].isna().sum()) if column in frame else int(len(frame))
            )
    return summary


def audit(cleaned_dir: Path) -> dict[str, Any]:
    id_sets: dict[str, set[str]] = {}
    datasets: dict[str, Any] = {}
    for name, filename in DATASETS.items():
        path = cleaned_dir / filename
        if not path.exists():
            datasets[name] = {"path": str(path), "missing": True}
            continue
        frame = pd.read_excel(path)
        id_sets[name] = set(frame["article_file_id"].dropna().astype(str))
        datasets[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            **_summarize_frame(frame, name in TAGGED),
        }

    overlaps: dict[str, int] = {}
    for left, right in (("train_tagged", "test_tagged"), ("unlabeled_train", "unlabeled_test")):
        if left in id_sets and right in id_sets:
            overlaps[f"{left}_vs_{right}"] = len(id_sets[left] & id_sets[right])

    required_present = len(id_sets) == len(DATASETS)
    identifiers_clean = all(
        item.get("missing_article_id") == 0 and item.get("duplicate_article_id") == 0
        for item in datasets.values()
        if not item.get("missing")
    )
    labels_complete = all(
        datasets[name].get("missing_event_name") == 0
        and datasets[name].get("missing_event_emotion") == 0
        for name in TAGGED
        if name in datasets and not datasets[name].get("missing")
    )
    return {
        "cleaned_dir": str(cleaned_dir),
        "datasets": datasets,
        "id_overlap": overlaps,
        "gates": {
            "all_required_files_present": required_present,
            "identifiers_complete_and_unique": identifiers_clean,
            "tagged_labels_complete": labels_complete,
            "ready_as_pipeline_input": required_present and identifiers_clean and labels_complete,
        },
        "model_policy": {
            "classifier_train": "cleaned tagged train for company events; raw train industry sheet until a cleaned industry file exists",
            "classifier_evaluation": "test_tagged_cleaned.xlsx only",
            "unlabeled_data": "inference and clustering only; never supervised labels",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cleaned competition datasets")
    parser.add_argument("--cleaned-dir", type=Path, default=PROJECT_ROOT / "cleaned_data")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "results" / "data_quality_report.json",
    )
    args = parser.parse_args()

    report = audit(args.cleaned_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
