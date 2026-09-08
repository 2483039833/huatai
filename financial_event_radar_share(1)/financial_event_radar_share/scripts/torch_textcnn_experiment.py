"""Checkpointable lightweight TextCNN experiment for company event classification.

Each invocation trains only a small number of epochs and saves a checkpoint, so
longer experiments can continue across bounded execution calls:
    D:\\myanaconda\\python.exe event_radar/scripts/torch_textcnn_experiment.py --epochs 2 --reset
    D:\\myanaconda\\python.exe event_radar/scripts/torch_textcnn_experiment.py --epochs 2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from event_extraction_baseline import DATA_DIR, allowed_labels, extract_entities, iter_jsonl, load_taxonomy, macro_f1
from event_model_experiments import stratified_split


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "event_radar" / "models"
REPORT_DIR = ROOT / "event_radar" / "reports"
CHECKPOINT_PATH = MODEL_DIR / "company_textcnn.pt"
MAX_LENGTH = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text(record: dict[str, Any]) -> str:
    return f"{record.get('title') or ''}{record.get('title') or ''}{(record.get('content') or '')[:300]}"


def build_vocab(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(character for record in records for character in text(record))
    return {character: index + 2 for index, (character, count) in enumerate(counts.most_common()) if count >= 2}


def encode(record: dict[str, Any], vocab: dict[str, int]) -> list[int]:
    values = [vocab.get(character, 1) for character in text(record)[:MAX_LENGTH]]
    return values + [0] * (MAX_LENGTH - len(values))


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size: int, class_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 96, padding_idx=0)
        self.convolutions = nn.ModuleList([nn.Conv1d(96, 80, kernel_size) for kernel_size in (2, 3, 4, 5)])
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(80 * 4, class_count)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        vectors = [torch.relu(convolution(embedded)).amax(dim=2) for convolution in self.convolutions]
        return self.output(self.dropout(torch.cat(vectors, dim=1)))


def build_candidate_indices(record: dict[str, Any], labels: list[str], taxonomy: dict[str, dict[str, set[str]]], entities: dict[str, list[dict[str, str]]]) -> list[int]:
    allowed = allowed_labels("company", extract_entities(record, "company", entities), taxonomy, type("Model", (), {"labels": labels})())
    index = {label: position for position, label in enumerate(labels)}
    return [index[label] for label in allowed]


def evaluate(model: TextCNN, records: list[dict[str, Any]], vocab: dict[str, int], labels: list[str]) -> dict[str, float]:
    model.eval()
    taxonomy, entities = load_taxonomy()
    expected = [record["event_name"] for record in records]
    predicted = []
    with torch.no_grad():
        for start in range(0, len(records), 48):
            batch = records[start : start + 48]
            logits = model(torch.tensor([encode(record, vocab) for record in batch], dtype=torch.long, device=DEVICE)).cpu()
            for record, row in zip(batch, logits):
                candidates = build_candidate_indices(record, labels, taxonomy, entities)
                winner = max(candidates, key=lambda position: float(row[position]))
                predicted.append(labels[winner])
    return {
        "accuracy": round(sum(left == right for left, right in zip(expected, predicted)) / len(expected), 4),
        "macro_f1": round(macro_f1(expected, predicted), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a checkpointable company-event TextCNN.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20260727)
    random.seed(20260727)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = list(iter_jsonl(DATA_DIR / "news_labeled_train_company.jsonl.gz"))
    train, validation = stratified_split(records)

    if CHECKPOINT_PATH.exists() and not args.reset:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        vocab, labels, completed_epochs = checkpoint["vocab"], checkpoint["labels"], checkpoint["epochs"]
        model = TextCNN(len(vocab) + 2, len(labels)).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    else:
        vocab = build_vocab(train)
        labels = sorted({record["event_name"] for record in train})
        completed_epochs = 0
        model = TextCNN(len(vocab) + 2, len(labels)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    label_index = {label: index for index, label in enumerate(labels)}
    counts = Counter(record["event_name"] for record in train)
    class_weights = torch.tensor([len(train) / (len(labels) * counts[label]) for label in labels], dtype=torch.float, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.train()
    for epoch in range(args.epochs):
        shuffled = list(train)
        random.shuffle(shuffled)
        losses = []
        for start in range(0, len(shuffled), 32):
            batch = shuffled[start : start + 32]
            tokens = torch.tensor([encode(record, vocab) for record in batch], dtype=torch.long, device=DEVICE)
            targets = torch.tensor([label_index[record["event_name"]] for record in batch], dtype=torch.long, device=DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(tokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        completed_epochs += 1
        validation_metrics = evaluate(model, validation, vocab, labels)
        print(f"epoch={completed_epochs} loss={sum(losses)/len(losses):.4f} validation_accuracy={validation_metrics['accuracy']:.4f} validation_macro_f1={validation_metrics['macro_f1']:.4f}", flush=True)

    torch.save({"vocab": vocab, "labels": labels, "epochs": completed_epochs, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, CHECKPOINT_PATH)
    diagnostic = evaluate(model, list(iter_jsonl(DATA_DIR / "news_labeled_test_company.jsonl.gz")), vocab, labels)
    report = {"device": str(DEVICE), "completed_epochs": completed_epochs, "validation": evaluate(model, validation, vocab, labels), "provided_test_diagnostic": diagnostic, "checkpoint": str(CHECKPOINT_PATH)}
    (REPORT_DIR / "textcnn_experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
