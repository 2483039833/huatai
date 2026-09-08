"""Retrain B's character Word2Vec asset in the unified runtime.

Only cleaned labelled-train and unlabelled-train headlines are used. The
official labelled test split is deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def title_chars(title: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", str(title or ""))


def build_corpus(paths: list[Path], corpus_path: Path) -> tuple[int, dict[str, int]]:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_titles = 0
    source_rows: dict[str, int] = {}
    with corpus_path.open("w", encoding="utf-8", newline="\n") as target:
        for path in paths:
            source_rows[path.name] = 0
            with zipfile.ZipFile(path) as archive:
                with archive.open("xl/worksheets/sheet1.xml") as sheet_xml:
                    for _, element in ET.iterparse(sheet_xml, events=("end",)):
                        if not element.tag.endswith("}c"):
                            continue
                        cell_ref = element.attrib.get("r", "")
                        if cell_ref.startswith("B"):
                            title = "".join(element.itertext())
                            if cell_ref == "B1" and title != "article_title":
                                raise ValueError(f"Unexpected title column in {path}: {title!r}")
                            if cell_ref != "B1":
                                source_rows[path.name] += 1
                                tokens = title_chars(title)
                                if len(tokens) >= 2:
                                    target.write(" ".join(tokens) + "\n")
                                    corpus_titles += 1
                        element.clear()
    return corpus_titles, source_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B character Word2Vec without test leakage")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "b_word2vec_titles.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "b_company_title_char_w2v.model",
    )
    args = parser.parse_args()
    paths = [
        PROJECT_ROOT / "cleaned_data" / "train_tagged_cleaned.xlsx",
        PROJECT_ROOT / "cleaned_data" / "unlabeled_train_cleaned.xlsx",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing training data: {missing}")

    print("Streaming cleaned training titles into the corpus cache...", flush=True)
    corpus_titles, source_rows = build_corpus(paths, args.corpus_cache)
    print(f"Prepared {corpus_titles} titles; building Word2Vec vocabulary...", flush=True)
    corpus = LineSentence(str(args.corpus_cache))
    model = Word2Vec(
        sentences=corpus,
        vector_size=96,
        window=5,
        min_count=2,
        workers=args.workers,
        sg=1,
        negative=8,
        seed=20260727,
        epochs=args.epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    metadata = {
        "model_path": str(args.output.resolve()),
        "training_runtime": {"executable": sys.executable, "python": sys.version.split()[0]},
        "sources": source_rows,
        "test_split_used": False,
        "corpus_titles": corpus_titles,
        "corpus_cache": str(args.corpus_cache.resolve()),
        "vector_size": model.vector_size,
        "vocabulary": len(model.wv),
        "epochs": args.epochs,
        "parameters": {
            "window": 5,
            "min_count": 2,
            "sg": 1,
            "negative": 8,
            "seed": 20260727,
            "workers": args.workers,
        },
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
