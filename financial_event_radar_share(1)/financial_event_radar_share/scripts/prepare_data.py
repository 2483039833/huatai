"""Prepare the competition data for the Financial Event Radar project.

The script deliberately uses only the Python standard library.  It reads the
provided XLSX files in a streaming manner, normalises their fields, and writes
newline-delimited JSON files that later modules can consume independently.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT
OUTPUT_DIR = ROOT / "event_radar" / "data" / "processed"
SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

XLSX_SPECS = {
    "news_without_tags_train.xlsx": [("sheet1", "news_unlabeled_train")],
    "news_without_tags_test.xlsx": [("sheet1", "news_unlabeled_test")],
    "news_with_tags_train.xlsx": [
        ("sheet1", "news_labeled_train_company"),
        ("sheet2", "news_labeled_train_industry"),
        ("sheet3", "duplicates_train_company"),
        ("sheet4", "duplicates_train_industry"),
    ],
    "news_with_tags_test.xlsx": [
        ("sheet1", "news_labeled_test_company"),
        ("sheet2", "news_labeled_test_industry"),
        ("sheet3", "duplicates_test_company"),
        ("sheet4", "duplicates_test_industry"),
    ],
}

SOURCE_COLUMNS = {
    "news_unlabeled_train": ["article_file_id", "article_title", "article_source", "content", "article_publish_time"],
    "news_unlabeled_test": ["article_file_id", "article_title", "article_source", "content", "article_publish_time"],
    "news_labeled_train_company": ["article_file_id", "article_title", "article_publish_time", "article_source", "content", "trading_code", "secu_abbr", "industry_code", "industry_name", "event_name", "event_emotion", "event_impact_analysis", "duplication_id"],
    "news_labeled_test_company": ["article_file_id", "article_title", "article_publish_time", "article_source", "content", "trading_code", "secu_abbr", "industry_code", "industry_name", "event_name", "event_emotion", "event_impact_analysis", "duplication_id"],
    "news_labeled_train_industry": ["article_file_id", "article_title", "article_publish_time", "article_source", "content", "industry_code", "industry_name", "event_name", "event_emotion", "event_impact_analysis", "duplication_id"],
    "news_labeled_test_industry": ["article_file_id", "article_title", "article_publish_time", "article_source", "content", "industry_code", "industry_name", "event_name", "event_emotion", "event_impact_analysis", "duplication_id"],
    "duplicates_train_company": ["article_file_id", "article_title", "article_source", "content", "article_publish_time", "duplication_id"],
    "duplicates_train_industry": ["article_file_id", "article_title", "article_source", "content", "article_publish_time", "duplication_id"],
    "duplicates_test_company": ["article_file_id", "article_title", "article_source", "content", "article_publish_time", "duplication_id"],
    "duplicates_test_industry": ["article_file_id", "article_title", "article_source", "content", "article_publish_time", "duplication_id"],
}

WHITESPACE = re.compile(r"[\u200b\ufeff\xa0\t\r\n ]+")


def clean_text(value: Any) -> str | None:
    """Return a compact, nullable text value without changing its meaning."""
    if value is None:
        return None
    text = str(value).replace("\u3000", " ").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return WHITESPACE.sub(" ", text)


def excel_datetime(value: Any) -> str | None:
    """Convert an Excel serial date to ISO-8601, retaining times when present."""
    if value is None:
        return None
    if isinstance(value, (int, float)) or str(value).replace(".", "", 1).isdigit():
        serial = float(value)
        date = datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=serial)
        return date.isoformat().replace("+00:00", "Z")
    return clean_text(value)


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    entry = archive.open("xl/sharedStrings.xml")
    values: list[str] = []
    for _, element in ET.iterparse(entry, events=("end",)):
        if element.tag == f"{SPREADSHEET_NS}si":
            values.append("".join(node.text or "" for node in element.iter(f"{SPREADSHEET_NS}t")))
            element.clear()
    return values


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{SPREADSHEET_NS}t"))
    value = cell.find(f"{SPREADSHEET_NS}v")
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def column_index(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    result = 0
    for char in letters:
        result = result * 26 + ord(char.upper()) - ord("A") + 1
    return result - 1


def iter_xlsx_rows(path: Path, sheet_name: str) -> Iterator[list[str | None]]:
    """Yield rows from one XLSX worksheet without loading the worksheet XML."""
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet_path = f"xl/worksheets/{sheet_name}.xml"
        with archive.open(sheet_path) as source:
            for _, row in ET.iterparse(source, events=("end",)):
                if row.tag != f"{SPREADSHEET_NS}row":
                    continue
                cells = list(row.findall(f"{SPREADSHEET_NS}c"))
                if not cells:
                    row.clear()
                    continue
                width = max(column_index(cell.attrib.get("r", "A1")) for cell in cells) + 1
                values: list[str | None] = [None] * width
                for cell in cells:
                    values[column_index(cell.attrib.get("r", "A1"))] = cell_value(cell, shared_strings)
                yield values
                row.clear()


def normalise_record(row: dict[str, str | None], dataset: str) -> dict[str, Any]:
    """Map source-specific field names to one stable news-record schema."""
    record = {
        "article_id": clean_text(row.get("article_file_id")),
        "title": clean_text(row.get("article_title")),
        "published_at": excel_datetime(row.get("article_publish_time")),
        "source": clean_text(row.get("article_source")),
        "content": clean_text(row.get("content")),
        "security_code": clean_text(row.get("trading_code")),
        "security_name": clean_text(row.get("secu_abbr")),
        "industry_code": clean_text(row.get("industry_code")),
        "industry_name": clean_text(row.get("industry_name")),
        "event_name": clean_text(row.get("event_name")),
        "event_emotion": clean_text(row.get("event_emotion")),
        "impact_analysis": clean_text(row.get("event_impact_analysis")),
        "duplication_id": clean_text(row.get("duplication_id")),
        "dataset": dataset,
    }
    record["text"] = "\n".join(part for part in (record["title"], record["content"]) if part)
    return record


def prepare_taxonomy() -> dict[str, Any]:
    sources = (
        ("事件类型_标的.json", "company", "company"),
        ("事件类型_行业.json", "industry", "industry"),
    )
    rows: list[dict[str, Any]] = []
    for filename, scope, entity_key in sources:
        source = RAW_DIR / filename
        for entity_group in json.loads(source.read_text(encoding="utf-8")):
            entity = entity_group[entity_key]
            for item in entity_group["event_schema"]:
                rows.append(
                    {
                        "scope": scope,
                        "entity_code": clean_text(entity.get("trading_code") or entity.get("inducode")),
                        "entity_name": clean_text(entity.get("secu_abbr") or entity.get("induname")),
                        "event_name": clean_text(item["event_name"]),
                        "definition": clean_text(item["description"]),
                    }
                )

    taxonomy_path = OUTPUT_DIR / "event_taxonomy.jsonl"
    with taxonomy_path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_scope = Counter(row["scope"] for row in rows)
    event_names = defaultdict(set)
    for row in rows:
        event_names[row["scope"]].add(row["event_name"])
    return {
        "records": len(rows),
        "records_by_scope": dict(by_scope),
        "unique_event_types_by_scope": {scope: len(names) for scope, names in event_names.items()},
    }


def prepare_sheet(path: Path, sheet_name: str, dataset: str) -> dict[str, Any]:
    output_path = OUTPUT_DIR / f"{dataset}.jsonl.gz"
    iterator = iter_xlsx_rows(path, sheet_name)
    header = next(iterator)
    columns = [clean_text(value) for value in header]
    if not all(columns):
        raise ValueError(f"{path.name}/{sheet_name} contains an empty header")

    total = invalid = 0
    sources: Counter[str] = Counter()
    event_labels: Counter[str] = Counter()
    with gzip.open(output_path, "wt", encoding="utf-8") as target:
        for values in iterator:
            raw = {columns[index]: values[index] if index < len(values) else None for index in range(len(columns))}
            record = normalise_record(raw, dataset)
            total += 1
            if not record["article_id"] or not record["text"]:
                invalid += 1
                continue
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["source"]:
                sources[record["source"]] += 1
            if record["event_name"]:
                event_labels[record["event_name"]] += 1
            if total % 10_000 == 0:
                print(f"  {dataset}: {total:,} records processed", flush=True)

    return {
        "source_file": path.name,
        "source_sheet": sheet_name,
        "output_file": output_path.name,
        "columns": columns,
        "input_records": total,
        "valid_records": total - invalid,
        "skipped_records": invalid,
        "unique_sources": len(sources),
        "event_label_distribution": dict(event_labels.most_common()),
    }


def inspect_prepared_sheet(path: Path, sheet_name: str, dataset: str) -> dict[str, Any]:
    """Rebuild manifest information from an existing compressed JSONL file."""
    output_path = OUTPUT_DIR / f"{dataset}.jsonl.gz"
    total = 0
    sources: Counter[str] = Counter()
    event_labels: Counter[str] = Counter()
    with gzip.open(output_path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            total += 1
            if record.get("source"):
                sources[record["source"]] += 1
            if record.get("event_name"):
                event_labels[record["event_name"]] += 1
    return {
        "source_file": path.name,
        "source_sheet": sheet_name,
        "output_file": output_path.name,
        "columns": SOURCE_COLUMNS[dataset],
        "input_records": total,
        "valid_records": total,
        "skipped_records": 0,
        "unique_sources": len(sources),
        "event_label_distribution": dict(event_labels.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare event-radar competition data.")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Also normalise the two large unlabelled news workbooks.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rebuild the manifest from already prepared compressed JSONL files.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_version": "1.0",
        "record_schema": [
            "article_id", "title", "published_at", "source", "content", "text",
            "security_code", "security_name", "industry_code", "industry_name",
            "event_name", "event_emotion", "impact_analysis", "duplication_id", "dataset",
        ],
        "taxonomy": prepare_taxonomy(),
        "datasets": {},
    }

    if args.refresh_manifest:
        for filename, sheets in XLSX_SPECS.items():
            path = RAW_DIR / filename
            for sheet_name, dataset in sheets:
                if (OUTPUT_DIR / f"{dataset}.jsonl.gz").exists():
                    manifest["datasets"][dataset] = inspect_prepared_sheet(path, sheet_name, dataset)
        (OUTPUT_DIR / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Refreshed manifest for {len(manifest['datasets'])} datasets")
        return

    for filename, sheets in XLSX_SPECS.items():
        is_unlabeled = filename.startswith("news_without_tags")
        if is_unlabeled and not args.include_unlabeled:
            continue
        path = RAW_DIR / filename
        for sheet_name, dataset in sheets:
            print(f"Preparing {dataset}...", flush=True)
            manifest["datasets"][dataset] = prepare_sheet(path, sheet_name, dataset)

    (OUTPUT_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(manifest['datasets'])} datasets in {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, zipfile.BadZipFile) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
