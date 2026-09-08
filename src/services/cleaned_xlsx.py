"""Low-memory reader for A's inline-string cleaned XLSX exports."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterator


def iter_cleaned_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield worksheet rows without expanding the full XLSX in OpenPyXL."""
    headers: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        with archive.open("xl/worksheets/sheet1.xml") as sheet_xml:
            for _, element in ET.iterparse(sheet_xml, events=("end",)):
                if not element.tag.endswith("}row"):
                    continue
                values: dict[str, str] = {}
                for cell in element:
                    if not cell.tag.endswith("}c"):
                        continue
                    column = re.sub(r"\d", "", cell.attrib.get("r", ""))
                    values[column] = "".join(cell.itertext())
                if not headers:
                    headers = values
                else:
                    yield {name: values.get(column, "") for column, name in headers.items()}
                element.clear()

