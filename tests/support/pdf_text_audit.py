"""Validate the font size of text glyphs embedded in exported plot PDFs.

Matplotlib's artist tree reports the parent font size, but mathtext may export
subscripts and superscripts at a smaller size.  This module inspects MuPDF's
structured-text representation of the finished PDF so the publication audit
measures the glyphs that will actually be printed.

Example:
    ``assert_pdf_minimum_text_size(Path("figure.pdf"))``
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PdfTextSpan:
    """One text span reported by MuPDF for an exported PDF page."""

    page_number: int
    text: str
    font_size_pt: float


def inventory_pdf_text_spans(path: Path) -> tuple[PdfTextSpan, ...]:
    """Return every non-empty text span and its exported font size."""
    mutool = shutil.which("mutool")
    if mutool is None:
        raise RuntimeError("mutool is required for exported-PDF text auditing.")
    result = subprocess.run(
        [mutool, "draw", "-F", "stext.json", "-o", "-", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    spans: list[PdfTextSpan] = []
    for page_number, page in enumerate(payload.get("pages", []), start=1):
        for item in _walk_json(page):
            font = item.get("font")
            text = item.get("text")
            if not isinstance(font, dict) or not isinstance(text, str) or not text.strip():
                continue
            size = font.get("size")
            if isinstance(size, bool) or not isinstance(size, (int, float)):
                continue
            numeric_size = float(size)
            if math.isfinite(numeric_size) and numeric_size > 0:
                spans.append(PdfTextSpan(page_number, text, numeric_size))
    return tuple(spans)


def assert_pdf_minimum_text_size(path: Path, *, minimum_pt: float = 8.0) -> tuple[PdfTextSpan, ...]:
    """Raise if an exported PDF contains a visible text span below the floor."""
    if not math.isfinite(minimum_pt) or minimum_pt <= 0:
        raise ValueError("minimum_pt must be a finite positive number.")
    spans = inventory_pdf_text_spans(path)
    offenders = [span for span in spans if span.font_size_pt + 1e-9 < minimum_pt]
    if offenders:
        details = "\n".join(
            f"- page {span.page_number} {span.text!r}: {span.font_size_pt:.2f} pt"
            for span in offenders
        )
        raise AssertionError(
            f"Exported PDF text is below the {minimum_pt:.2f} pt final-print minimum in {path}:\n{details}"
        )
    return spans


def _walk_json(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)
