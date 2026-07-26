from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Highlight, Text
from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject

from .config import config

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Mark:
    page: int
    x: float
    y: float
    width: float
    height: float
    color: str
    note: str
    kind: str


def _quads(x1: float, y1: float, x2: float, y2: float) -> ArrayObject:
    corners = [x1, y2, x2, y2, x1, y1, x2, y1]
    return ArrayObject([FloatObject(value) for value in corners])


def _rect(mark: Mark, width: float, height: float) -> tuple[float, float, float, float]:
    x1 = mark.x * width
    x2 = (mark.x + mark.width) * width
    y_top = height - mark.y * height
    y_bottom = height - (mark.y + mark.height) * height
    return x1, y_bottom, x2, y_top


def safe_stem(name: str) -> str:
    stem = SAFE_NAME.sub("_", Path(name).stem).strip("_")
    return stem or "document"


def apply(source: Path, file_name: str, marks: list[Mark]) -> tuple[Path, int]:
    if not marks:
        raise ValueError("no annotations supplied")

    reader = PdfReader(str(source))
    writer = PdfWriter(clone_from=reader)
    applied = 0

    for mark in marks:
        position = mark.page - 1
        if position < 0 or position >= len(writer.pages):
            continue
        page = writer.pages[position]
        box = page.mediabox
        width = float(box.width)
        height = float(box.height)
        x1, y1, x2, y2 = _rect(mark, width, height)

        if mark.kind == "note":
            annotation = Text(
                rect=(x1, y1, x1 + 20, y1 + 20),
                text=mark.note or "note",
                open=False,
            )
        else:
            annotation = Highlight(
                rect=(x1, y1, x2, y2),
                quad_points=_quads(x1, y1, x2, y2),
                highlight_color=mark.color.lstrip("#"),
            )
            if mark.note:
                annotation[NameObject("/Contents")] = TextStringObject(mark.note)

        writer.add_annotation(page_number=position, annotation=annotation)
        applied += 1

    if applied == 0:
        raise ValueError("no annotation landed on an existing page")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    target = config.annotated_dir / f"{safe_stem(file_name)}-annotated-{stamp}.pdf"
    with target.open("wb") as handle:
        writer.write(handle)
    return target, applied
