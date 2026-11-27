from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePath

from app.errors import UnsupportedFormat


class DocFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    MARKDOWN = "md"
    CSV = "csv"
    IMAGE = "image"


_EXTENSIONS: dict[str, DocFormat] = {
    ".pdf": DocFormat.PDF,
    ".docx": DocFormat.DOCX,
    ".pptx": DocFormat.PPTX,
    ".xlsx": DocFormat.XLSX,
    ".html": DocFormat.HTML,
    ".htm": DocFormat.HTML,
    ".md": DocFormat.MARKDOWN,
    ".csv": DocFormat.CSV,
    ".png": DocFormat.IMAGE,
    ".jpg": DocFormat.IMAGE,
    ".jpeg": DocFormat.IMAGE,
    ".tiff": DocFormat.IMAGE,
}


def supported_extensions() -> tuple[str, ...]:
    return tuple(_EXTENSIONS)


def detect_format(filename: str) -> DocFormat:
    found = _EXTENSIONS.get(PurePath(filename).suffix.lower())
    if found is None:
        raise UnsupportedFormat(f"unsupported file type: {filename}")
    return found


class Status(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TableView:
    index: int
    caption: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    headings: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class Stats:
    pages: int
    texts: int
    tables: int
    pictures: int
    headings: int
    words: int


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    markdown: str
    html: str
    tables: tuple[TableView, ...]
    chunks: tuple[Chunk, ...]
    stats: Stats


@dataclass(frozen=True, slots=True)
class Conversion:
    id: str
    filename: str
    format: DocFormat
    ocr: bool
    status: Status
    created_at: datetime
    duration_ms: int
    document: ConvertedDocument | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    id: str
    filename: str
    format: DocFormat
    ocr: bool
    status: Status
    created_at: datetime
    duration_ms: int
    pages: int
    tables: int
    error: str | None

    @classmethod
    def of(cls, c: Conversion) -> "ConversionSummary":
        stats = c.document.stats if c.document else None
        return cls(
            id=c.id,
            filename=c.filename,
            format=c.format,
            ocr=c.ocr,
            status=c.status,
            created_at=c.created_at,
            duration_ms=c.duration_ms,
            pages=stats.pages if stats else 0,
            tables=stats.tables if stats else 0,
            error=c.error,
        )
