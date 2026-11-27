import threading
from io import BytesIO

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TableItem, TitleItem
from docling_core.types.io import DocumentStream

from app.domain import Chunk, ConvertedDocument, Stats, TableView
from app.errors import EngineError


def _table_view(index: int, table: TableItem, doc: DoclingDocument) -> TableView:
    grid = tuple(tuple(cell.text for cell in row) for row in table.data.grid)
    return TableView(
        index=index,
        caption=table.caption_text(doc),
        headers=grid[0] if grid else (),
        rows=grid[1:],
    )


def _chunks(doc: DoclingDocument) -> tuple[Chunk, ...]:
    return tuple(
        Chunk(index=i, headings=tuple(getattr(c.meta, "headings", None) or ()), text=c.text)
        for i, c in enumerate(HierarchicalChunker().chunk(doc))
    )


def _stats(doc: DoclingDocument, markdown: str) -> Stats:
    headings = sum(1 for t in doc.texts if isinstance(t, (SectionHeaderItem, TitleItem)))
    return Stats(
        pages=doc.num_pages(),
        texts=len(doc.texts),
        tables=len(doc.tables),
        pictures=len(doc.pictures),
        headings=headings,
        words=len(markdown.split()),
    )


def to_converted(doc: DoclingDocument) -> ConvertedDocument:
    markdown = doc.export_to_markdown()
    return ConvertedDocument(
        markdown=markdown,
        html=doc.export_to_html(),
        tables=tuple(_table_view(i, t, doc) for i, t in enumerate(doc.tables)),
        chunks=_chunks(doc),
        stats=_stats(doc, markdown),
    )


def _build_converter(ocr: bool) -> DocumentConverter:
    options = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
        }
    )


class DoclingEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._converters: dict[bool, DocumentConverter] = {}

    def _converter(self, ocr: bool) -> DocumentConverter:
        if ocr not in self._converters:
            self._converters[ocr] = _build_converter(ocr)
        return self._converters[ocr]

    def convert(self, filename: str, data: bytes, ocr: bool) -> ConvertedDocument:
        with self._lock:
            try:
                result = self._converter(ocr).convert(
                    DocumentStream(name=filename, stream=BytesIO(data)), raises_on_error=False
                )
            except (ValueError, RuntimeError, OSError) as e:
                raise EngineError(str(e)) from e
            if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                details = "; ".join(err.error_message for err in result.errors) or result.status.value
                raise EngineError(details)
            return to_converted(result.document)
