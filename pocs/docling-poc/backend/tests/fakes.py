from datetime import UTC, datetime, timedelta

from app.domain import Chunk, ConvertedDocument, Stats, TableView
from app.errors import EngineError
from app.repo import InMemoryConversionRepository
from app.service import ConversionService


class FakeEngine:
    def __init__(self, fail_with: str | None = None) -> None:
        self.fail_with = fail_with
        self.calls: list[tuple[str, bool]] = []

    def convert(self, filename: str, data: bytes, ocr: bool) -> ConvertedDocument:
        self.calls.append((filename, ocr))
        if self.fail_with:
            raise EngineError(self.fail_with)
        return ConvertedDocument(
            markdown="# Title\n\nbody",
            html="<h1>Title</h1>",
            tables=(TableView(index=0, caption="", headers=("a",), rows=(("1",),)),),
            chunks=(Chunk(index=0, headings=("Title",), text="body"),),
            stats=Stats(pages=3, texts=2, tables=1, pictures=0, headings=1, words=3),
        )


class StepClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        self._now = self._now + timedelta(milliseconds=250)
        return self._now


class SeqIds:
    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> str:
        self._n += 1
        return f"id{self._n}"


def make_service(engine: FakeEngine, max_upload_bytes: int = 1000) -> ConversionService:
    return ConversionService(
        engine=engine,
        repo=InMemoryConversionRepository(),
        clock=StepClock(),
        new_id=SeqIds(),
        max_upload_bytes=max_upload_bytes,
    )
