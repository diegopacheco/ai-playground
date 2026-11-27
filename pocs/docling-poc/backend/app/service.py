from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from app import log
from app.domain import Conversion, ConversionSummary, ConvertedDocument, Status, detect_format
from app.errors import ConversionFailed, EngineError, Invalid, NotFound, PayloadTooLarge

logger = log.get(__name__)


class DocumentEngine(Protocol):
    def convert(self, filename: str, data: bytes, ocr: bool) -> ConvertedDocument: ...


class ConversionRepository(Protocol):
    def save(self, conversion: Conversion) -> None: ...
    def get(self, conversion_id: str) -> Conversion | None: ...
    def all(self) -> tuple[Conversion, ...]: ...


class ConversionService:
    def __init__(
        self,
        engine: DocumentEngine,
        repo: ConversionRepository,
        clock: Callable[[], datetime],
        new_id: Callable[[], str],
        max_upload_bytes: int,
    ) -> None:
        self._engine = engine
        self._repo = repo
        self._clock = clock
        self._new_id = new_id
        self._max_upload_bytes = max_upload_bytes

    def convert(self, filename: str, data: bytes, ocr: bool) -> Conversion:
        if not filename.strip():
            raise Invalid("filename is required")
        if not data:
            raise Invalid("file is empty")
        if len(data) > self._max_upload_bytes:
            raise PayloadTooLarge(f"file is larger than {self._max_upload_bytes} bytes")
        fmt = detect_format(filename)
        started = self._clock()
        try:
            document = self._engine.convert(filename, data, ocr)
            error = None
        except EngineError as e:
            document = None
            error = str(e)
        elapsed = int((self._clock() - started).total_seconds() * 1000)
        conversion = Conversion(
            id=self._new_id(),
            filename=filename,
            format=fmt,
            ocr=ocr,
            status=Status.SUCCESS if document else Status.FAILED,
            created_at=started,
            duration_ms=elapsed,
            document=document,
            error=error,
        )
        self._repo.save(conversion)
        logger.info("conversion id=%s file=%s status=%s ms=%s", conversion.id, filename, conversion.status, elapsed)
        if document is None:
            raise ConversionFailed(f"conversion {conversion.id} failed: {error}")
        return conversion

    def get(self, conversion_id: str) -> Conversion:
        found = self._repo.get(conversion_id)
        if found is None:
            raise NotFound(f"conversion {conversion_id} not found")
        return found

    def history(self) -> tuple[ConversionSummary, ...]:
        ordered = sorted(self._repo.all(), key=lambda c: c.created_at, reverse=True)
        return tuple(ConversionSummary.of(c) for c in ordered)
