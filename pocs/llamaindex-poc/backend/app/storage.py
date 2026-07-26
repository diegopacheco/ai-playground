from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config


@dataclass
class DocumentRecord:
    doc_id: str
    file_name: str
    size_bytes: int
    pages: int
    chunks: int
    chars: int
    ingested_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentStore:
    def __init__(self, manifest_path: Path, uploads_dir: Path) -> None:
        self._manifest_path = manifest_path
        self._uploads_dir = uploads_dir
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = self._read()

    def _read(self) -> dict[str, DocumentRecord]:
        if not self._manifest_path.is_file():
            return {}
        raw: dict[str, Any] = json.loads(self._manifest_path.read_text())
        return {key: DocumentRecord(**value) for key, value in raw.items()}

    def _write(self) -> None:
        payload = {key: value.to_dict() for key, value in self._records.items()}
        self._manifest_path.write_text(json.dumps(payload, indent=2))

    def path_for(self, doc_id: str) -> Path:
        return self._uploads_dir / f"{doc_id}.pdf"

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._records.get(doc_id)

    def list(self) -> list[DocumentRecord]:
        return sorted(self._records.values(), key=lambda record: record.ingested_at, reverse=True)

    def save_pdf(self, file_name: str, payload: bytes) -> tuple[str, Path, bool]:
        doc_id = hashlib.sha256(payload).hexdigest()[:16]
        target = self.path_for(doc_id)
        existing = doc_id in self._records
        if not existing:
            target.write_bytes(payload)
        return doc_id, target, existing

    def register(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.doc_id] = record
            self._write()

    def remove(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            record = self._records.pop(doc_id, None)
            if record is None:
                return None
            self.path_for(doc_id).unlink(missing_ok=True)
            self._write()
            return record


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


store: DocumentStore = DocumentStore(config.data_dir / "documents.json", config.uploads_dir)
