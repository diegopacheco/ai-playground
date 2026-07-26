from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..indexing import service
from ..storage import store

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
def list_documents() -> dict[str, Any]:
    return {"documents": [record.to_dict() for record in store.list()], "stats": service.stats()}


@router.post("")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for upload_file in files:
        name = upload_file.filename or "untitled.pdf"
        payload = await upload_file.read()
        if not payload:
            results.append({"file_name": name, "status": "error", "detail": "empty file"})
            continue
        if not payload.startswith(b"%PDF"):
            results.append({"file_name": name, "status": "error", "detail": "not a pdf"})
            continue

        doc_id, path, existing = store.save_pdf(name, payload)
        if existing:
            results.append({"file_name": name, "status": "duplicate", "doc_id": doc_id})
            continue
        try:
            record = service.ingest(doc_id, name, path)
        except Exception as error:
            path.unlink(missing_ok=True)
            results.append({"file_name": name, "status": "error", "detail": str(error)})
            continue
        results.append({"status": "indexed", **record.to_dict()})

    return {"results": results, "stats": service.stats()}


@router.get("/{doc_id}/file")
def download(doc_id: str) -> FileResponse:
    record = store.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return FileResponse(
        store.path_for(doc_id),
        media_type="application/pdf",
        filename=record.file_name,
    )


@router.delete("/{doc_id}")
def remove(doc_id: str) -> dict[str, Any]:
    record = store.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    service.delete(record)
    store.remove(doc_id)
    return {"deleted": doc_id, "stats": service.stats()}
