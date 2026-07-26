from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import annotate
from ..config import config
from ..schemas import AnnotateRequest
from ..storage import store

router = APIRouter(prefix="/api/annotations", tags=["annotations"])


@router.get("")
def saved() -> dict[str, Any]:
    files = sorted(
        config.annotated_dir.glob("*.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return {
        "files": [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in files
        ]
    }


@router.post("")
def save(request: AnnotateRequest) -> dict[str, Any]:
    record = store.get(request.doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")

    marks = [annotate.Mark(**mark.model_dump()) for mark in request.marks]
    try:
        target, applied = annotate.apply(store.path_for(request.doc_id), record.file_name, marks)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    return {
        "name": target.name,
        "applied": applied,
        "size_bytes": target.stat().st_size,
        "url": f"/api/annotations/{target.name}",
    }


@router.get("/{name}")
def download(name: str) -> FileResponse:
    target = (config.annotated_dir / name).resolve()
    if target.parent != config.annotated_dir.resolve() or not target.is_file():
        raise HTTPException(status_code=404, detail="annotated file not found")
    return FileResponse(target, media_type="application/pdf", filename=name)
