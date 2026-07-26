from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .. import rustllama
from ..schemas import RustRequest
from ..storage import store

router = APIRouter(prefix="/api/rust", tags=["rust"])


@router.get("/status")
def status() -> dict[str, Any]:
    return rustllama.status()


@router.post("/ask")
def ask(request: RustRequest) -> dict[str, Any]:
    record = store.get(request.doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    try:
        result = rustllama.run(store.path_for(request.doc_id), request.question)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error))
    return {"file_name": record.file_name, **result}
