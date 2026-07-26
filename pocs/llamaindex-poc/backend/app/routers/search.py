from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ..indexing import service
from ..rag import as_source
from ..schemas import SearchRequest

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("")
def search(request: SearchRequest) -> dict[str, Any]:
    started = time.monotonic()
    doc_ids = request.doc_ids or None
    try:
        if request.mode == "keyword":
            nodes = service.scan(request.query, doc_ids)[: request.top_k]
        else:
            nodes = service.retrieve(request.query, request.top_k, doc_ids)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error))

    return {
        "mode": request.mode,
        "hits": [as_source(node, position) for position, node in enumerate(nodes, start=1)],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
