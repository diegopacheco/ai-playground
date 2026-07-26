from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .. import rag
from ..schemas import ChatRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        return rag.answer(
            request.question,
            [turn.model_dump() for turn in request.history],
            request.top_k,
            request.doc_ids or None,
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error))
