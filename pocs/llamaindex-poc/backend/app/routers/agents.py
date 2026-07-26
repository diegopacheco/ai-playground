from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .. import agents
from ..schemas import AgentAskRequest, AgentConfigRequest

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("")
def config() -> dict[str, Any]:
    return {"agents": agents.availability(), "preferences": agents.preferences.values()}


@router.put("")
def save(request: AgentConfigRequest) -> dict[str, Any]:
    try:
        preferences = agents.preferences.update(request.active, request.models, request.timeout)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"agents": agents.availability(), "preferences": preferences}


@router.post("/ask")
def ask(request: AgentAskRequest) -> dict[str, Any]:
    prompt = request.prompt
    if request.context:
        prompt = f"{request.context}\n\n{prompt}"
    try:
        return agents.ask(prompt, request.agent)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error))
