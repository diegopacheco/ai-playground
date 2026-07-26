from __future__ import annotations

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_ids: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="semantic", pattern="^(semantic|keyword)$")
    top_k: int = Field(default=10, ge=1, le=50)
    doc_ids: list[str] = Field(default_factory=list)


class RustRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class AgentAskRequest(BaseModel):
    prompt: str = Field(min_length=1)
    agent: str | None = None
    context: str | None = None


class AgentConfigRequest(BaseModel):
    active: str | None = None
    models: dict[str, str] | None = None
    timeout: int | None = Field(default=None, ge=10, le=900)


class MarkRequest(BaseModel):
    page: int = Field(ge=1)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)
    color: str = Field(default="#ffe066")
    note: str = ""
    kind: str = Field(default="highlight", pattern="^(highlight|note)$")


class AnnotateRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    marks: list[MarkRequest] = Field(min_length=1)
