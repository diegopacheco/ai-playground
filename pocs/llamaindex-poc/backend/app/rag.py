from __future__ import annotations

import time
from typing import Any

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore

from .indexing import service

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages below. "
    "Cite the passages you used as [1], [2] and so on. "
    "If the context does not answer the question, say that plainly."
)


def as_source(node: NodeWithScore, position: int) -> dict[str, Any]:
    return {
        "position": position,
        "doc_id": node.metadata.get("doc_id", ""),
        "file_name": node.metadata.get("file_name", "unknown"),
        "page": node.metadata.get("page", 0),
        "score": round(float(node.score or 0.0), 4),
        "text": node.get_content().strip(),
    }


def build_context(nodes: list[NodeWithScore]) -> str:
    blocks = []
    for position, node in enumerate(nodes, start=1):
        header = f"[{position}] {node.metadata.get('file_name')} page {node.metadata.get('page')}"
        blocks.append(f"{header}\n{node.get_content().strip()}")
    return "\n\n".join(blocks)


def answer(
    question: str,
    history: list[dict[str, str]],
    top_k: int,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    nodes = service.retrieve(question, top_k, doc_ids)
    if not nodes:
        return {
            "answer": "Nothing is indexed yet that matches this question. Upload PDFs on the Ingest tab first.",
            "sources": [],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }

    messages = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
    for turn in history[-6:]:
        role = MessageRole.USER if turn.get("role") == "user" else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=turn.get("content", "")))
    messages.append(
        ChatMessage(
            role=MessageRole.USER,
            content=f"CONTEXT:\n{build_context(nodes)}\n\nQUESTION: {question}",
        )
    )

    response = Settings.llm.chat(messages)
    return {
        "answer": str(response.message.content or "").strip(),
        "sources": [as_source(node, position) for position, node in enumerate(nodes, start=1)],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
