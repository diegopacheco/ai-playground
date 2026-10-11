from collections.abc import Sequence
from typing import Protocol

from blograg.ollama import QUERY_PREFIX
from blograg.store import Facet, Hit

CONTEXT_CHUNKS = 6
NO_MATCH = "I could not find any indexed post about that."
SYSTEM = (
    "You answer questions about the blog posts written by Diego Pacheco. "
    "Answer directly, using only the numbered excerpts you receive. "
    "Only when none of the excerpts relate to the question, reply that the blog does not cover it. "
    "Be concise, at most two short paragraphs, and mention the titles of the posts you used."
)


class Llm(Protocol):
    def embed(self, texts: Sequence[str], prefix: str) -> list[list[float]]: ...
    def chat(self, system: str, user: str) -> str: ...


class Index(Protocol):
    def search_posts(self, vector: list[float], limit: int, label: str | None, year: int | None) -> list[Hit]: ...
    def search_chunks(self, vector: list[float], limit: int, label: str | None, year: int | None) -> list[Hit]: ...
    def count_posts(self) -> int: ...
    def count_chunks(self) -> int: ...
    def facet(self, key: str, limit: int) -> list[Facet]: ...


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    excerpts = "\n\n".join(f"[{number}] {hit.title} ({hit.url})\n{hit.text}" for number, hit in enumerate(hits, 1))
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def unique_posts(hits: Sequence[Hit]) -> list[Hit]:
    by_url: dict[str, Hit] = {}
    for hit in hits:
        by_url.setdefault(hit.url, hit)
    return list(by_url.values())


def require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


class Rag:
    def __init__(self, index: Index, llm: Llm) -> None:
        self.index = index
        self.llm = llm

    def _embed_query(self, text: str) -> list[float]:
        return self.llm.embed([text], QUERY_PREFIX)[0]

    def search(self, query: object, limit: int, label: str | None, year: int | None) -> list[Hit]:
        text = require_text(query, "query")
        return self.index.search_posts(self._embed_query(text), limit, label, year)

    def ask(self, question: object, label: str | None, year: int | None) -> tuple[str, list[Hit]]:
        text = require_text(question, "question")
        hits = self.index.search_chunks(self._embed_query(text), CONTEXT_CHUNKS, label, year)
        if not hits:
            return NO_MATCH, []
        return self.llm.chat(SYSTEM, build_prompt(text, hits)), unique_posts(hits)
