import sys
from dataclasses import asdict
from typing import Any

from blograg.rag import Index, Rag

MAX_LIMIT = 30
DEFAULT_LIMIT = 10


def parse_limit(body: dict[str, Any]) -> int:
    return min(max(int(body.get("limit", DEFAULT_LIMIT)), 1), MAX_LIMIT)


def parse_label(body: dict[str, Any]) -> str | None:
    label = body.get("label")
    return label if isinstance(label, str) and label else None


def parse_year(body: dict[str, Any]) -> int | None:
    year = body.get("year")
    return int(year) if year else None


class Api:
    def __init__(self, rag: Rag, index: Index) -> None:
        self.rag = rag
        self.index = index

    def stats(self) -> dict[str, Any]:
        years = sorted(self.index.facet("year", 100), key=lambda facet: facet.name, reverse=True)
        return {
            "posts": self.index.count_posts(),
            "chunks": self.index.count_chunks(),
            "labels": [asdict(facet) for facet in self.index.facet("labels", 300)],
            "years": [asdict(facet) for facet in years],
        }

    def search(self, body: dict[str, Any]) -> dict[str, Any]:
        hits = self.rag.search(body.get("query"), parse_limit(body), parse_label(body), parse_year(body))
        return {"hits": [asdict(hit) for hit in hits]}

    def ask(self, body: dict[str, Any]) -> dict[str, Any]:
        answer, sources = self.rag.ask(body.get("question"), parse_label(body), parse_year(body))
        return {"answer": answer, "sources": [asdict(hit) for hit in sources]}

    def dispatch(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            match (method, path):
                case ("GET", "/api/stats"):
                    return 200, self.stats()
                case ("POST", "/api/search"):
                    return 200, self.search(body)
                case ("POST", "/api/ask"):
                    return 200, self.ask(body)
                case _:
                    return 404, {"error": "not found"}
        except ValueError as error:
            return 400, {"error": str(error)}
        except Exception as error:
            print(f"{method} {path} failed: {error!r}", file=sys.stderr)
            return 502, {"error": str(error)}
