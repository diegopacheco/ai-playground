import json
from collections.abc import Sequence
from typing import Any
from urllib.request import Request, urlopen

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Ollama:
    def __init__(self, base_url: str, embed_model: str, chat_model: str) -> None:
        self.base_url = base_url
        self.embed_model = embed_model
        self.chat_model = chat_model

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=300) as response:
            result: dict[str, Any] = json.load(response)
        return result

    def embed(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        body = {"model": self.embed_model, "input": [prefix + text for text in texts]}
        vectors: list[list[float]] = self._post("/api/embed", body)["embeddings"]
        return vectors

    def chat(self, system: str, user: str) -> str:
        body = {
            "model": self.chat_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        answer: str = self._post("/api/chat", body)["message"]["content"]
        return answer.strip()
