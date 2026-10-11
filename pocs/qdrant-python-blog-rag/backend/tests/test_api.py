import unittest
from collections.abc import Sequence

from blograg.api import Api
from blograg.ollama import QUERY_PREFIX
from blograg.rag import NO_MATCH, Rag, build_prompt, unique_posts
from blograg.store import Facet, Hit


def hit(url: str, text: str = "body", title: str = "Title") -> Hit:
    return Hit(url=url, title=title, published="2026-01-01", labels=["AI"], score=0.9, text=text)


class FakeLlm:
    def __init__(self) -> None:
        self.embedded: list[tuple[list[str], str]] = []
        self.prompts: list[str] = []

    def embed(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        self.embedded.append((list(texts), prefix))
        return [[0.1, 0.2] for _ in texts]

    def chat(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return "grounded answer"


class FakeIndex:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.filters: list[tuple[int, str | None, int | None]] = []
        self.failure: Exception | None = None

    def search_posts(self, vector: list[float], limit: int, label: str | None, year: int | None) -> list[Hit]:
        self.filters.append((limit, label, year))
        return self.hits[:limit]

    def search_chunks(self, vector: list[float], limit: int, label: str | None, year: int | None) -> list[Hit]:
        self.filters.append((limit, label, year))
        return self.hits[:limit]

    def count_posts(self) -> int:
        return 2

    def count_chunks(self) -> int:
        return 7

    def facet(self, key: str, limit: int) -> list[Facet]:
        if self.failure:
            raise self.failure
        if key == "year":
            return [Facet("2019", 3), Facet("2026", 4)]
        return [Facet("AI", 5)]


def build(hits: list[Hit]) -> tuple[Api, FakeLlm, FakeIndex]:
    llm = FakeLlm()
    index = FakeIndex(hits)
    return Api(Rag(index, llm), index), llm, index


class PromptTest(unittest.TestCase):
    def test_prompt_carries_every_excerpt_with_its_title_and_url_so_the_model_can_cite(self) -> None:
        prompt = build_prompt("why rust?", [hit("https://blog/a", "rust is fast", "Rust"), hit("https://blog/b", "go is simple", "Go")])
        for expected in ("[1] Rust (https://blog/a)", "rust is fast", "[2] Go (https://blog/b)", "go is simple", "Question: why rust?"):
            self.assertIn(expected, prompt)

    def test_sources_list_each_post_once_even_when_several_chunks_matched(self) -> None:
        sources = unique_posts([hit("https://blog/a", "one"), hit("https://blog/b"), hit("https://blog/a", "two")])
        self.assertEqual([source.url for source in sources], ["https://blog/a", "https://blog/b"])
        self.assertEqual(sources[0].text, "one")


class ApiTest(unittest.TestCase):
    def test_blank_question_is_rejected_before_spending_a_model_call(self) -> None:
        api, llm, _ = build([hit("https://blog/a")])
        for body in ({}, {"question": "   "}, {"question": 42}):
            status, payload = api.dispatch("POST", "/api/ask", body)
            self.assertEqual((status, payload), (400, {"error": "question is required"}))
        self.assertEqual(llm.embedded, [])

    def test_query_is_embedded_with_the_query_prefix_the_embedding_model_expects(self) -> None:
        api, llm, _ = build([hit("https://blog/a")])
        api.dispatch("POST", "/api/search", {"query": " qdrant "})
        self.assertEqual(llm.embedded, [(["qdrant"], QUERY_PREFIX)])

    def test_filters_and_limit_reach_the_vector_search(self) -> None:
        api, _, index = build([hit("https://blog/a")])
        api.dispatch("POST", "/api/search", {"query": "qdrant", "label": "AI", "year": "2026", "limit": 500})
        self.assertEqual(index.filters, [(30, "AI", 2026)])

    def test_ask_answers_from_retrieved_excerpts_and_returns_their_sources(self) -> None:
        api, llm, _ = build([hit("https://blog/a", "rust is fast")])
        status, payload = api.dispatch("POST", "/api/ask", {"question": "why rust?"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["answer"], "grounded answer")
        self.assertEqual([source["url"] for source in payload["sources"]], ["https://blog/a"])
        self.assertIn("rust is fast", llm.prompts[0])

    def test_ask_without_matches_does_not_let_the_model_invent_an_answer(self) -> None:
        api, llm, _ = build([])
        status, payload = api.dispatch("POST", "/api/ask", {"question": "anything"})
        self.assertEqual((status, payload), (200, {"answer": NO_MATCH, "sources": []}))
        self.assertEqual(llm.prompts, [])

    def test_stats_lists_newest_year_first_for_the_filter_dropdown(self) -> None:
        api, _, _ = build([])
        status, payload = api.dispatch("GET", "/api/stats", {})
        self.assertEqual(status, 200)
        self.assertEqual((payload["posts"], payload["chunks"]), (2, 7))
        self.assertEqual([year["name"] for year in payload["years"]], ["2026", "2019"])

    def test_unknown_route_is_not_found(self) -> None:
        api, _, _ = build([])
        self.assertEqual(api.dispatch("GET", "/api/nope", {})[0], 404)

    def test_upstream_failure_is_reported_as_bad_gateway_instead_of_crashing(self) -> None:
        api, _, index = build([])
        index.failure = ConnectionError("qdrant unreachable")
        self.assertEqual(api.dispatch("GET", "/api/stats", {}), (502, {"error": "qdrant unreachable"}))


if __name__ == "__main__":
    unittest.main()
