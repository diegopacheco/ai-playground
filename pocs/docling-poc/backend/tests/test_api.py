import asyncio
import json

from app.api import Api
from app.asgi import Message
from tests.fakes import FakeEngine, make_service


def call(api: Api, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None, query: str = "") -> tuple[int, dict[str, str], object]:
    sent: list[Message] = []
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query.encode(),
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(api(scope, receive, send))
    start, payload = sent
    response_headers = {k.decode(): v.decode() for k, v in start["headers"]}
    return start["status"], response_headers, json.loads(payload["body"])


def make_api(engine: FakeEngine | None = None, limit: int = 1000) -> Api:
    return Api(make_service(engine or FakeEngine(), limit), limit)


def test_upload_returns_camel_case_document_for_the_ui() -> None:
    status, _, body = call(make_api(), "POST", "/api/conversions", b"%PDF", {"x-filename": "my%20report.pdf"}, "ocr=true")
    assert status == 201
    assert isinstance(body, dict)
    assert body["filename"] == "my report.pdf"
    assert body["ocr"] is True
    assert body["durationMs"] == 250
    assert body["document"]["stats"]["pages"] == 3


def test_upload_without_filename_is_a_client_error() -> None:
    status, _, body = call(make_api(), "POST", "/api/conversions", b"x")
    assert status == 400
    assert body == {"error": "x-filename header is required"}


def test_failed_conversion_maps_to_422() -> None:
    status, _, _ = call(make_api(FakeEngine(fail_with="boom")), "POST", "/api/conversions", b"x", {"x-filename": "a.pdf"})
    assert status == 422


def test_body_over_limit_is_rejected_while_streaming() -> None:
    status, _, _ = call(make_api(limit=2), "POST", "/api/conversions", b"too big", {"x-filename": "a.pdf"})
    assert status == 413


def test_unknown_conversion_is_404() -> None:
    status, _, _ = call(make_api(), "GET", "/api/conversions/missing")
    assert status == 404


def test_correlation_id_is_echoed_so_logs_can_be_joined() -> None:
    _, headers, _ = call(make_api(), "GET", "/api/health", headers={"x-correlation-id": "abc"})
    assert headers["x-correlation-id"] == "abc"


def test_correlation_id_is_created_when_missing() -> None:
    _, headers, _ = call(make_api(), "GET", "/api/health")
    assert len(headers["x-correlation-id"]) == 32
