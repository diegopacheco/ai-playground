import asyncio

import pytest
from mcp import Client
from mcp.types import TextContent

from calculator.server import server, transport_security

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def call(client: Client, name: str, arguments: dict[str, object]) -> tuple[bool, str]:
    result = await client.call_tool(name, arguments)
    text = " ".join(item.text for item in result.content if isinstance(item, TextContent))
    return bool(result.is_error), text


async def test_server_exposes_every_calculator_tool():
    async with Client(server) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert names == {"add", "subtract", "multiply", "divide", "evaluate", "polish", "english"}


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("add", {"a": 1, "b": 2}, "3"),
        ("subtract", {"a": 1, "b": 2}, "-1"),
        ("multiply", {"a": 1.5, "b": 4}, "6"),
        ("divide", {"a": 1, "b": 4}, "0.25"),
        ("evaluate", {"expression": "(1 + 2) * 3"}, "9"),
        ("polish", {"expression": "* + 1 2 3"}, "9"),
        ("english", {"text": "one plus two"}, "3"),
    ],
)
async def test_tools_return_results_over_mcp(name: str, arguments: dict[str, object], expected: str):
    async with Client(server) as client:
        assert await call(client, name, arguments) == (False, expected)


async def test_division_by_zero_is_reported_as_tool_error_not_crash():
    async with Client(server) as client:
        is_error, text = await call(client, "divide", {"a": 1, "b": 0})
        assert is_error
        assert "division by zero" in text
        assert await call(client, "add", {"a": 2, "b": 2}) == (False, "4")


async def test_concurrent_calls_on_one_session_keep_their_own_results():
    async with Client(server) as client:
        results = await asyncio.gather(*(call(client, "add", {"a": n, "b": n}) for n in range(20)))
    assert results == [(False, str(n * 2)) for n in range(20)]


def test_public_host_is_allowed_only_when_configured():
    local = transport_security("")
    assert local.allowed_hosts == ["localhost:*", "127.0.0.1:*"]
    assert local.allowed_origins == []
    public = transport_security("box.tailnet.ts.net")
    assert "box.tailnet.ts.net" in public.allowed_hosts
    assert "box.tailnet.ts.net:*" in public.allowed_hosts
    assert public.allowed_origins == ["https://box.tailnet.ts.net"]
