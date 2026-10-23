from pathlib import Path

import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.types import TextContent, TextResourceContents
from starlette.testclient import TestClient

from calculator_app.server import build_server
from calculator_app.views import VIEW_URI

pytestmark = pytest.mark.anyio

VIEW_HTML = "<!doctype html><html><body><div id='root'></div></body></html>"
HOST_HTML = "<!doctype html><html><body>host</body></html>"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    (tmp_path / "view.html").write_text(VIEW_HTML, encoding="utf-8")
    (tmp_path / "host.html").write_text(HOST_HTML, encoding="utf-8")
    return tmp_path


@pytest.fixture
def server(dist: Path) -> MCPServer:
    return build_server(dist)


async def test_calculate_tool_points_hosts_to_the_ui_resource(server: MCPServer):
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {"calculate"}
    meta = tools["calculate"].meta or {}
    assert meta["ui"] == {"resourceUri": VIEW_URI, "visibility": ["model", "app"]}


async def test_ui_resource_is_served_with_the_mcp_app_mime_type(server: MCPServer):
    async with Client(server) as client:
        listed = {str(resource.uri): resource for resource in (await client.list_resources()).resources}
        contents = (await client.read_resource(VIEW_URI)).contents
    assert listed[VIEW_URI].mime_type == "text/html;profile=mcp-app"
    page = contents[0]
    assert isinstance(page, TextResourceContents)
    assert page.mime_type == "text/html;profile=mcp-app"
    assert page.text == VIEW_HTML


async def test_calculate_returns_structured_content_for_the_view(server: MCPServer):
    async with Client(server) as client:
        result = await client.call_tool("calculate", {"expression": "(2 + 3) * 4"})
    assert not result.is_error
    assert result.structured_content == {
        "expression": "(2 + 3) * 4",
        "result": "20",
        "steps": ["2 + 3 = 5", "5 * 4 = 20"],
    }


async def test_calculate_reports_bad_input_as_a_tool_error_not_a_crash(server: MCPServer):
    async with Client(server) as client:
        result = await client.call_tool("calculate", {"expression": "1 / 0"})
    text = " ".join(item.text for item in result.content if isinstance(item, TextContent))
    assert result.is_error
    assert "division by zero" in text


def test_root_serves_the_local_host_page_and_health(server: MCPServer):
    with TestClient(server.streamable_http_app()) as browser:
        assert browser.get("/").text == HOST_HTML
        assert browser.get("/health").json() == {"status": "UP"}
