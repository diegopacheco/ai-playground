from pathlib import Path

from mcp.server import MCPServer
from mcp.server.apps import Apps
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resources import FileResource
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from calculator_app.calculator import Calculation, calculate
from calculator_app.tokens import CalculatorError
from calculator_app.views import HOST_FILE, VIEW_FILE, VIEW_URI, dist_dir, read_page

INSTRUCTIONS = (
    "Visual calculator. Call calculate with an arithmetic expression using + - * / % ^ and parentheses; "
    "the host renders an interactive calculator showing the result and every evaluation step."
)


def build_apps(dist: Path) -> Apps:
    apps = Apps()

    @apps.tool(
        name="calculate",
        resource_uri=VIEW_URI,
        visibility=["model", "app"],
        description="Evaluate an arithmetic expression such as '(2 + 3) * 4 ^ 2' and show it on a visual calculator",
    )
    async def calculate_expression(expression: str) -> Calculation:
        try:
            return calculate(expression)
        except CalculatorError as error:
            raise ToolError(str(error)) from error

    apps.add_resource(
        FileResource(
            uri=VIEW_URI,
            name="calculator_view",
            title="Calculator",
            description="Interactive React calculator with keypad, display and evaluation steps",
            path=dist / VIEW_FILE,
            meta={"ui": {"prefersBorder": True}},
        )
    )
    return apps


def build_server(dist: Path | None = None) -> MCPServer:
    folder = dist or dist_dir()
    server = MCPServer(
        name="calculator-app",
        version="0.1.0",
        instructions=INSTRUCTIONS,
        extensions=[build_apps(folder)],
    )

    @server.custom_route("/", methods=["GET"])
    async def host_page(request: Request) -> Response:
        return HTMLResponse(read_page(folder, HOST_FILE))

    @server.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> Response:
        return JSONResponse({"status": "UP"})

    return server
