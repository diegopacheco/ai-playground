import os
import sys
from collections.abc import Callable
from fractions import Fraction

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

from calculator.arithmetic import CalculatorError, Number, apply, to_fraction, to_number
from calculator.english import evaluate_english
from calculator.infix import evaluate_infix
from calculator.polish import evaluate_polish

server = MCPServer(
    name="calculator",
    version="0.1.0",
    instructions="Calculator with +, -, *, /, infix expressions, Polish notation and plain English.",
)


def calculate(compute: Callable[[], Fraction]) -> Number:
    try:
        return to_number(compute())
    except CalculatorError as error:
        raise ToolError(str(error)) from error


def binary(operator: str, a: float, b: float) -> Number:
    return calculate(lambda: apply(operator, to_fraction(a), to_fraction(b)))


@server.tool(description="Add two numbers: a + b")
async def add(a: float, b: float) -> Number:
    return binary("+", a, b)


@server.tool(description="Subtract two numbers: a - b")
async def subtract(a: float, b: float) -> Number:
    return binary("-", a, b)


@server.tool(description="Multiply two numbers: a * b")
async def multiply(a: float, b: float) -> Number:
    return binary("*", a, b)


@server.tool(description="Divide two numbers: a / b")
async def divide(a: float, b: float) -> Number:
    return binary("/", a, b)


@server.tool(description="Evaluate an infix expression such as '(1 + 2) * 3'")
async def evaluate(expression: str) -> Number:
    return calculate(lambda: evaluate_infix(expression))


@server.tool(description="Evaluate a Polish (prefix) notation expression such as '* + 1 2 3'")
async def polish(expression: str) -> Number:
    return calculate(lambda: evaluate_polish(expression))


@server.tool(description="Evaluate basic English such as 'one plus two' or 'ten divided by four'")
async def english(text: str) -> Number:
    return calculate(lambda: evaluate_english(text))


def transport_security(public_host: str) -> TransportSecuritySettings:
    hosts = ["localhost:*", "127.0.0.1:*"]
    origins: list[str] = []
    if public_host:
        hosts += [public_host, f"{public_host}:*"]
        origins.append(f"https://{public_host}")
    return TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)


def main() -> None:
    if "--stdio" in sys.argv:
        server.run("stdio")
        return
    server.run(
        "streamable-http",
        host=os.environ.get("MCP_HOST", "127.0.0.1"),
        port=int(os.environ["MCP_PORT"]),
        transport_security=transport_security(os.environ.get("MCP_PUBLIC_HOST", "")),
    )


if __name__ == "__main__":
    main()
