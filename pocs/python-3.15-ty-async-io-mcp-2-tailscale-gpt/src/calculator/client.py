import asyncio
import os
import sys
from typing import Any

from mcp import Client
from mcp.types import TextContent


def build_arguments(schema: dict[str, Any], values: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = schema.get("properties", {})
    if len(values) != len(properties):
        raise SystemExit(f"expected {len(properties)} argument(s): {' '.join(properties)}")
    return {
        name: float(value) if spec.get("type") == "number" else value
        for (name, spec), value in zip(properties.items(), values)
    }


async def run(url: str, argv: list[str]) -> int:
    async with Client(url) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        if not argv:
            for tool in tools.values():
                print(f"{tool.name}: {tool.description}")
            return 0
        name, *values = argv
        if name not in tools:
            raise SystemExit(f"unknown tool: {name}")
        result = await client.call_tool(name, build_arguments(tools[name].input_schema, values))
        text = " ".join(item.text for item in result.content if isinstance(item, TextContent))
        print(text, file=sys.stderr if result.is_error else sys.stdout)
        return 1 if result.is_error else 0


def main() -> None:
    url = f"http://127.0.0.1:{os.environ['MCP_PORT']}/mcp"
    sys.exit(asyncio.run(run(url, sys.argv[1:])))


if __name__ == "__main__":
    main()
