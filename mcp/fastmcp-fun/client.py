import asyncio
from fastmcp import Client

async def main():
    async with Client("server.py") as client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool.name}")

        print("\nCalling tools:")

        result = await client.call_tool("add", {"a": 10, "b": 5})
        print(f"  add(10, 5) = {result.content[0].text}")

        result = await client.call_tool("subtract", {"a": 10, "b": 5})
        print(f"  subtract(10, 5) = {result.content[0].text}")

        result = await client.call_tool("multiply", {"a": 10, "b": 5})
        print(f"  multiply(10, 5) = {result.content[0].text}")

        result = await client.call_tool("divide", {"a": 10, "b": 3})
        print(f"  divide(10, 3) = {result.content[0].text}")

        templates = await client.list_resource_templates()
        print(f"\nResource templates: {len(templates)}")

        resource = await client.read_resource("greeting://World")
        print(f"  greeting://World = {resource[0].text}")

        prompts = await client.list_prompts()
        print(f"\nPrompts:")
        for p in prompts:
            print(f"  - {p.name}")

        prompt_result = await client.get_prompt("math_prompt", {
            "operation": "add",
            "a": "42",
            "b": "58"
        })
        print(f"  math_prompt result: {prompt_result.messages[0].content.text}")

if __name__ == "__main__":
    asyncio.run(main())
