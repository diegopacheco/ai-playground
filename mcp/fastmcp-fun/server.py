from fastmcp import FastMCP

mcp = FastMCP("Calculator")

@mcp.tool
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool
def subtract(a: int, b: int) -> int:
    return a - b

@mcp.tool
def multiply(a: int, b: int) -> int:
    return a * b

@mcp.tool
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    return f"Hello, {name}! Welcome to FastMCP."

@mcp.prompt()
def math_prompt(operation: str, a: str, b: str) -> str:
    return f"Please {operation} the numbers {a} and {b} using the available tools."

if __name__ == "__main__":
    mcp.run()
