# FastMCP Fun

A Python 3.13 POC using FastMCP - the fast, Pythonic way to build MCP (Model Context Protocol) servers and clients.

FastMCP does **NOT** use `litellm` or `dspy`.

## What it does

* **server.py** — MCP server exposing calculator tools (add, subtract, multiply, divide), a greeting resource template, and a math prompt.
* **client.py** — MCP client that connects to the server, lists tools/resources/prompts, and calls each one printing results.

## How to run

```bash
./install-deps.sh
./run.sh
```

## Output
```
Available tools:
  - add: ...
  - subtract: ...
  - multiply: ...
  - divide: ...

Calling tools:
  add(10, 5) = 15
  subtract(10, 5) = 5
  multiply(10, 5) = 50
  divide(10, 3) = 3.3333333333333335

Resource templates: 1
  greeting://World = Hello, World! Welcome to FastMCP.

Prompts:
  - math_prompt
  math_prompt result: Please add the numbers 42 and 58 using the available tools.
```
