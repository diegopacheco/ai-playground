import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-06-18"


def number_pair(description):
    return {
        "type": "object",
        "properties": {"a": {"type": "number", "description": description[0]}, "b": {"type": "number", "description": description[1]}},
        "required": ["a", "b"],
    }


def divide(a, b):
    if b == 0:
        raise ValueError("division by zero")
    return a / b


TOOLS = {
    "add": ("Add two numbers, returns a + b", number_pair(("first number", "second number")), lambda a, b: a + b),
    "subtract": ("Subtract two numbers, returns a - b", number_pair(("number to subtract from", "number to subtract")), lambda a, b: a - b),
    "multiply": ("Multiply two numbers, returns a * b", number_pair(("first factor", "second factor")), lambda a, b: a * b),
    "divide": ("Divide two numbers, returns a / b", number_pair(("dividend", "divisor")), divide),
}


def tidy(value):
    return int(value) if isinstance(value, float) and value.is_integer() else value


def as_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a number")
    try:
        return float(value) if isinstance(value, str) else value
    except ValueError:
        raise ValueError(f"{name} must be a number") from None


def calculate(name, arguments):
    if name not in TOOLS:
        raise KeyError(name)
    a = as_number(arguments.get("a"), "a")
    b = as_number(arguments.get("b"), "b")
    return tidy(TOOLS[name][2](a, b))


class CallLog:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []

    def record(self, name, arguments, result, failed):
        with self.lock:
            self.calls.append({"seq": len(self.calls) + 1, "tool": name, "arguments": arguments, "result": result, "error": failed})

    def since(self, seq):
        with self.lock:
            return [call for call in self.calls if call["seq"] > seq]

    def last(self):
        with self.lock:
            return len(self.calls)


def tool_list():
    return {"tools": [{"name": name, "description": description, "inputSchema": schema} for name, (description, schema, _) in TOOLS.items()]}


def call_tool(params, log):
    name = params.get("name", "")
    arguments = params.get("arguments") or {}
    try:
        result = calculate(name, arguments)
    except KeyError:
        return None
    except ValueError as failure:
        log.record(name, arguments, str(failure), True)
        return {"content": [{"type": "text", "text": str(failure)}], "isError": True}
    log.record(name, arguments, result, False)
    return {"content": [{"type": "text", "text": str(result)}], "structuredContent": {"result": result}, "isError": False}


def rpc_result(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_rpc(message, log):
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
        return rpc_error(message.get("id") if isinstance(message, dict) else None, -32600, "invalid request")
    if "id" not in message:
        return None
    request_id = message["id"]
    method = message["method"]
    params = message.get("params") or {}
    if method == "initialize":
        return rpc_result(request_id, {"protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION), "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "math", "version": "1.0.0"}})
    if method == "ping":
        return rpc_result(request_id, {})
    if method == "tools/list":
        return rpc_result(request_id, tool_list())
    if method == "tools/call":
        result = call_tool(params, log)
        if result is None:
            return rpc_error(request_id, -32602, f"unknown tool {params.get('name', '')}")
        return rpc_result(request_id, result)
    return rpc_error(request_id, -32601, f"method not found {method}")


def handle_body(payload, log):
    if isinstance(payload, list):
        replies = [reply for reply in (handle_rpc(message, log) for message in payload) if reply is not None]
        return replies or None
    return handle_rpc(payload, log)


def handler(log):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/health":
                self.send_json(200, {"status": "ok", "tools": sorted(TOOLS)})
            elif path == "/calls":
                since = int(dict(part.split("=", 1) for part in query.split("&") if "=" in part).get("since", "0") or 0)
                self.send_json(200, {"last": log.last(), "calls": log.since(since)})
            elif path == "/mcp":
                self.send_json(405, {"error": "this server does not open an SSE stream"})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/mcp":
                self.send_json(404, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
            except json.JSONDecodeError:
                self.send_json(400, rpc_error(None, -32700, "parse error"))
                return
            reply = handle_body(payload, log)
            if reply is None:
                self.respond(202, b"", None)
                return
            session = self.headers.get("Mcp-Session-Id") or uuid.uuid4().hex
            self.send_json(200, reply, {"Mcp-Session-Id": session})

        def do_DELETE(self):
            self.respond(200, b"", None)

        def send_json(self, status, payload, headers=None):
            self.respond(status, json.dumps(payload).encode("utf-8"), "application/json", headers)

        def respond(self, status, data, content_type, headers=None):
            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            sys.stderr.write(f"mcp {self.address_string()} {fmt % args}\n")

    return Handler


def main():
    port = int(os.environ.get("MCP_PORT", "8093"))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(CallLog()))
    print(f"math mcp server listening on http://127.0.0.1:{port}/mcp", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
