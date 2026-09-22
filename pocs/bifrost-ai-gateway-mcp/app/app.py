import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

INDEX = Path(__file__).resolve().parent / "index.html"

PROVIDERS = ["claude-cli", "codex-cli", "agy-cli", "ollama-cli"]


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def http_post(url, payload, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as failure:
        with failure:
            return failure.code, json.loads(failure.read() or b"{}")


def gateway_error(body):
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or body)


class ToolLog:
    def __init__(self, base_url, get=http_get):
        self.base_url = base_url.rstrip("/")
        self.get = get

    def last(self):
        return self.get(f"{self.base_url}/calls?since=0")["last"]

    def since(self, seq):
        return [{key: call[key] for key in ("tool", "arguments", "result", "error")} for call in self.get(f"{self.base_url}/calls?since={seq}")["calls"]]


class Gateway:
    def __init__(self, base_url, post=http_post, tool_log=None):
        self.base_url = base_url.rstrip("/")
        self.post = post
        self.tool_log = tool_log

    def ask(self, provider, model, question):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider}")
        if not model.strip():
            raise ValueError("model is required")
        if not question.strip():
            raise ValueError("question is required")
        seq = self.tool_log.last() if self.tool_log else 0
        started = time.monotonic()
        status, body = self.post(f"{self.base_url}/v1/chat/completions", {"model": f"{provider}/{model}", "messages": [{"role": "user", "content": question}]})
        elapsed = round((time.monotonic() - started) * 1000)
        if status != 200:
            raise RuntimeError(f"bifrost returned {status}: {gateway_error(body)}")
        extra = body.get("extra_fields", {})
        return {
            "answer": body["choices"][0]["message"]["content"],
            "provider": extra.get("provider", provider),
            "model": body.get("model", model),
            "latency_ms": elapsed,
            "tool_calls": self.tool_log.since(seq) if self.tool_log else [],
        }


def handler(gateway):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.respond(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/providers":
                self.send_json(200, {"providers": PROVIDERS})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/ask":
                self.send_json(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self.send_json(200, gateway.ask(str(body.get("provider", "")), str(body.get("model", "")), str(body.get("question", ""))))
            except (ValueError, json.JSONDecodeError) as failure:
                self.send_json(400, {"error": str(failure)})
            except (RuntimeError, OSError) as failure:
                self.send_json(502, {"error": str(failure)})

        def send_json(self, status, payload):
            self.respond(status, json.dumps(payload).encode("utf-8"), "application/json")

        def respond(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            sys.stderr.write(f"app {self.address_string()} {fmt % args}\n")

    return Handler


def main():
    port = int(os.environ.get("APP_PORT", "8092"))
    gateway = Gateway(os.environ.get("BIFROST_URL", "http://127.0.0.1:8080"), tool_log=ToolLog(os.environ.get("MCP_URL", "http://127.0.0.1:8093")))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(gateway))
    print(f"app listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
