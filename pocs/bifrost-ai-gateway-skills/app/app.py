import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

INDEX = Path(__file__).resolve().parent / "index.html"

PROVIDERS = ["claude-cli", "codex-cli", "agy-cli", "ollama-cli"]

SIGNATURE = re.compile(r"^\W*skill:\s*([a-z0-9-]+)", re.IGNORECASE | re.MULTILINE)


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


def skills_used(answer):
    return list(dict.fromkeys(name.lower() for name in SIGNATURE.findall(answer)))


class SkillCatalog:
    def __init__(self, base_url, get=http_get):
        self.base_url = base_url.rstrip("/")
        self.get = get

    def list(self):
        skills = self.get(f"{self.base_url}/api/skills?limit=100&sort_by=name&order=asc")["skills"]
        return [{"name": skill["name"], "description": skill["description"], "version": skill["latest_version"], "files": skill.get("file_count", 0)} for skill in skills]


class Gateway:
    def __init__(self, base_url, post=http_post):
        self.base_url = base_url.rstrip("/")
        self.post = post

    def ask(self, provider, model, question):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider}")
        if not model.strip():
            raise ValueError("model is required")
        if not question.strip():
            raise ValueError("question is required")
        started = time.monotonic()
        status, body = self.post(f"{self.base_url}/v1/chat/completions", {"model": f"{provider}/{model}", "messages": [{"role": "user", "content": question}]})
        elapsed = round((time.monotonic() - started) * 1000)
        if status != 200:
            raise RuntimeError(f"bifrost returned {status}: {gateway_error(body)}")
        extra = body.get("extra_fields", {})
        answer = body["choices"][0]["message"]["content"]
        return {
            "answer": answer,
            "provider": extra.get("provider", provider),
            "model": body.get("model", model),
            "latency_ms": elapsed,
            "skills_used": skills_used(answer),
        }


class Server(ThreadingHTTPServer):
    request_queue_size = 64


def handler(gateway, catalog):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.respond(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/providers":
                self.send_json(200, {"providers": PROVIDERS})
            elif self.path == "/api/skills":
                try:
                    self.send_json(200, {"skills": catalog.list()})
                except OSError as failure:
                    self.send_json(502, {"error": f"could not list skills from bifrost: {failure}"})
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
    port = int(os.environ.get("APP_PORT", "8192"))
    bifrost_url = os.environ.get("BIFROST_URL", "http://127.0.0.1:8180")
    server = Server(("127.0.0.1", port), handler(Gateway(bifrost_url), SkillCatalog(bifrost_url)))
    print(f"app listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
