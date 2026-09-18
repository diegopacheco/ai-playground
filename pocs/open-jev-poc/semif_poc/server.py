import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .decide import decide

ROOT = Path(__file__).resolve().parent
INDEX = (ROOT.parent / "static" / "index.html").read_bytes()
SAMPLES = (ROOT / "samples.json").read_bytes()


def make_handler(model, tokenizer, metadata):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, kind: str = "application/json"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, payload):
            self.send(status, json.dumps(payload).encode())

        def do_GET(self):
            if self.path == "/":
                self.send(200, INDEX, "text/html; charset=utf-8")
            elif self.path == "/api/samples":
                self.send(200, SAMPLES)
            elif self.path == "/api/health":
                self.send_json(200, {"status": "UP", "model": metadata["source"], "device": metadata["device"]})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/decide":
                self.send_json(404, {"error": "not found"})
                return
            try:
                row = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with lock:
                    result = decide(model, tokenizer, metadata, row)
            except (ValueError, TypeError, AttributeError, KeyError) as error:
                self.send_json(400, {"error": str(error)})
                return
            self.send_json(200, result)

    return Handler


def main():
    from .model import load

    port = int(os.environ.get("PORT", "8787"))
    model, tokenizer, metadata = load()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(model, tokenizer, metadata))
    print(f"SemIf POC on http://localhost:{port} using {metadata['source']} on {metadata['device']}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
