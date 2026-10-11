import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from blograg import config
from blograg.api import Api
from blograg.ollama import Ollama
from blograg.rag import Rag
from blograg.store import Store


class ApiServer(ThreadingHTTPServer):
    request_queue_size = 128

    def __init__(self, port: int, api: Api) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.api = api


class Handler(BaseHTTPRequestHandler):
    server: ApiServer

    def read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        return body

    def respond(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self.respond(*self.server.api.dispatch("GET", self.path, {}))

    def do_POST(self) -> None:
        try:
            body = self.read_body()
        except ValueError as error:
            self.respond(400, {"error": str(error)})
            return
        self.respond(*self.server.api.dispatch("POST", self.path, body))


def main() -> None:
    settings = config.load()
    store = Store(settings.qdrant_url, settings.collection)
    store.ensure_collection()
    llm = Ollama(settings.ollama_url, settings.embed_model, settings.chat_model)
    server = ApiServer(settings.api_port, Api(Rag(store, llm), store))
    print(f"api listening on http://localhost:{settings.api_port}/api", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
