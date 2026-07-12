import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:latest")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TOP_K = int(os.environ.get("TOP_K", "3"))


def build_engine():
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    print(f"Indexed {len(documents)} documents into {len(index.docstore.docs)} nodes")
    return index.as_query_engine(similarity_top_k=TOP_K, response_mode="compact")


def answer(engine, question):
    response = engine.query(question)
    sources = []
    for node in response.source_nodes:
        sources.append(
            {
                "file": os.path.basename(node.node.metadata.get("file_name", "")),
                "score": round(float(node.score), 4) if node.score is not None else None,
                "text": node.node.get_content().strip(),
            }
        )
    return {"answer": str(response).strip(), "sources": sources}


class Handler(BaseHTTPRequestHandler):
    engine = None

    def _send(self, code, body, content_type="application/json"):
        payload = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"status": "ok", "model": LLM_MODEL}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/query":
            self._send(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            question = json.loads(self.rfile.read(length)).get("question", "").strip()
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "invalid json"}))
            return
        if not question:
            self._send(400, json.dumps({"error": "question is required"}))
            return
        self._send(200, json.dumps(answer(self.engine, question)))

    def log_message(self, *args):
        pass


def main():
    print(f"Loading LlamaIndex with LLM={LLM_MODEL} embeddings={EMBED_MODEL}")
    Handler.engine = build_engine()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serving on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
