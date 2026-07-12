import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from model import IrisNetwork


ROOT = Path(__file__).parent
MODEL_PATH = Path("model/artifacts/iris-network.pt")


class Predictor:
    def __init__(self) -> None:
        self.network = None
        self.mean = None
        self.std = None
        self.classes = None
        self.modified = 0

    def load(self) -> None:
        artifact = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        self.network = IrisNetwork()
        self.network.load_state_dict(artifact["state_dict"])
        self.network.eval()
        self.mean = artifact["mean"]
        self.std = artifact["std"]
        self.classes = artifact["classes"]
        self.modified = MODEL_PATH.stat().st_mtime_ns

    def predict(self, values: list[float]) -> dict:
        if self.network is None or self.modified != MODEL_PATH.stat().st_mtime_ns:
            self.load()
        inputs = torch.tensor([values], dtype=torch.float32)
        with torch.no_grad():
            probabilities = torch.softmax(self.network((inputs - self.mean) / self.std), dim=1)[0]
        index = int(probabilities.argmax())
        return {
            "species": self.classes[index],
            "confidence": round(float(probabilities[index]) * 100, 2),
            "probabilities": {name: round(float(value) * 100, 2) for name, value in zip(self.classes, probabilities)},
        }


predictor = Predictor()


class Handler(BaseHTTPRequestHandler):
    def send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/assets/iris-hero.png": ("assets/iris-hero.png", "image/png"),
        }
        if self.path in files:
            name, content_type = files[self.path]
            self.send_file(ROOT / name, content_type)
        elif self.path == "/health":
            self.send_json(200, {"status": "ready", "model": MODEL_PATH.exists()})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            values = [float(payload[field]) for field in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
            if any(value <= 0 or value > 10 for value in values):
                raise ValueError("Measurements must be between 0 and 10 cm")
            self.send_json(200, predictor.predict(values))
        except (ValueError, KeyError, json.JSONDecodeError, FileNotFoundError) as error:
            self.send_json(400, {"error": str(error)})

    def log_message(self, format, *args) -> None:
        print(f"{self.address_string()} {format % args}")


if __name__ == "__main__":
    print("Inference UI is ready at http://localhost:8080")
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
