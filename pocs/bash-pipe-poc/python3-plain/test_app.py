import json
import socket
import threading
import time
import urllib.request
from http.server import HTTPServer

from app import Handler


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Server:
    def __init__(self):
        self.port = _free_port()
        self.srv = HTTPServer(("127.0.0.1", self.port), Handler)
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.t.start()
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=0.5)
                break
            except Exception:
                time.sleep(0.05)
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


def test_health_endpoint_returns_200():
    with _Server() as s:
        r = urllib.request.urlopen(f"http://127.0.0.1:{s.port}/health")
        assert r.status == 200


def test_health_payload_shape():
    with _Server() as s:
        r = urllib.request.urlopen(f"http://127.0.0.1:{s.port}/health")
        body = json.loads(r.read().decode("utf-8"))
        assert body == {"status": "ok"}


def test_root_or_app_specific_smoke():
    with _Server() as s:
        r = urllib.request.urlopen(f"http://127.0.0.1:{s.port}/")
        assert r.status == 200
        assert r.read() == b"python3-plain"
