from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import webbrowser
import os
import signal
import argparse
import sys

ROOT = Path(__file__).resolve().parent
PID_FILE = Path.home() / ".claude" / "jit-testing" / "dashboard.pid"

class Handler(BaseHTTPRequestHandler):
    repo: Path = Path(".")

    def log_message(self, *args, **kwargs):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            return self._serve_file(ROOT / "index.html", "text/html; charset=utf-8")
        if self.path == "/style.css":
            return self._serve_file(ROOT / "style.css", "text/css; charset=utf-8")
        if self.path == "/app.js":
            return self._serve_file(ROOT / "app.js", "application/javascript; charset=utf-8")
        if self.path == "/api/runs":
            return self._json(_list_runs(self.repo))
        if self.path.startswith("/api/run/"):
            rid = self.path.split("/api/run/", 1)[1]
            return self._json(_load_run(self.repo, rid))
        if self.path == "/api/summary":
            return self._json(_summary(self.repo))
        self._not_found()

    def do_POST(self):
        if self.path.startswith("/api/verdict/"):
            parts = self.path.split("/")
            if len(parts) >= 5:
                rid, cid = parts[3], parts[4]
                length = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(length).decode() if length else "{}"
                try:
                    data = json.loads(body)
                except Exception:
                    data = {}
                _set_verdict(self.repo, rid, cid, data.get("verdict", "deferred"))
                return self._json({"ok": True})
        self._not_found()

    def _serve_file(self, path: Path, ctype: str):
        if not path.exists():
            return self._not_found()
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")

def _list_runs(repo: Path):
    base = repo / ".jit-testing" / "runs"
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir(), reverse=True):
        meta_file = d / "catches.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        catches = meta.get("catches", [])
        out.append({
            "run_id": meta.get("run_id"),
            "target": meta.get("target"),
            "workflow": meta.get("workflow"),
            "diff": meta.get("diff"),
            "started_at": meta.get("started_at"),
            "total": len(catches),
            "surfaced": sum(1 for c in catches if c.get("rubfake", {}).get("score", 0) > 0.3),
            "dismissed": sum(1 for c in catches if c.get("verdict") == "expected"),
            "confirmed": sum(1 for c in catches if c.get("verdict") == "confirmed"),
        })
    return out

def _load_run(repo: Path, rid: str):
    f = repo / ".jit-testing" / "runs" / rid / "catches.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())

def _set_verdict(repo: Path, rid: str, cid: str, verdict: str):
    import time
    f = repo / ".jit-testing" / "runs" / rid / "catches.json"
    if not f.exists():
        return
    meta = json.loads(f.read_text())
    for c in meta.get("catches", []):
        if c.get("id") == cid:
            c["verdict"] = verdict
            c["verdict_ts"] = int(time.time())
            break
    f.write_text(json.dumps(meta, indent=2))

def _summary(repo: Path):
    runs = _list_runs(repo)
    total_catches = sum(r["total"] for r in runs)
    by_target = {}
    for r in runs:
        by_target[r["target"]] = by_target.get(r["target"], 0) + r["total"]
    return {
        "runs": len(runs),
        "total_catches": total_catches,
        "surfaced": sum(r["surfaced"] for r in runs),
        "confirmed": sum(r["confirmed"] for r in runs),
        "dismissed": sum(r["dismissed"] for r in runs),
        "by_target": by_target,
        "trend": [r["total"] for r in reversed(runs)][-20:],
    }

def _free_port(start=8765):
    for p in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return 0

def serve(args):
    p = argparse.ArgumentParser(prog="jit dashboard")
    p.add_argument("--repo", default=".")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-browser", action="store_true")
    ns = p.parse_args(args)

    repo = Path(ns.repo).resolve()
    Handler.repo = repo
    port = ns.port or _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    print(f"dashboard: {url}")
    print(f"repo: {repo}")
    print("ctrl-c to stop")
    if not ns.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()

def stop_server():
    if not PID_FILE.exists():
        print("no running dashboard")
        return
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"stopped pid {pid}")
    except Exception as e:
        print(f"could not stop: {e}")
    if PID_FILE.exists():
        PID_FILE.unlink()
