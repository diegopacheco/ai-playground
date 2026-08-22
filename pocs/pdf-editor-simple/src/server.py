import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

import render
from document import Document

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}

document = Document()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            return self._file("index.html")
        if self.path in ("/app.css", "/app.js"):
            return self._file(self.path.lstrip("/"))
        if self.path == "/state":
            return self._json(document.state())
        if self.path.startswith("/thumb/"):
            return self._thumbnail()
        if self.path.startswith("/view/"):
            return self._preview()
        if self.path.startswith("/runs"):
            return self._runs()
        if self.path == "/save":
            return self._save()
        self.send_error(404)

    def do_POST(self):
        try:
            if self.path == "/open":
                document.open(self.headers.get("X-Filename", "document.pdf"), self._body())
            elif self.path == "/add":
                document.add(self._body())
            elif self.path == "/op":
                self._operate(json.loads(self._body() or b"{}"))
            elif self.path == "/text":
                request = json.loads(self._body() or b"{}")
                document.edit_text(int(request["page"]), request.get("edits", {}))
            else:
                return self.send_error(404)
        except Exception as error:
            return self._json({"error": str(error)}, status=400)
        self._json(document.state())

    def _operate(self, request):
        name = request.get("op")
        uids = request.get("pages", [])
        if name == "delete":
            document.delete(uids)
        elif name == "keep":
            document.keep(uids)
        elif name == "rotate":
            document.rotate(uids, int(request.get("angle", 90)))
        elif name == "reorder":
            document.reorder(uids)
        elif name == "undo":
            document.undo()
        else:
            raise ValueError(f"unknown operation '{name}'")

    def _runs(self):
        uid = int(self.path.split("uid=")[1])
        try:
            width, height, runs = document.runs(uid)
        except ValueError as error:
            return self._json({"error": str(error)}, status=400)
        self._json({
            "width": width,
            "height": height,
            "runs": [
                {"id": run["id"], "text": run["text"], "box": [round(value, 2) for value in run["box"]],
                 "size": round(run["size"], 2), "mode": run["mode"]}
                for run in runs
            ],
        })

    def _preview(self):
        source, page = self.path[len("/view/"):].split("?")[0].replace(".png", "").split("/")
        try:
            data = render.preview(document.sources, int(source), int(page))
        except (IndexError, ValueError):
            return self.send_error(404)
        self._send(data, "image/png", {"Cache-Control": "no-store"})

    def _thumbnail(self):
        source, page = self.path[len("/thumb/"):].split("?")[0].replace(".png", "").split("/")
        try:
            data = document.thumbnail(int(source), int(page))
        except (IndexError, ValueError):
            return self.send_error(404)
        self._send(data, "image/png", {"Cache-Control": "max-age=86400"})

    def _save(self):
        if not document.state()["pages"]:
            return self.send_error(404)
        name = document.name.removesuffix(".pdf") + "-edited.pdf"
        self._send(document.save(), "application/pdf", {"Content-Disposition": f'attachment; filename="{name}"'})

    def _file(self, name):
        with open(os.path.join(WEB_DIR, name), "rb") as handle:
            self._send(handle.read(), TYPES[os.path.splitext(name)[1]])

    def _json(self, payload, status=200):
        self._send(json.dumps(payload).encode(), "application/json", status=status)

    def _send(self, data, content_type, headers=None, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def log_message(self, *args):
        pass


def main():
    port = int(os.environ.get("PORT", "8099"))
    print(f"pdf editor on http://127.0.0.1:{port}", flush=True)
    try:
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
