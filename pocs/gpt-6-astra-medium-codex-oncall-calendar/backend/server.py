import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from schedule import Settings, calendar_file, generate


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if content_type.startswith("text/calendar"):
            self.send_header("Content-Disposition", 'attachment; filename="onward-oncall.ics"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.respond(200, b'{"status":"ok"}')
            return
        if parsed.path == "/api/config":
            self.respond(200, json.dumps({"googleClientId": os.environ.get("GOOGLE_CLIENT_ID", "")}).encode())
            return
        if parsed.path not in ("/api/schedule", "/api/calendar.ics"):
            self.respond(404, b'{"error":"Not found"}')
            return
        try:
            params = parse_qs(parsed.query)
            settings: Settings = {"primary": params["primary"][0], "secondary": params["secondary"][0], "interval": int(params["interval"][0]), "timezone": params["timezone"][0]}
            schedule = generate(settings)
            if parsed.path.endswith(".ics"):
                self.respond(200, calendar_file(schedule).encode(), "text/calendar; charset=utf-8")
            else:
                self.respond(200, json.dumps(schedule.to_dict()).encode())
        except (ValueError, KeyError) as error:
            message = "Provide both dates, interval, and time zone." if isinstance(error, KeyError) else str(error)
            self.respond(400, json.dumps({"error": message}).encode())


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Onward API listening on http://127.0.0.1:8000", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
