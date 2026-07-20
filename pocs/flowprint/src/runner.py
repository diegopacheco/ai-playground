import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


class RunnerBusy(Exception):
    pass


class FlowPrintRunner:
    def __init__(self, project_dir=None, execute=None, runtime_dir=None):
        self.project_dir = Path(project_dir or Path(__file__).resolve().parent.parent)
        self.sample_dir = self.project_dir / "sample"
        self.artifact_dir = Path(runtime_dir or "/tmp/flowprint/run")
        self.report_dir = self.artifact_dir / "playwright-report"
        self.spec_path = self.artifact_dir / "flow.spec.ts"
        self.execute = execute or self.execute_playwright
        self.running = False
        self.last_run = None
        self.lock = threading.Lock()

    def report_available(self):
        return (self.report_dir / "index.html").is_file()

    def status(self):
        with self.lock:
            return {
                "lastRun": self.last_run,
                "reportAvailable": self.report_available(),
                "running": self.running,
            }

    def run(self, spec):
        with self.lock:
            if self.running:
                raise RunnerBusy()
            self.running = True
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.report_dir, ignore_errors=True)
            modules = self.artifact_dir / "node_modules"
            if not modules.exists() and not modules.is_symlink():
                modules.symlink_to(self.sample_dir / "node_modules", target_is_directory=True)
            self.spec_path.write_text(spec, encoding="utf-8")
            exit_code, output = self.execute()
            result = {
                "exitCode": exit_code,
                "finishedAt": datetime.now(timezone.utc).isoformat(),
                "passed": exit_code == 0,
            }
            with self.lock:
                self.last_run = result
            return {
                **result,
                "output": output[-65536:],
                "reportAvailable": self.report_available(),
            }
        finally:
            with self.lock:
                self.running = False

    def execute_playwright(self):
        executable = self.sample_dir / "node_modules" / ".bin" / ("playwright.cmd" if os.name == "nt" else "playwright")
        environment = {
            **os.environ,
            "PLAYWRIGHT_HTML_OPEN": "never",
            "PLAYWRIGHT_HTML_OUTPUT_DIR": str(self.report_dir),
        }
        process = subprocess.run(
            [str(executable), "test", self.spec_path.name, "--reporter=html"],
            cwd=self.artifact_dir,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        return process.returncode, f"{process.stdout}{process.stderr}"


class FlowPrintHandler(BaseHTTPRequestHandler):
    extension_origin = re.compile(r"^chrome-extension://[a-p]{32}$")

    def log_message(self, format, *args):
        return

    def allowed(self):
        origin = self.headers.get("Origin", "")
        return not origin or self.extension_origin.fullmatch(origin)

    def headers_for_origin(self):
        origin = self.headers.get("Origin", "")
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"} if origin else {}

    def send_json(self, status, value):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        for name, content in self.headers_for_origin().items():
            self.send_header(name, content)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        if not self.allowed():
            return self.send_json(403, {"error": "Forbidden origin"})
        self.send_response(204)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        for name, content in self.headers_for_origin().items():
            self.send_header(name, content)
        self.end_headers()

    def do_GET(self):
        if not self.allowed():
            return self.send_json(403, {"error": "Forbidden origin"})
        route = urlparse(self.path).path
        if route == "/status":
            return self.send_json(200, self.server.runner.status())
        if route == "/report":
            self.send_response(302)
            self.send_header("Location", "/report/")
            return self.end_headers()
        if route.startswith("/report/"):
            return self.send_report(route)
        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self.allowed():
            return self.send_json(403, {"error": "Forbidden origin"})
        if urlparse(self.path).path != "/run":
            return self.send_json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1024 * 1024:
                return self.send_json(413, {"error": "Spec is too large"})
            body = json.loads(self.rfile.read(length))
            spec = body.get("spec")
            if not isinstance(spec, str) or not spec.strip():
                return self.send_json(400, {"error": "A generated spec is required"})
            self.send_json(200, self.server.runner.run(spec))
        except RunnerBusy:
            self.send_json(409, {"error": "Playwright is already running"})
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "Invalid JSON"})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def send_report(self, route):
        relative = unquote(route[len("/report/"):]) or "index.html"
        file_path = (self.server.runner.report_dir / relative).resolve()
        report_dir = self.server.runner.report_dir.resolve()
        try:
            file_path.relative_to(report_dir)
            if file_path.is_dir():
                file_path = file_path / "index.html"
            content = file_path.read_bytes()
            media_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (OSError, ValueError):
            self.send_json(404, {"error": "Report not found"})


def create_server(project_dir=None, port=17339, execute=None, runtime_dir=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), FlowPrintHandler)
    server.runner = FlowPrintRunner(project_dir, execute, runtime_dir)
    return server


def main():
    port = 17339
    project_dir = sys.argv[1] if len(sys.argv) > 1 else None
    server = create_server(project_dir=project_dir, port=port)
    sys.stdout.write(f"FlowPrint runner listening at http://127.0.0.1:{port}\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
