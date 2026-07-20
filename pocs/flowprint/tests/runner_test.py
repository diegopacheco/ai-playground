import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from runner import create_server


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        def execute():
            report_dir = self.server.runner.report_dir
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "index.html").write_text("<h1>Passed</h1>", encoding="utf-8")
            return 0, "passed"

        self.server = create_server(self.temp.name, 0, execute, Path(self.temp.name) / "runtime")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, route, method="GET", value=None, headers=None):
        data = json.dumps(value).encode("utf-8") if value is not None else None
        request = urllib.request.Request(f"{self.url}{route}", data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers

    def test_runs_spec_and_serves_report(self):
        status, body, headers = self.request(
            "/run",
            "POST",
            {"spec": "import { test } from '@playwright/test';"},
            {"Content-Type": "application/json", "Origin": "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        )
        result = json.loads(body)
        self.assertEqual(200, status)
        self.assertTrue(result["passed"])
        self.assertTrue(result["reportAvailable"])
        self.assertEqual("chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", headers["Access-Control-Allow-Origin"])
        status, body, unused = self.request("/report/")
        self.assertEqual(200, status)
        self.assertEqual(b"<h1>Passed</h1>", body)

    def test_rejects_empty_spec(self):
        status, unused, headers = self.request("/run", "POST", {"spec": ""}, {"Content-Type": "application/json"})
        self.assertEqual(400, status)

    def test_rejects_web_page_origin(self):
        status, unused, headers = self.request("/status", headers={"Origin": "https://site.test"})
        self.assertEqual(403, status)


if __name__ == "__main__":
    unittest.main()
