import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from semif_phase1.core import validate_row

from semif_poc import server


@pytest.fixture(scope="module")
def base_url():
    def fake_decide(model, tokenizer, metadata, row):
        validate_row(row)
        return {"id": row["id"], "decision": row["options"][0]["id"]}

    original = server.decide
    server.decide = fake_decide
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(None, None, {"source": "fake", "device": "cpu"}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    server.decide = original


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_every_bundled_sample_is_a_valid_semif_row(base_url):
    status, samples = call(base_url + "/api/samples")
    assert status == 200 and len(samples) >= 2
    for row in samples:
        validate_row(row)


def test_decide_returns_the_decision(base_url):
    row = json.loads(server.SAMPLES)[0]
    assert call(base_url + "/api/decide", row) == (200, {"id": row["id"], "decision": row["options"][0]["id"]})


@pytest.mark.parametrize("body", [{"id": "x"}, [], {"id": "x", "state": "s", "question": "q", "options": "yes,no"}])
def test_bad_input_is_a_client_error_not_a_crash(base_url, body):
    status, payload = call(base_url + "/api/decide", body)
    assert status == 400 and payload["error"]


def test_health_reports_model_and_device(base_url):
    assert call(base_url + "/api/health") == (200, {"status": "UP", "model": "fake", "device": "cpu"})


def test_unknown_path_is_404(base_url):
    assert call(base_url + "/nope")[0] == 404
