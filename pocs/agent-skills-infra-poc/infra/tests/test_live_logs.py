import time
import unittest
import uuid
from collections.abc import Callable

from tests.live import SETTINGS, bearer, http

LOGS = SETTINGS.logs_url


def messages(headers: dict[str, str], level: str | None = None) -> list[str]:
    url = f"{LOGS}/v1/logs?since_minutes=5&limit=500" + (f"&level={level}" if level else "")
    status, body = http("GET", url, headers=headers)
    assert status == 200, body
    return [item["message"] for item in body["items"]]


def eventually(predicate: Callable[[], bool], tries: int = 10) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        time.sleep(0.5)
    return False


class LogsIsolationLiveTest(unittest.TestCase):
    def test_a_tenant_reads_its_own_lines_and_never_another_tenants(self) -> None:
        alpha, beta = bearer("itest-alpha"), bearer("itest-beta")
        marker = f"itest-marker-{uuid.uuid4().hex}"
        status, body = http("POST", f"{LOGS}/v1/logs", {"level": "warn", "message": marker, "fields": {"n": 1}}, alpha)
        self.assertEqual(status, 202, body)
        self.assertTrue(eventually(lambda: marker in messages(alpha)))
        self.assertTrue(eventually(lambda: marker in messages(alpha, "warn")))
        self.assertNotIn(marker, messages(alpha, "error"))
        self.assertNotIn(marker, messages(beta))

    def test_writing_needs_logs_write_and_bad_levels_are_rejected(self) -> None:
        self.assertEqual(http("POST", f"{LOGS}/v1/logs", {"message": "x"}, bearer("itest-readonly"))[0], 403)
        status, body = http("POST", f"{LOGS}/v1/logs", {"level": "fatal", "message": "x"}, bearer("itest-alpha"))
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_level")


if __name__ == "__main__":
    unittest.main()
