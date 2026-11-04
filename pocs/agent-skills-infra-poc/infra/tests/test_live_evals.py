import unittest

from tests.live import SETTINGS, bearer, http

EVALS = SETTINGS.evals_url


def post_run(client: str, outputs: list[str]) -> dict:
    cases = [{"name": f"c{i}", "output": o, "checks": [{"type": "contains", "value": "ok"}]} for i, o in enumerate(outputs)]
    status, body = http("POST", f"{EVALS}/v1/evals/runs", {"suite": "itest-suite", "cases": cases}, bearer(client))
    assert status == 201, body
    return body


class EvalsLiveTest(unittest.TestCase):
    def test_score_is_the_share_of_cases_where_every_check_passed(self) -> None:
        run = post_run("itest-alpha", ["ok", "ok", "nope"])
        self.assertEqual((run["passed"], run["failed"]), (2, 1))
        self.assertAlmostEqual(run["score"], 0.6667, places=4)
        _, stored = http("GET", f"{EVALS}/v1/evals/runs/{run['id']}", headers=bearer("itest-alpha"))
        self.assertEqual([c["passed"] for c in stored["cases"]], [True, True, False])

    def test_runs_are_private_to_the_tenant_but_visible_to_the_viewer(self) -> None:
        alpha_run = post_run("itest-alpha", ["ok"])
        beta_run = post_run("itest-beta", ["ok"])
        _, beta_list = http("GET", f"{EVALS}/v1/evals/runs?suite=itest-suite&limit=500", headers=bearer("itest-beta"))
        beta_ids = {r["id"] for r in beta_list["items"]}
        self.assertIn(beta_run["id"], beta_ids)
        self.assertNotIn(alpha_run["id"], beta_ids)
        self.assertEqual(http("GET", f"{EVALS}/v1/evals/runs/{alpha_run['id']}", headers=bearer("itest-beta"))[0], 404)
        _, viewer_list = http("GET", f"{EVALS}/v1/evals/runs?suite=itest-suite&limit=500", headers=bearer("eval-viewer"))
        viewer_ids = {r["id"] for r in viewer_list["items"]}
        self.assertTrue({alpha_run["id"], beta_run["id"]} <= viewer_ids)

    def test_viewer_cannot_write_runs(self) -> None:
        body = {"suite": "s", "cases": [{"output": "ok", "checks": [{"type": "contains", "value": "ok"}]}]}
        self.assertEqual(http("POST", f"{EVALS}/v1/evals/runs", body, bearer("eval-viewer"))[0], 403)

    def test_invalid_suite_names_are_rejected(self) -> None:
        body = {"suite": "bad name!", "cases": [{"output": "ok", "checks": [{"type": "contains", "value": "ok"}]}]}
        self.assertEqual(http("POST", f"{EVALS}/v1/evals/runs", body, bearer("itest-alpha"))[0], 400)


if __name__ == "__main__":
    unittest.main()
