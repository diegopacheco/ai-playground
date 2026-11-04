import asyncio
import time
import unittest
import uuid

import psycopg

from tests.live import SETTINGS, bearer, http


class AdminScopesLiveTest(unittest.TestCase):
    def test_tenant_tokens_cannot_use_any_admin_endpoint(self) -> None:
        alpha = bearer("itest-alpha")
        for url in [
            f"{SETTINGS.auth_url}/v1/admin/clients",
            f"{SETTINGS.state_url}/v1/admin/state",
            f"{SETTINGS.state_url}/v1/admin/state/summary",
            f"{SETTINGS.logs_url}/v1/admin/logs",
            f"{SETTINGS.logs_url}/v1/admin/logs/summary",
        ]:
            self.assertEqual(http("GET", url, headers=alpha)[0], 403, url)

    def test_admin_reads_everything_but_cannot_write_tenant_data(self) -> None:
        admin = bearer("infra-admin")
        self.assertEqual(http("PUT", f"{SETTINGS.state_url}/v1/state/x", {"value": 1}, admin)[0], 403)
        self.assertEqual(http("POST", f"{SETTINGS.logs_url}/v1/logs", {"message": "x"}, admin)[0], 403)
        body = {"suite": "s", "cases": [{"output": "ok", "checks": [{"type": "contains", "value": "ok"}]}]}
        self.assertEqual(http("POST", f"{SETTINGS.evals_url}/v1/evals/runs", body, admin)[0], 403)

    def test_client_list_never_exposes_secret_material(self) -> None:
        _, body = http("GET", f"{SETTINGS.auth_url}/v1/admin/clients", headers=bearer("infra-admin"))
        ids = {c["client_id"] for c in body["items"]}
        self.assertTrue({"decision-log", "bookmarks-mcp", "infra-admin"} <= ids)
        for client in body["items"]:
            self.assertEqual(set(client), {"client_id", "kind", "description", "scopes"})

    def test_me_describes_the_caller_only(self) -> None:
        _, body = http("GET", f"{SETTINGS.auth_url}/v1/me", headers=bearer("itest-alpha", "state:read"))
        self.assertEqual((body["client_id"], body["kind"], body["scopes"]), ("itest-alpha", "test", ["state:read"]))
        self.assertEqual(http("GET", f"{SETTINGS.auth_url}/v1/me")[0], 401)


class AdminStateLiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.key = f"itest/admin-{uuid.uuid4().hex}"
        self.alpha, self.beta = bearer("itest-alpha"), bearer("itest-beta")
        http("PUT", f"{SETTINGS.state_url}/v1/state/{self.key}", {"value": "a"}, self.alpha)
        http("PUT", f"{SETTINGS.state_url}/v1/state/{self.key}", {"value": "b"}, self.beta)

    def tearDown(self) -> None:
        http("DELETE", f"{SETTINGS.state_url}/v1/state/{self.key}", headers=self.alpha)
        http("DELETE", f"{SETTINGS.state_url}/v1/state/{self.key}", headers=self.beta)

    def test_all_apps_shows_both_tenants_and_the_filter_shows_one(self) -> None:
        admin = bearer("infra-admin")
        _, everything = http("GET", f"{SETTINGS.state_url}/v1/admin/state?prefix={self.key}", headers=admin)
        self.assertEqual(sorted((i["tenant"], i["value"]) for i in everything["items"]), [("itest-alpha", "a"), ("itest-beta", "b")])
        _, only_beta = http("GET", f"{SETTINGS.state_url}/v1/admin/state?prefix={self.key}&tenant=itest-beta", headers=admin)
        self.assertEqual([(i["tenant"], i["value"]) for i in only_beta["items"]], [("itest-beta", "b")])

    def test_summary_counts_keys_per_tenant(self) -> None:
        _, body = http("GET", f"{SETTINGS.state_url}/v1/admin/state/summary", headers=bearer("infra-admin"))
        tenants = {row["tenant"]: row for row in body["items"]}
        self.assertGreaterEqual(tenants["itest-alpha"]["keys"], 1)
        self.assertIn(self.key, tenants["itest-beta"]["recent_keys"])


class ReadAllPolicyLiveTest(unittest.TestCase):
    def test_read_all_grants_select_only_never_cross_tenant_writes(self) -> None:
        async def scenario() -> tuple[int, int]:
            key = f"itest/{uuid.uuid4().hex}"
            async with await psycopg.AsyncConnection.connect(SETTINGS.database_url, autocommit=True) as conn:
                await conn.execute("SELECT set_config('app.tenant', 'itest-alpha', false)")
                await conn.execute("INSERT INTO kv (tenant, key, value) VALUES ('itest-alpha', %s, '1')", (key,))
                await conn.execute("SELECT set_config('app.tenant', 'infra-admin', false)")
                await conn.execute("SELECT set_config('app.read_all', 'on', false)")
                seen = await (await conn.execute("SELECT count(*) FROM kv WHERE key = %s", (key,))).fetchone()
                deleted = await conn.execute("DELETE FROM kv WHERE key = %s", (key,))
                await conn.execute("UPDATE kv SET value = '2' WHERE key = %s", (key,))
                await conn.execute("SELECT set_config('app.read_all', 'off', false)")
                await conn.execute("SELECT set_config('app.tenant', 'itest-alpha', false)")
                value = await (await conn.execute("SELECT value FROM kv WHERE key = %s", (key,))).fetchone()
                await conn.execute("DELETE FROM kv WHERE key = %s", (key,))
            assert seen is not None and value is not None
            return int(seen[0]) + deleted.rowcount * 10, int(value[0])

        seen_plus_deleted, value = asyncio.run(scenario())
        self.assertEqual(seen_plus_deleted, 1)
        self.assertEqual(value, 1)


class AdminLogsLiveTest(unittest.TestCase):
    def test_admin_filter_by_app_returns_only_that_apps_lines(self) -> None:
        marker = f"itest-admin-{uuid.uuid4().hex}"
        http("POST", f"{SETTINGS.logs_url}/v1/logs", {"level": "info", "message": marker}, bearer("itest-alpha"))
        admin = bearer("infra-admin")

        def messages(query: str) -> list[tuple[str, str]]:
            _, body = http("GET", f"{SETTINGS.logs_url}/v1/admin/logs?since_minutes=5&limit=1000{query}", headers=admin)
            return [(i["tenant"], i["message"]) for i in body["items"]]

        for _ in range(10):
            if ("itest-alpha", marker) in messages(""):
                break
            time.sleep(0.5)
        self.assertIn(("itest-alpha", marker), messages(""))
        self.assertIn(("itest-alpha", marker), messages("&tenant=itest-alpha"))
        self.assertNotIn(marker, [m for _, m in messages("&tenant=itest-beta")])
        self.assertEqual(http("GET", f"{SETTINGS.logs_url}/v1/admin/logs?tenant=nobody", headers=admin)[0], 400)


class AdminEvalsLiveTest(unittest.TestCase):
    def test_read_all_can_narrow_runs_to_one_app_and_tenants_cannot_widen(self) -> None:
        body = {"suite": "itest-admin", "cases": [{"output": "ok", "checks": [{"type": "contains", "value": "ok"}]}]}
        http("POST", f"{SETTINGS.evals_url}/v1/evals/runs", body, bearer("itest-alpha"))
        http("POST", f"{SETTINGS.evals_url}/v1/evals/runs", body, bearer("itest-beta"))
        url = f"{SETTINGS.evals_url}/v1/evals/runs?suite=itest-admin&limit=500"
        _, only_beta = http("GET", url + "&tenant=itest-beta", headers=bearer("infra-admin"))
        self.assertEqual({r["tenant"] for r in only_beta["items"]}, {"itest-beta"})
        _, alpha_asks_beta = http("GET", url + "&tenant=itest-beta", headers=bearer("itest-alpha"))
        self.assertEqual(alpha_asks_beta["items"], [])


if __name__ == "__main__":
    unittest.main()
