import asyncio
import unittest
import uuid

import psycopg

from tests.live import SETTINGS, bearer, http

STATE = SETTINGS.state_url


class StateIsolationLiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.key = f"itest/{uuid.uuid4().hex}"
        self.alpha = bearer("itest-alpha")
        self.beta = bearer("itest-beta")

    def tearDown(self) -> None:
        http("DELETE", f"{STATE}/v1/state/{self.key}", headers=self.alpha)
        http("DELETE", f"{STATE}/v1/state/{self.key}", headers=self.beta)

    def test_a_tenant_cannot_read_another_tenants_key(self) -> None:
        status, _ = http("PUT", f"{STATE}/v1/state/{self.key}", {"value": {"secret": "alpha"}}, self.alpha)
        self.assertEqual(status, 200)
        status, body = http("GET", f"{STATE}/v1/state/{self.key}", headers=self.beta)
        self.assertEqual(status, 404)
        status, body = http("GET", f"{STATE}/v1/state", headers=self.beta)
        self.assertNotIn(self.key, [i["key"] for i in body["items"]])

    def test_same_key_in_two_tenants_holds_two_independent_values(self) -> None:
        http("PUT", f"{STATE}/v1/state/{self.key}", {"value": "alpha"}, self.alpha)
        http("PUT", f"{STATE}/v1/state/{self.key}", {"value": "beta"}, self.beta)
        self.assertEqual(http("GET", f"{STATE}/v1/state/{self.key}", headers=self.alpha)[1]["value"], "alpha")
        self.assertEqual(http("GET", f"{STATE}/v1/state/{self.key}", headers=self.beta)[1]["value"], "beta")

    def test_a_tenant_cannot_delete_another_tenants_key(self) -> None:
        http("PUT", f"{STATE}/v1/state/{self.key}", {"value": 1}, self.alpha)
        status, _ = http("DELETE", f"{STATE}/v1/state/{self.key}", headers=self.beta)
        self.assertEqual(status, 404)
        self.assertEqual(http("GET", f"{STATE}/v1/state/{self.key}", headers=self.alpha)[0], 200)

    def test_list_prefix_wildcards_are_literal(self) -> None:
        http("PUT", f"{STATE}/v1/state/{self.key}", {"value": 1}, self.alpha)
        _, body = http("GET", f"{STATE}/v1/state?prefix=itest%2F%25", headers=self.alpha)
        self.assertEqual(body["items"], [])

    def test_missing_token_is_401_and_read_only_token_cannot_write(self) -> None:
        self.assertEqual(http("GET", f"{STATE}/v1/state")[0], 401)
        status, body = http("PUT", f"{STATE}/v1/state/{self.key}", {"value": 1}, bearer("itest-readonly"))
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "insufficient_scope")
        self.assertEqual(http("GET", f"{STATE}/v1/state", headers=bearer("itest-alpha", "logs:write"))[0], 403)

    def test_tampered_token_is_rejected(self) -> None:
        tampered = {"authorization": self.alpha["authorization"][:-2] + "xx"}
        self.assertEqual(http("GET", f"{STATE}/v1/state", headers=tampered)[0], 401)


class RowLevelSecurityLiveTest(unittest.TestCase):
    def test_database_itself_hides_rows_and_refuses_cross_tenant_writes(self) -> None:
        async def scenario() -> tuple[int, int, bool]:
            key = f"itest/{uuid.uuid4().hex}"
            async with await psycopg.AsyncConnection.connect(SETTINGS.database_url, autocommit=True) as conn:
                await conn.execute("SELECT set_config('app.tenant', 'itest-alpha', false)")
                await conn.execute("INSERT INTO kv (tenant, key, value) VALUES ('itest-alpha', %s, '1')", (key,))
                await conn.execute("SELECT set_config('app.tenant', '', false)")
                no_tenant = await (await conn.execute("SELECT count(*) FROM kv")).fetchone()
                await conn.execute("SELECT set_config('app.tenant', 'itest-beta', false)")
                as_beta = await (await conn.execute("SELECT count(*) FROM kv WHERE key = %s", (key,))).fetchone()
                try:
                    await conn.execute("INSERT INTO kv (tenant, key, value) VALUES ('itest-alpha', %s, '2')", (key + "x",))
                    refused = False
                except psycopg.errors.InsufficientPrivilege:
                    refused = True
                await conn.execute("SELECT set_config('app.tenant', 'itest-alpha', false)")
                await conn.execute("DELETE FROM kv WHERE key = %s", (key,))
            assert no_tenant is not None and as_beta is not None
            return int(no_tenant[0]), int(as_beta[0]), refused

        no_tenant, as_beta, refused = asyncio.run(scenario())
        self.assertEqual(no_tenant, 0)
        self.assertEqual(as_beta, 0)
        self.assertTrue(refused)


if __name__ == "__main__":
    unittest.main()
