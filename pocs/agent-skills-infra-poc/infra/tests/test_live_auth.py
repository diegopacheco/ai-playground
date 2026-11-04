import unittest

from tests.live import SETTINGS, basic, http, token

AUTH = SETTINGS.auth_url


class AuthLiveTest(unittest.TestCase):
    def test_metadata_advertises_only_client_credentials(self) -> None:
        status, body = http("GET", f"{AUTH}/.well-known/oauth-authorization-server")
        self.assertEqual(status, 200)
        self.assertEqual(body["grant_types_supported"], ["client_credentials"])
        self.assertEqual(body["token_endpoint"], f"{AUTH}/oauth/token")

    def test_token_carries_the_client_as_tenant(self) -> None:
        status, body = http("POST", f"{AUTH}/oauth/introspect", headers=basic("itest-alpha"), form={"token": token("itest-alpha")})
        self.assertEqual(status, 200)
        self.assertTrue(body["active"])
        self.assertEqual(body["sub"], "itest-alpha")

    def test_wrong_secret_is_invalid_client(self) -> None:
        status, body = http("POST", f"{AUTH}/oauth/token", headers=basic("itest-alpha", "wrong"), form={"grant_type": "client_credentials"})
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "invalid_client")

    def test_unknown_grant_type_is_rejected(self) -> None:
        status, body = http("POST", f"{AUTH}/oauth/token", headers=basic("itest-alpha"), form={"grant_type": "password"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "unsupported_grant_type")

    def test_a_client_cannot_ask_for_a_scope_it_was_not_granted(self) -> None:
        status, body = http(
            "POST", f"{AUTH}/oauth/token", headers=basic("itest-alpha"), form={"grant_type": "client_credentials", "scope": "evals:read:all"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_scope")

    def test_down_scoped_token_only_has_the_requested_scope(self) -> None:
        status, body = http(
            "POST", f"{AUTH}/oauth/token", headers=basic("itest-alpha"), form={"grant_type": "client_credentials", "scope": "state:read"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["scope"], "state:read")

    def test_garbage_token_introspects_as_inactive(self) -> None:
        status, body = http("POST", f"{AUTH}/oauth/introspect", headers=basic("itest-alpha"), form={"token": "a.b.c"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"active": False})


if __name__ == "__main__":
    unittest.main()
