import unittest

from common.jwt import Claims, TokenError, sign, verify

KEY = b"k" * 32
ISSUER = "http://auth"


def claims(exp: int = 2000, scopes: frozenset[str] = frozenset({"state:read"})) -> Claims:
    return Claims("tenant-a", ISSUER, scopes, 1000, exp, "j1")


class JwtTest(unittest.TestCase):
    def test_round_trip_keeps_tenant_and_scopes_because_services_trust_only_the_token(self) -> None:
        token = sign(claims(scopes=frozenset({"state:read", "state:write"})), KEY)
        verified = verify(token, KEY, ISSUER, 1500)
        self.assertEqual(verified.sub, "tenant-a")
        self.assertEqual(verified.scopes, {"state:read", "state:write"})

    def test_changing_the_tenant_in_the_payload_breaks_the_signature(self) -> None:
        import base64
        import json

        header, payload, signature = sign(claims(), KEY).split(".")
        body = json.loads(base64.urlsafe_b64decode(payload + "=="))
        body["sub"] = "tenant-b"
        forged = base64.urlsafe_b64encode(json.dumps(body).encode()).rstrip(b"=").decode()
        with self.assertRaisesRegex(TokenError, "bad signature"):
            verify(f"{header}.{forged}.{signature}", KEY, ISSUER, 1500)

    def test_token_signed_with_another_key_is_rejected(self) -> None:
        with self.assertRaises(TokenError):
            verify(sign(claims(), b"x" * 32), KEY, ISSUER, 1500)

    def test_expired_token_is_rejected_at_the_exact_expiry_second(self) -> None:
        token = sign(claims(exp=2000), KEY)
        verify(token, KEY, ISSUER, 1999)
        with self.assertRaisesRegex(TokenError, "expired"):
            verify(token, KEY, ISSUER, 2000)

    def test_token_from_another_issuer_is_rejected(self) -> None:
        with self.assertRaisesRegex(TokenError, "issuer"):
            verify(sign(claims(), KEY), KEY, "http://other", 1500)

    def test_malformed_tokens_raise_token_error_not_crash(self) -> None:
        for bad in ["", "a.b", "a.b.c.d", "...."]:
            with self.assertRaises(TokenError):
                verify(bad, KEY, ISSUER, 1500)


if __name__ == "__main__":
    unittest.main()
