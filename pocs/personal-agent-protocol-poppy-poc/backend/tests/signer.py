import json
import secrets
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from poppy_company.b64 import b64url_encode
from poppy_company.jose import thumbprint
from poppy_company.json_types import JsonObject

BASE = "http://localhost:8741"


def public_jwk(key: ec.EllipticCurvePrivateKey) -> JsonObject:
    numbers = key.public_key().public_numbers()
    return {"kty": "EC", "crv": "P-256", "x": b64url_encode(numbers.x.to_bytes(32)), "y": b64url_encode(numbers.y.to_bytes(32))}


def sign(key: ec.EllipticCurvePrivateKey, header: JsonObject, payload: JsonObject) -> str:
    signing_input = f"{b64url_encode(json.dumps(header).encode())}.{b64url_encode(json.dumps(payload).encode())}"
    r, s = decode_dss_signature(key.sign(signing_input.encode(), ec.ECDSA(hashes.SHA256())))
    return f"{signing_input}.{b64url_encode(r.to_bytes(32) + s.to_bytes(32))}"


@dataclass
class SigningAgent:
    name: str = "atlas"
    sub: str = "usr_alpha"
    now: int = 1_790_900_000
    root: ec.EllipticCurvePrivateKey = field(default_factory=lambda: ec.generate_private_key(ec.SECP256R1()))
    delegated: ec.EllipticCurvePrivateKey = field(default_factory=lambda: ec.generate_private_key(ec.SECP256R1()))

    @property
    def client_id(self) -> str:
        return f"{BASE}/agent-host/{self.name}/agent.json"

    @property
    def root_kid(self) -> str:
        return thumbprint(public_jwk(self.root))

    def metadata(self) -> JsonObject:
        return {"client_id": self.client_id, "client_name": "Atlas", "jwks": {"keys": [public_jwk(self.root) | {"kid": self.root_kid}]}}

    def delegation(self, exp_offset: int = 3600, signer: ec.EllipticCurvePrivateKey | None = None) -> str:
        header: JsonObject = {"alg": "ES256", "typ": "poppy-delegation+jwt", "kid": self.root_kid}
        payload: JsonObject = {"iss": self.client_id, "cnf": {"jwk": public_jwk(self.delegated)}, "iat": self.now, "exp": self.now + exp_offset, "jti": secrets.token_urlsafe(8)}
        return sign(signer or self.root, header, payload)

    def session(self, method: str, url: str, delegation: str | None = None, **overrides: str | int) -> str:
        header: JsonObject = {"typ": "poppy-session+jwt", "alg": "ES256", "kid": thumbprint(public_jwk(self.delegated)), "poppy_dlg": delegation or self.delegation()}
        payload: JsonObject = {"iss": self.client_id, "aud": BASE, "sid": "sid_one", "sub": self.sub, "htm": method, "htu": url, "iat": self.now, "jti": secrets.token_urlsafe(12)}
        payload.update(overrides)
        return sign(self.delegated, header, payload)
