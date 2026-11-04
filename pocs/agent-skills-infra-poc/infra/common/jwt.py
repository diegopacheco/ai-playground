import base64
import hashlib
import hmac
import json
from dataclasses import dataclass


class TokenError(Exception):
    pass


@dataclass(frozen=True)
class Claims:
    sub: str
    iss: str
    scopes: frozenset[str]
    iat: int
    exp: int
    jti: str


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(signing_input: str, key: bytes) -> str:
    return _b64encode(hmac.new(key, signing_input.encode(), hashlib.sha256).digest())


def sign(claims: Claims, key: bytes) -> str:
    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64encode(
        json.dumps(
            {
                "sub": claims.sub,
                "iss": claims.iss,
                "scope": " ".join(sorted(claims.scopes)),
                "iat": claims.iat,
                "exp": claims.exp,
                "jti": claims.jti,
            },
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{payload}"
    return f"{signing_input}.{_signature(signing_input, key)}"


def verify(token: str, key: bytes, issuer: str, now: int) -> Claims:
    parts = token.split(".")
    if len(parts) != 3:
        raise TokenError("malformed token")
    header_part, payload_part, signature_part = parts
    if not hmac.compare_digest(_signature(f"{header_part}.{payload_part}", key), signature_part):
        raise TokenError("bad signature")
    try:
        header = json.loads(_b64decode(header_part))
        payload = json.loads(_b64decode(payload_part))
    except ValueError as error:
        raise TokenError("malformed token") from error
    if header.get("alg") != "HS256":
        raise TokenError("unsupported alg")
    if payload.get("iss") != issuer:
        raise TokenError("wrong issuer")
    if int(payload.get("exp", 0)) <= now:
        raise TokenError("token expired")
    return Claims(
        sub=str(payload["sub"]),
        iss=str(payload["iss"]),
        scopes=frozenset(str(payload.get("scope", "")).split()),
        iat=int(payload["iat"]),
        exp=int(payload["exp"]),
        jti=str(payload["jti"]),
    )
