import hashlib
import json
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from poppy_company.b64 import b64url_decode, b64url_encode
from poppy_company.errors import invalid_session
from poppy_company.json_types import JsonObject


@dataclass(frozen=True)
class CompactJws:
    header: JsonObject
    payload: JsonObject
    signing_input: bytes
    signature: bytes


def parse_compact(token: str) -> CompactJws:
    parts = token.split(".")
    if len(parts) != 3:
        raise invalid_session("JWT must have three parts")
    try:
        header = json.loads(b64url_decode(parts[0]))
        payload = json.loads(b64url_decode(parts[1]))
        signature = b64url_decode(parts[2])
    except ValueError as error:
        raise invalid_session("JWT is not valid base64url JSON") from error
    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise invalid_session("JWT header and payload must be objects")
    return CompactJws(header, payload, f"{parts[0]}.{parts[1]}".encode(), signature)


def ec_jwk_members(jwk: JsonObject) -> tuple[str, str]:
    x, y = jwk.get("x"), jwk.get("y")
    if jwk.get("kty") != "EC" or jwk.get("crv") != "P-256" or not isinstance(x, str) or not isinstance(y, str):
        raise invalid_session("key must be an EC P-256 JWK")
    return x, y


def thumbprint(jwk: JsonObject) -> str:
    x, y = ec_jwk_members(jwk)
    canonical = json.dumps({"crv": "P-256", "kty": "EC", "x": x, "y": y}, separators=(",", ":"))
    return b64url_encode(hashlib.sha256(canonical.encode()).digest())


def public_key(jwk: JsonObject) -> ec.EllipticCurvePublicKey:
    x, y = ec_jwk_members(jwk)
    numbers = ec.EllipticCurvePublicNumbers(int.from_bytes(b64url_decode(x)), int.from_bytes(b64url_decode(y)), ec.SECP256R1())
    try:
        return numbers.public_key()
    except ValueError as error:
        raise invalid_session("key is not on the P-256 curve") from error


def verify_es256(jwk: JsonObject, jws: CompactJws) -> None:
    if len(jws.signature) != 64:
        raise invalid_session("ES256 signature must be 64 bytes")
    der = encode_dss_signature(int.from_bytes(jws.signature[:32]), int.from_bytes(jws.signature[32:]))
    try:
        public_key(jwk).verify(der, jws.signing_input, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as error:
        raise invalid_session("signature does not verify") from error
