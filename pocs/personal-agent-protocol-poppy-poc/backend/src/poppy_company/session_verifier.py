from collections.abc import Callable
from dataclasses import dataclass

from poppy_company.agent_directory import AgentDirectory
from poppy_company.errors import invalid_session
from poppy_company.jose import parse_compact, thumbprint, verify_es256
from poppy_company.json_types import JsonObject, JsonValue

SESSION_TYP = "poppy-session+jwt"
DELEGATION_TYP = "poppy-delegation+jwt"


@dataclass(frozen=True)
class Session:
    client_id: str
    sid: str
    sub: str


def text_claim(claims: JsonObject, name: str) -> str:
    value = claims.get(name)
    if not isinstance(value, str) or not value:
        raise invalid_session(f"{name} is required")
    return value


def int_claim(claims: JsonObject, name: str) -> int:
    value = claims.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_session(f"{name} must be a number")
    return value


def object_claim(value: JsonValue, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise invalid_session(f"{name} must be an object")
    return value


class SessionVerifier:
    def __init__(self, directory: AgentDirectory, audience: str, clock: Callable[[], float], window: int = 120) -> None:
        self._directory = directory
        self._audience = audience
        self._clock = clock
        self._window = window
        self._seen: dict[str, float] = {}

    def verify(self, token: str, method: str, url: str) -> Session:
        session = parse_compact(token)
        if session.header.get("typ") != SESSION_TYP or session.header.get("alg") != "ES256":
            raise invalid_session(f"Session JWT must be typ {SESSION_TYP} and alg ES256")
        delegated_key = self._verify_delegation(text_claim(session.header, "poppy_dlg"), text_claim(session.payload, "iss"))
        if text_claim(session.header, "kid") != thumbprint(delegated_key):
            raise invalid_session("kid is not the thumbprint of the delegated key")
        verify_es256(delegated_key, session)
        return self._check_claims(session.payload, method, url)

    def _verify_delegation(self, token: str, client_id: str) -> JsonObject:
        delegation = parse_compact(token)
        if delegation.header.get("typ") != DELEGATION_TYP or delegation.header.get("alg") != "ES256":
            raise invalid_session(f"delegation must be typ {DELEGATION_TYP} and alg ES256")
        if text_claim(delegation.payload, "iss") != client_id:
            raise invalid_session("delegation iss does not match the Session iss")
        verify_es256(self._directory.key(client_id, text_claim(delegation.header, "kid")), delegation)
        now = self._clock()
        if int_claim(delegation.payload, "exp") <= now or int_claim(delegation.payload, "iat") > now + self._window:
            raise invalid_session("delegation is expired or not yet valid")
        return object_claim(object_claim(delegation.payload.get("cnf"), "cnf").get("jwk"), "cnf.jwk")

    def _check_claims(self, claims: JsonObject, method: str, url: str) -> Session:
        if text_claim(claims, "aud") != self._audience:
            raise invalid_session(f"aud must be {self._audience}")
        if text_claim(claims, "htm") != method or text_claim(claims, "htu") != url:
            raise invalid_session("htm and htu must match this request")
        now = self._clock()
        if abs(now - int_claim(claims, "iat")) > self._window:
            raise invalid_session("iat is outside the allowed window")
        jti = text_claim(claims, "jti")
        self._seen = {key: expiry for key, expiry in self._seen.items() if expiry > now}
        if jti in self._seen:
            raise invalid_session("jti was already used")
        self._seen[jti] = now + self._window * 2
        return Session(text_claim(claims, "iss"), text_claim(claims, "sid"), text_claim(claims, "sub"))
