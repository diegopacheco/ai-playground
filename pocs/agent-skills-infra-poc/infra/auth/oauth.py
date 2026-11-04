import base64
import secrets
import time
from dataclasses import dataclass
from urllib.parse import unquote

from common.http import HttpError, Request
from common.jwt import Claims, TokenError, sign, verify
from common.secrets import Client, verify_secret


@dataclass(frozen=True)
class TokenIssuer:
    clients: dict[str, Client]
    key: bytes
    issuer: str
    ttl_seconds: int

    def authenticate_client(self, request: Request, form: dict[str, str]) -> Client:
        client_id, client_secret = _client_credentials(request, form)
        client = self.clients.get(client_id)
        if client is None or not verify_secret(client, client_secret):
            raise HttpError(401, "invalid_client", "unknown client or bad secret")
        return client

    def issue(self, client: Client, requested: str) -> dict[str, str | int]:
        scopes = frozenset(requested.split()) if requested.strip() else client.scopes
        if not scopes <= client.scopes:
            raise HttpError(400, "invalid_scope", "scope not granted to this client")
        now = int(time.time())
        claims = Claims(client.client_id, self.issuer, scopes, now, now + self.ttl_seconds, secrets.token_hex(8))
        return {
            "access_token": sign(claims, self.key),
            "token_type": "Bearer",
            "expires_in": self.ttl_seconds,
            "scope": " ".join(sorted(scopes)),
        }

    def introspect(self, token: str) -> dict[str, str | int | bool]:
        try:
            claims = verify(token, self.key, self.issuer, int(time.time()))
        except TokenError:
            return {"active": False}
        if claims.sub not in self.clients:
            return {"active": False}
        return {
            "active": True,
            "sub": claims.sub,
            "client_id": claims.sub,
            "scope": " ".join(sorted(claims.scopes)),
            "iss": claims.iss,
            "exp": claims.exp,
            "iat": claims.iat,
            "token_type": "Bearer",
        }


def _client_credentials(request: Request, form: dict[str, str]) -> tuple[str, str]:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(header[6:].strip()).decode()
        except ValueError as error:
            raise HttpError(401, "invalid_client", "bad basic header") from error
        client_id, _, client_secret = decoded.partition(":")
        return unquote(client_id), unquote(client_secret)
    if "client_id" in form and "client_secret" in form:
        return form["client_id"], form["client_secret"]
    raise HttpError(401, "invalid_client", "client authentication required")
