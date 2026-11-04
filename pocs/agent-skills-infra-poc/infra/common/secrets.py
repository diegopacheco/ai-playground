import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

ITERATIONS = 200_000


@dataclass(frozen=True)
class Client:
    client_id: str
    secret_hash: str
    salt: str
    scopes: frozenset[str]


def hash_secret(secret: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", secret.encode(), bytes.fromhex(salt), ITERATIONS).hex()


def new_client(client_id: str, scopes: list[str]) -> tuple[Client, str]:
    secret = secrets.token_urlsafe(32)
    salt = secrets.token_hex(16)
    return Client(client_id, hash_secret(secret, salt), salt, frozenset(scopes)), secret


def verify_secret(client: Client, secret: str) -> bool:
    return hmac.compare_digest(client.secret_hash, hash_secret(secret, client.salt))


def load_clients(secrets_dir: Path) -> dict[str, Client]:
    raw = json.loads((secrets_dir / "clients.json").read_text())
    return {
        item["client_id"]: Client(item["client_id"], item["secret_hash"], item["salt"], frozenset(item["scopes"]))
        for item in raw
    }


def save_clients(secrets_dir: Path, clients: list[Client]) -> None:
    data = [
        {"client_id": c.client_id, "secret_hash": c.secret_hash, "salt": c.salt, "scopes": sorted(c.scopes)}
        for c in clients
    ]
    (secrets_dir / "clients.json").write_text(json.dumps(data, indent=2))


def load_signing_key(secrets_dir: Path) -> bytes:
    return bytes.fromhex((secrets_dir / "jwt.key").read_text().strip())
