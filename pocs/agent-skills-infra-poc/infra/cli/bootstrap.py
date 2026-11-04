import secrets
from pathlib import Path

from common.catalog import CATALOG
from common.secrets import Client, load_clients, new_client, save_clients


def _write_credentials(credentials_dir: Path, client_id: str, secret: str) -> None:
    path = credentials_dir / f"{client_id}.env"
    path.write_text(f"CLIENT_ID={client_id}\nCLIENT_SECRET={secret}\n")
    path.chmod(0o600)


def bootstrap(secrets_dir: Path, force: bool) -> dict[str, object]:
    credentials_dir = secrets_dir / "credentials"
    credentials_dir.mkdir(parents=True, exist_ok=True)
    secrets_dir.chmod(0o700)
    key_file = secrets_dir / "jwt.key"
    fresh = force or not (secrets_dir / "clients.json").exists() or not key_file.exists()
    if fresh:
        key_file.write_text(secrets.token_hex(32))
        key_file.chmod(0o600)
    existing = {} if fresh else load_clients(secrets_dir)
    registry: list[Client] = []
    created: list[str] = []
    for entry in CATALOG:
        current = existing.get(entry.client_id)
        if current is not None and current.scopes == frozenset(entry.scopes):
            registry.append(current)
            continue
        client, secret = new_client(entry.client_id, list(entry.scopes))
        registry.append(client)
        created.append(entry.client_id)
        _write_credentials(credentials_dir, entry.client_id, secret)
    save_clients(secrets_dir, registry)
    tokens = secrets_dir / "tokens"
    if tokens.exists():
        for client_id in created:
            (tokens / f"{client_id}.json").unlink(missing_ok=True)
    return {"created": sorted(created), "clients": [entry.client_id for entry in CATALOG]}
