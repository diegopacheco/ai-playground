import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from common.db import tenant_tx
from common.http import HttpError

KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]{1,200}$")
MAX_VALUE_BYTES = 65_536


@dataclass(frozen=True)
class Entry:
    key: str
    value: Any
    updated_at: datetime

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "updated_at": self.updated_at.isoformat()}


def validate_key(key: str) -> str:
    if not KEY_PATTERN.match(key):
        raise HttpError(400, "invalid_key", "key must match [A-Za-z0-9._:/-]{1,200}")
    return key


def like_prefix(prefix: str) -> str:
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def validate_value(value: Any) -> str:
    encoded = json.dumps(value)
    if len(encoded.encode()) > MAX_VALUE_BYTES:
        raise HttpError(413, "value_too_large", f"value must be at most {MAX_VALUE_BYTES} bytes")
    return encoded


@dataclass(frozen=True)
class StateStore:
    database_url: str

    async def put(self, tenant: str, key: str, value: Any) -> Entry:
        encoded = validate_value(value)
        async with tenant_tx(self.database_url, tenant) as cur:
            await cur.execute(
                "INSERT INTO kv (tenant, key, value) VALUES (%s, %s, %s::jsonb) "
                "ON CONFLICT (tenant, key) DO UPDATE SET value = EXCLUDED.value, updated_at = now() "
                "RETURNING key, value, updated_at",
                (tenant, validate_key(key), encoded),
            )
            row = await cur.fetchone()
        assert row is not None
        return Entry(row[0], row[1], row[2])

    async def get(self, tenant: str, key: str) -> Entry | None:
        async with tenant_tx(self.database_url, tenant) as cur:
            await cur.execute("SELECT key, value, updated_at FROM kv WHERE key = %s", (validate_key(key),))
            row = await cur.fetchone()
        return Entry(row[0], row[1], row[2]) if row else None

    async def list_prefix(self, tenant: str, prefix: str) -> list[Entry]:
        async with tenant_tx(self.database_url, tenant) as cur:
            await cur.execute(
                "SELECT key, value, updated_at FROM kv WHERE key LIKE %s ORDER BY key LIMIT 1000",
                (like_prefix(prefix),),
            )
            rows = await cur.fetchall()
        return [Entry(r[0], r[1], r[2]) for r in rows]

    async def list_all(self, admin: str, tenant: str | None, prefix: str) -> list[dict[str, Any]]:
        sql = "SELECT tenant, key, value, updated_at FROM kv WHERE key LIKE %s"
        params: list[Any] = [like_prefix(prefix)]
        if tenant:
            sql += " AND tenant = %s"
            params.append(tenant)
        sql += " ORDER BY updated_at DESC LIMIT 500"
        async with tenant_tx(self.database_url, admin, read_all=True) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [{"tenant": r[0], **Entry(r[1], r[2], r[3]).to_json()} for r in rows]

    async def summary(self, admin: str) -> list[dict[str, Any]]:
        async with tenant_tx(self.database_url, admin, read_all=True) as cur:
            await cur.execute(
                "SELECT tenant, count(*), max(updated_at), "
                "(array_agg(key ORDER BY updated_at DESC))[1:5] FROM kv GROUP BY tenant ORDER BY tenant"
            )
            rows = await cur.fetchall()
        return [
            {"tenant": r[0], "keys": int(r[1]), "last_updated": r[2].isoformat(), "recent_keys": list(r[3])} for r in rows
        ]

    async def delete(self, tenant: str, key: str) -> bool:
        async with tenant_tx(self.database_url, tenant) as cur:
            await cur.execute("DELETE FROM kv WHERE key = %s", (validate_key(key),))
            return cur.rowcount > 0
