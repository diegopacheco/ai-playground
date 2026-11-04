import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from common.http import HttpError

LEVELS = ("debug", "info", "warn", "error")


@dataclass(frozen=True)
class LogLine:
    timestamp_ns: int
    level: str
    message: str
    fields: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"timestamp_ns": str(self.timestamp_ns), "level": self.level, "message": self.message, "fields": self.fields}


def validate(level: str, message: str, fields: Any) -> None:
    if level not in LEVELS:
        raise HttpError(400, "invalid_level", f"level must be one of {', '.join(LEVELS)}")
    if not message or len(message) > 4000:
        raise HttpError(400, "invalid_message", "message must have 1 to 4000 characters")
    if not isinstance(fields, dict):
        raise HttpError(400, "invalid_fields", "fields must be an object")


def push_body(line: LogLine) -> dict[str, Any]:
    payload = json.dumps({"message": line.message, "fields": line.fields})
    return {
        "streams": [
            {
                "stream": {"service_name": "agentinfra", "level": line.level},
                "values": [[str(line.timestamp_ns), payload]],
            }
        ]
    }


def selector(level: str | None) -> str:
    if level is None:
        return '{service_name="agentinfra"}'
    if level not in LEVELS:
        raise HttpError(400, "invalid_level", f"level must be one of {', '.join(LEVELS)}")
    return f'{{service_name="agentinfra", level="{level}"}}'


def parse_query(result: dict[str, Any]) -> list[LogLine]:
    lines: list[LogLine] = []
    for stream in result.get("data", {}).get("result", []):
        level = stream.get("stream", {}).get("level", "info")
        for ts, raw in stream.get("values", []):
            try:
                body = json.loads(raw)
            except ValueError:
                body = {"message": raw, "fields": {}}
            lines.append(LogLine(int(ts), level, str(body.get("message", "")), dict(body.get("fields", {}))))
    return sorted(lines, key=lambda line: line.timestamp_ns, reverse=True)


def parse_counts(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in result.get("data", {}).get("result", []):
        metric = item.get("metric", {})
        rows.append({"tenant": metric.get("__tenant_id__", ""), "level": metric.get("level", ""), "count": int(float(item["value"][1]))})
    return sorted(rows, key=lambda r: (r["tenant"], r["level"]))


def parse_recent(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stream in result.get("data", {}).get("result", []):
        tenant = stream.get("stream", {}).get("__tenant_id__", "")
        for line in parse_query({"data": {"result": [stream]}}):
            rows.append({"tenant": tenant, **line.to_json()})
    return sorted(rows, key=lambda r: int(r["timestamp_ns"]), reverse=True)


@dataclass(frozen=True)
class LokiClient:
    base_url: str

    def _call(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            with error:
                detail = error.read().decode()[:300]
            raise HttpError(502, "loki_error", detail) from error
        except urllib.error.URLError as error:
            raise HttpError(503, "loki_unavailable", str(error.reason)) from error
        return json.loads(raw) if raw else {}

    def _push(self, tenant: str, line: LogLine) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/loki/api/v1/push",
            data=json.dumps(push_body(line)).encode(),
            headers={"content-type": "application/json", "x-scope-orgid": tenant},
            method="POST",
        )
        self._call(request)

    def _query(self, tenant: str, level: str | None, since_minutes: int, limit: int) -> list[LogLine]:
        now = time.time_ns()
        params = urlencode(
            {
                "query": selector(level),
                "start": str(now - since_minutes * 60 * 1_000_000_000),
                "end": str(now),
                "limit": str(limit),
                "direction": "backward",
            }
        )
        request = urllib.request.Request(
            f"{self.base_url}/loki/api/v1/query_range?{params}", headers={"x-scope-orgid": tenant}
        )
        return parse_query(self._call(request))

    def _summary(self, tenants: list[str], since_minutes: int, limit: int) -> dict[str, Any]:
        header = {"x-scope-orgid": "|".join(tenants)}
        window = f"{since_minutes}m"
        counts_query = urlencode(
            {"query": f'sum by (__tenant_id__, level) (count_over_time({selector(None)}[{window}]))'}
        )
        counts = self._call(urllib.request.Request(f"{self.base_url}/loki/api/v1/query?{counts_query}", headers=header))
        now = time.time_ns()
        recent_query = urlencode(
            {
                "query": selector(None),
                "start": str(now - since_minutes * 60 * 1_000_000_000),
                "end": str(now),
                "limit": str(limit),
                "direction": "backward",
            }
        )
        recent = self._call(urllib.request.Request(f"{self.base_url}/loki/api/v1/query_range?{recent_query}", headers=header))
        return {"counts": parse_counts(counts), "recent": parse_recent(recent)}

    def _query_many(self, tenants: list[str], level: str | None, since_minutes: int, limit: int) -> list[dict[str, Any]]:
        now = time.time_ns()
        params = urlencode(
            {
                "query": selector(level),
                "start": str(now - since_minutes * 60 * 1_000_000_000),
                "end": str(now),
                "limit": str(limit),
                "direction": "backward",
            }
        )
        request = urllib.request.Request(
            f"{self.base_url}/loki/api/v1/query_range?{params}", headers={"x-scope-orgid": "|".join(tenants)}
        )
        result = self._call(request)
        if len(tenants) == 1:
            return [{"tenant": tenants[0], **line.to_json()} for line in parse_query(result)]
        return parse_recent(result)

    async def query_many(self, tenants: list[str], level: str | None, since_minutes: int, limit: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._query_many, tenants, level, since_minutes, limit)

    async def summary(self, tenants: list[str], since_minutes: int, limit: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._summary, tenants, since_minutes, limit)

    async def push(self, tenant: str, line: LogLine) -> None:
        await asyncio.to_thread(self._push, tenant, line)

    async def query(self, tenant: str, level: str | None, since_minutes: int, limit: int) -> list[LogLine]:
        return await asyncio.to_thread(self._query, tenant, level, since_minutes, limit)
