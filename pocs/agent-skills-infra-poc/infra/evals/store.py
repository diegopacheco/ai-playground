import json
from dataclasses import dataclass
from typing import Any

from common.db import tenant_tx
from common.http import HttpError

from evals.graders import CaseInput, grade


@dataclass(frozen=True)
class GradedCase:
    name: str
    input: str
    output: str
    latency_ms: int
    passed: bool
    checks: list[dict[str, Any]]


def grade_cases(raw_cases: Any) -> list[GradedCase]:
    if not isinstance(raw_cases, list) or not raw_cases:
        raise HttpError(400, "invalid_request", "cases must be a non empty list")
    graded: list[GradedCase] = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict) or not isinstance(raw.get("checks"), list) or not raw["checks"]:
            raise HttpError(400, "invalid_request", f"case {index} needs a non empty checks list")
        output = str(raw.get("output", ""))
        latency = int(raw.get("latency_ms", 0))
        results = grade(CaseInput(output, latency), raw["checks"])
        graded.append(
            GradedCase(
                name=str(raw.get("name", f"case-{index + 1}")),
                input=str(raw.get("input", "")),
                output=output,
                latency_ms=latency,
                passed=all(r.passed for r in results),
                checks=[r.to_json() for r in results],
            )
        )
    return graded


def summary(run_id: int, tenant: str, suite: str, cases: list[GradedCase], created_at: str) -> dict[str, Any]:
    passed = sum(1 for c in cases if c.passed)
    return {
        "id": run_id,
        "tenant": tenant,
        "suite": suite,
        "passed": passed,
        "failed": len(cases) - passed,
        "score": round(passed / len(cases), 4),
        "created_at": created_at,
    }


RUN_COLUMNS = "id, tenant, suite, passed, failed, score, created_at"


def _run_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "tenant": row[1],
        "suite": row[2],
        "passed": row[3],
        "failed": row[4],
        "score": float(row[5]),
        "created_at": row[6].isoformat(),
    }


@dataclass(frozen=True)
class EvalStore:
    database_url: str

    async def save(self, tenant: str, suite: str, cases: list[GradedCase]) -> dict[str, Any]:
        passed = sum(1 for c in cases if c.passed)
        async with tenant_tx(self.database_url, tenant) as cur:
            await cur.execute(
                "INSERT INTO eval_runs (tenant, suite, passed, failed, score) VALUES (%s, %s, %s, %s, %s) "
                "RETURNING id, created_at",
                (tenant, suite, passed, len(cases) - passed, round(passed / len(cases), 4)),
            )
            row = await cur.fetchone()
            assert row is not None
            run_id = int(row[0])
            for c in cases:
                await cur.execute(
                    "INSERT INTO eval_cases (run_id, tenant, name, input, output, latency_ms, passed, checks) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                    (run_id, tenant, c.name, c.input, c.output, c.latency_ms, c.passed, json.dumps(c.checks)),
                )
        return summary(run_id, tenant, suite, cases, row[1].isoformat())

    async def list_runs(
        self, tenant: str, read_all: bool, suite: str | None, limit: int, only_tenant: str | None = None
    ) -> list[dict[str, Any]]:
        sql = f"SELECT {RUN_COLUMNS} FROM eval_runs WHERE true"
        params: list[Any] = []
        if suite:
            sql += " AND suite = %s"
            params.append(suite)
        if only_tenant:
            sql += " AND tenant = %s"
            params.append(only_tenant)
        sql += " ORDER BY created_at DESC, id DESC LIMIT %s"
        params.append(limit)
        async with tenant_tx(self.database_url, tenant, read_all) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [_run_row(r) for r in rows]

    async def get_run(self, tenant: str, read_all: bool, run_id: int) -> dict[str, Any] | None:
        async with tenant_tx(self.database_url, tenant, read_all) as cur:
            await cur.execute(f"SELECT {RUN_COLUMNS} FROM eval_runs WHERE id = %s", (run_id,))
            row = await cur.fetchone()
            if row is None:
                return None
            await cur.execute(
                "SELECT name, input, output, latency_ms, passed, checks FROM eval_cases WHERE run_id = %s ORDER BY id",
                (run_id,),
            )
            cases = await cur.fetchall()
        run = _run_row(row)
        run["cases"] = [
            {"name": c[0], "input": c[1], "output": c[2], "latency_ms": c[3], "passed": c[4], "checks": c[5]}
            for c in cases
        ]
        return run
