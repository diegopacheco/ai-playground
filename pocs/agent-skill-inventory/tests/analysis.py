#!/usr/bin/env python3
import json
import sys


def five(prefix):
    return [f"{prefix} number {i}" for i in range(1, 6)]


def cons(prefix):
    return [{"title": f"{prefix} problem {i}",
             "why": "It costs real time for the person maintaining this code.",
             "fix": "Change the named function and add a test for it."} for i in range(1, 6)]


def build():
    return {
        "project": {"name": "fixture", "summary": "A fixture repository used by the test suite.",
                    "codebaseUrl": "", "githubUrl": None},
        "architecture": {
            "summary": "Handlers read from PostgreSQL.",
            "nodes": [{"id": "api", "label": "API", "kind": "module", "layer": 0},
                      {"id": "db", "label": "PostgreSQL", "kind": "store", "layer": 1}],
            "edges": [{"from": "api", "to": "db", "label": "queries",
                       "evidence": "src/handlers/orders.rs sqlx::query_as"}],
        },
        "modules": [{
            "id": "handlers", "name": "Handlers", "icon": "🛣️",
            "description": "Reads orders from the database.\nJoins each order to its customer.\nReturns the result to the caller.",
            "paths": ["src/handlers"],
            "files": [{"path": "src/handlers/orders.rs", "role": "Lists orders"}],
            "pros": five("Handler pro"), "cons": cons("Handler"),
        }],
        "schema": {
            "present": True, "engine": "postgres", "tableCount": 2,
            "sizeNote": "n/a - not reachable from a static scan",
            "tables": [
                {"name": "orders", "purpose": "Order rows", "columns": 4, "rows": "n/a",
                 "size": "n/a", "source": "migrations/0001_create_orders.sql"},
                {"name": "customers", "purpose": "Customer rows", "columns": 2, "rows": "n/a",
                 "size": "n/a", "source": "migrations/0002_create_customers.sql"},
            ],
            "worstQueries": [{"query": "SELECT name FROM customers WHERE id = $1",
                              "path": "src/handlers/orders.rs", "line": 6,
                              "why": "It runs once per order row.",
                              "fix": "Join customers into the orders query."}],
            "pros": five("Schema pro"), "cons": cons("Schema"),
        },
        "committers": [{"name": "Fixture Author", "email": "fixture@example.com", "login": None,
                        "commits": 1, "firstCommit": "2026-01-01", "lastCommit": "2026-01-01",
                        "areas": ["src/handlers"], "worksOn": "Wrote the fixture.",
                        "profileUrl": "mailto:fixture@example.com"}],
        "techDebt": [{
            "id": "n-plus-one", "module": "handlers", "title": "One query per order",
            "severity": "high",
            "whyBad": "Each order triggers another query for its customer name.",
            "howToFix": "Join customers into the orders query and drop the loop.",
            "antiPatterns": [{"name": "N+1 query", "why": "One query per row multiplies round trips."}],
            "examples": [{"path": "src/handlers/orders.rs", "line": 5, "snippet": "for row in &rows {"}],
        }],
        "observability": {
            "logsScore": 30, "logsVerdict": "Two log lines in the whole crate.",
            "logsFindings": [{"title": "Almost no logs", "detail": "Only two calls.", "path": "src/main.rs"}],
            "dashboards": [], "alerts": [],
            "defensive": {"score": 40, "verdict": "Validation rules exist but are not called.",
                          "findings": [{"title": "validate() unused", "detail": "Declared only.",
                                        "path": "src/models/order.rs"}]},
        },
        "tests": {
            "summary": "Three suites exist.",
            "types": [{"type": "integration", "files": 1, "cases": 2, "verdict": "Two cases."},
                      {"type": "e2e", "files": 1, "cases": 1, "verdict": "One spec."},
                      {"type": "stress", "files": 1, "cases": 0, "verdict": "One k6 script."}],
            "coverage": [{"area": "Handlers with a test", "covered": 1, "total": 1}],
            "issues": [{"title": "No unit tests", "why": "Only integration coverage exists.",
                        "fix": "Add unit tests for the model rules.",
                        "path": "src/models/order.rs", "line": 1}],
        },
        "verification": {"passes": 3, "notes": ["fixed: nothing"]},
    }


BREAK = {
    "bad-path": lambda a: a["modules"][0]["files"].append({"path": "src/nope.rs", "role": "missing"}),
    "four-pros": lambda a: a["modules"][0]["pros"].pop(),
    "unknown-edge": lambda a: a["architecture"]["edges"].append(
        {"from": "api", "to": "ghost", "label": "x", "evidence": "none"}),
    "wrong-commits": lambda a: a["committers"][0].update({"commits": 99}),
    "one-pass": lambda a: a["verification"].update({"passes": 1}),
    "no-evidence": lambda a: a["architecture"]["edges"][0].update({"evidence": ""}),
    "eleven-debt": lambda a: a["techDebt"].extend(
        [dict(a["techDebt"][0], id=f"extra-{i}") for i in range(10)]),
    "wrong-test-count": lambda a: a["tests"]["types"][0].update({"files": 7}),
    "wrong-case-count": lambda a: a["tests"]["types"][0].update({"cases": 99}),
    "bad-severity": lambda a: a["techDebt"][0].update({"severity": "critical"}),
    "no-schema": lambda a: a.update({"schema": {"present": False}}),
}


def main():
    out = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "valid"
    analysis = build()
    if mode != "valid":
        BREAK[mode](analysis)
    json.dump(analysis, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
