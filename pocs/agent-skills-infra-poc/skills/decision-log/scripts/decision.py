import argparse
import json
import re
import sys
from datetime import UTC, datetime
from typing import Any

from agentinfra import InfraError, call, log

CLIENT = "decision-log"
STATUSES = ("proposed", "accepted", "superseded", "rejected")


def slug(title: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]
    if not value:
        raise ValueError("title must contain letters or digits")
    return value


def add(title: str, decision: str, why: str, status: str) -> dict[str, Any]:
    record = {
        "id": slug(title),
        "title": title,
        "decision": decision,
        "why": why,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    call(CLIENT, "state", "put", f"adr/{record['id']}", json.dumps(record))
    log(CLIENT, "info", "decision recorded", id=record["id"], status=status)
    return record


def show(decision_id: str) -> dict[str, Any]:
    return call(CLIENT, "state", "get", f"adr/{decision_id}")["value"]


def list_all(status: str | None) -> dict[str, Any]:
    items = [item["value"] for item in call(CLIENT, "state", "list", "--prefix", "adr/")["items"]]
    if status:
        items = [item for item in items if item["status"] == status]
    return {"count": len(items), "items": [{"id": i["id"], "status": i["status"], "title": i["title"]} for i in items]}


def supersede(decision_id: str, by: str) -> dict[str, Any]:
    replacement = show(by)
    record = show(decision_id)
    record["status"] = "superseded"
    record["superseded_by"] = replacement["id"]
    call(CLIENT, "state", "put", f"adr/{decision_id}", json.dumps(record))
    log(CLIENT, "warn", "decision superseded", id=decision_id, by=by)
    return record


def remove(decision_id: str) -> dict[str, Any]:
    call(CLIENT, "state", "delete", f"adr/{decision_id}")
    log(CLIENT, "warn", "decision removed", id=decision_id)
    return {"removed": decision_id}


def main() -> int:
    parser = argparse.ArgumentParser(prog="decision", description="Persistent architecture decision log")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add")
    a.add_argument("--title", required=True)
    a.add_argument("--decision", required=True)
    a.add_argument("--why", required=True)
    a.add_argument("--status", choices=STATUSES, default="accepted")
    sub.add_parser("show").add_argument("id")
    sub.add_parser("list").add_argument("--status", choices=STATUSES)
    s = sub.add_parser("supersede")
    s.add_argument("id")
    s.add_argument("--by", required=True)
    sub.add_parser("remove").add_argument("id")
    args = parser.parse_args()
    try:
        match args.command:
            case "add":
                result = add(args.title, args.decision, args.why, args.status)
            case "show":
                result = show(args.id)
            case "list":
                result = list_all(args.status)
            case "supersede":
                result = supersede(args.id, args.by)
            case _:
                result = remove(args.id)
    except (InfraError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
