import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcpkit.agentinfra import call, get_or_none, log
from mcpkit.stdio_server import Tool, ToolError, serve

CLIENT = "feature-flags-mcp"
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_name(name: str) -> str:
    if not NAME_PATTERN.match(name):
        raise ToolError("flag name must match [a-z0-9][a-z0-9_-]{0,63}")
    return name


def bucket(flag: str, user_id: str) -> int:
    return int(hashlib.sha256(f"{flag}:{user_id}".encode()).hexdigest()[:8], 16) % 100


def load_flag(name: str) -> dict[str, Any]:
    flag = get_or_none(CLIENT, f"flag/{validate_name(name)}")
    if flag is None:
        raise ToolError(f"unknown flag {name}")
    return flag


def set_flag(args: dict[str, Any]) -> dict[str, Any]:
    name = validate_name(str(args["name"]))
    rollout = int(args.get("rollout_percent", 100))
    if not 0 <= rollout <= 100:
        raise ToolError("rollout_percent must be between 0 and 100")
    before = get_or_none(CLIENT, f"flag/{name}")
    flag = {
        "name": name,
        "enabled": bool(args["enabled"]),
        "rollout_percent": rollout,
        "description": str(args.get("description", before["description"] if before else "")),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    call(CLIENT, "state", "put", f"flag/{name}", json.dumps(flag))
    log(
        CLIENT,
        "warn",
        "flag changed",
        flag=name,
        before={k: before[k] for k in ("enabled", "rollout_percent")} if before else None,
        after={"enabled": flag["enabled"], "rollout_percent": rollout},
    )
    return flag


def get_flag(args: dict[str, Any]) -> dict[str, Any]:
    return load_flag(str(args["name"]))


def list_flags(_: dict[str, Any]) -> dict[str, Any]:
    items = [item["value"] for item in call(CLIENT, "state", "list", "--prefix", "flag/")["items"]]
    return {"count": len(items), "items": items}


def evaluate_flag(args: dict[str, Any]) -> dict[str, Any]:
    flag = load_flag(str(args["name"]))
    user_id = str(args["user_id"])
    user_bucket = bucket(flag["name"], user_id)
    on = bool(flag["enabled"]) and user_bucket < int(flag["rollout_percent"])
    return {"name": flag["name"], "user_id": user_id, "bucket": user_bucket, "on": on}


def delete_flag(args: dict[str, Any]) -> dict[str, Any]:
    flag = load_flag(str(args["name"]))
    call(CLIENT, "state", "delete", f"flag/{flag['name']}")
    log(CLIENT, "warn", "flag deleted", flag=flag["name"])
    return {"deleted": flag["name"]}


TOOLS = [
    Tool(
        "set_flag",
        "Create or change a feature flag. Every change is audited in the centralized logs with before and after values.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "enabled": {"type": "boolean"},
                "rollout_percent": {"type": "integer", "minimum": 0, "maximum": 100},
                "description": {"type": "string"},
            },
            "required": ["name", "enabled"],
        },
        set_flag,
    ),
    Tool("get_flag", "Read one feature flag.", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}, get_flag),
    Tool("list_flags", "List every feature flag.", {"type": "object", "properties": {}}, list_flags),
    Tool(
        "evaluate_flag",
        "Decide if a flag is on for a user. The same user always lands in the same rollout bucket.",
        {"type": "object", "properties": {"name": {"type": "string"}, "user_id": {"type": "string"}}, "required": ["name", "user_id"]},
        evaluate_flag,
    ),
    Tool("delete_flag", "Delete a feature flag.", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}, delete_flag),
]

if __name__ == "__main__":
    sys.exit(serve(CLIENT, "0.1.0", TOOLS))
