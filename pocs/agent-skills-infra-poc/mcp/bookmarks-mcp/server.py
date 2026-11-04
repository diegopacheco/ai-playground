import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcpkit.agentinfra import call, get_or_none, log
from mcpkit.stdio_server import Tool, ToolError, serve

CLIENT = "bookmarks-mcp"


def key_for(url: str) -> str:
    return "bm/" + hashlib.sha256(url.encode()).hexdigest()[:16]


def validate_url(url: str) -> str:
    if not url.startswith(("http://", "https://")) or len(url) > 2000:
        raise ToolError("url must start with http:// or https://")
    return url


def all_bookmarks() -> list[dict[str, Any]]:
    return [item["value"] for item in call(CLIENT, "state", "list", "--prefix", "bm/")["items"]]


def save_bookmark(args: dict[str, Any]) -> dict[str, Any]:
    url = validate_url(str(args["url"]))
    tags = sorted({str(t).strip().lower() for t in args.get("tags", []) if str(t).strip()})
    record = {"url": url, "title": str(args.get("title") or url), "tags": tags, "saved_at": datetime.now(UTC).isoformat(timespec="seconds")}
    call(CLIENT, "state", "put", key_for(url), json.dumps(record))
    log(CLIENT, "info", "bookmark saved", url=url, tags=tags)
    return record


def list_bookmarks(args: dict[str, Any]) -> dict[str, Any]:
    tag = str(args.get("tag", "")).strip().lower()
    items = [b for b in all_bookmarks() if not tag or tag in b["tags"]]
    return {"count": len(items), "items": sorted(items, key=lambda b: b["title"].lower())}


def search_bookmarks(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args["query"]).strip().lower()
    if not query:
        raise ToolError("query must not be empty")
    items = [b for b in all_bookmarks() if query in b["title"].lower() or query in b["url"].lower()]
    return {"count": len(items), "items": items}


def delete_bookmark(args: dict[str, Any]) -> dict[str, Any]:
    url = validate_url(str(args["url"]))
    if get_or_none(CLIENT, key_for(url)) is None:
        raise ToolError(f"no bookmark for {url}")
    call(CLIENT, "state", "delete", key_for(url))
    log(CLIENT, "warn", "bookmark deleted", url=url)
    return {"deleted": url}


TOOLS = [
    Tool(
        "save_bookmark",
        "Save or update a bookmark with a title and tags. Stored in this server's own tenant of the shared state service.",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}, "title": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["url"],
        },
        save_bookmark,
    ),
    Tool(
        "list_bookmarks",
        "List saved bookmarks, optionally only those with a tag.",
        {"type": "object", "properties": {"tag": {"type": "string"}}},
        list_bookmarks,
    ),
    Tool(
        "search_bookmarks",
        "Find bookmarks whose title or url contains the query, case insensitive.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        search_bookmarks,
    ),
    Tool(
        "delete_bookmark",
        "Delete the bookmark for a url.",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        delete_bookmark,
    ),
]

if __name__ == "__main__":
    sys.exit(serve(CLIENT, "0.1.0", TOOLS))
