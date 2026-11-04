import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.settings import Settings, load_settings

from cli.api import ApiError, BearerClient
from cli.bootstrap import bootstrap
from cli.credentials import get_token
from cli.suite import load_suite


def parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def parse_fields(pairs: list[str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"field must be name=value: {pair}")
        fields[name] = parse_value(value)
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentinfra", description="CLI for the agent infra: auth, state, logs, evals")
    parser.add_argument("--client", default=os.environ.get("AGENTINFRA_CLIENT"), help="OAuth client id (tenant)")
    sub = parser.add_subparsers(dest="group", required=True)

    boot = sub.add_parser("bootstrap", help="generate the signing key and client secrets")
    boot.add_argument("--force", action="store_true")

    token = sub.add_parser("token", help="print a fresh access token")
    token.add_argument("--force", action="store_true")

    state = sub.add_parser("state", help="tenant scoped key/value state").add_subparsers(dest="action", required=True)
    put = state.add_parser("put")
    put.add_argument("key")
    put.add_argument("value", help="JSON value, plain text is stored as a string")
    state.add_parser("get").add_argument("key")
    state.add_parser("list").add_argument("--prefix", default="")
    state.add_parser("delete").add_argument("key")

    log = sub.add_parser("log", help="centralized logging").add_subparsers(dest="action", required=True)
    write = log.add_parser("write")
    write.add_argument("level", choices=["debug", "info", "warn", "error"])
    write.add_argument("message")
    write.add_argument("--field", action="append", default=[], help="name=value, repeatable")
    query = log.add_parser("query")
    query.add_argument("--since-minutes", default="60")
    query.add_argument("--limit", default="100")
    query.add_argument("--level")

    evals = sub.add_parser("eval", help="evals").add_subparsers(dest="action", required=True)
    evals.add_parser("run").add_argument("suite_file")
    runs = evals.add_parser("runs")
    runs.add_argument("--suite")
    runs.add_argument("--limit", default="20")
    evals.add_parser("show").add_argument("run_id")
    evals.add_parser("evaluators")
    return parser


def _state(api: Callable[[str], BearerClient], args: argparse.Namespace) -> Any:
    client = api("state")
    match args.action:
        case "put":
            return client.call("PUT", f"/v1/state/{args.key}", {"value": parse_value(args.value)})
        case "get":
            return client.call("GET", f"/v1/state/{args.key}")
        case "list":
            return client.call("GET", "/v1/state", query={"prefix": args.prefix})
        case _:
            return client.call("DELETE", f"/v1/state/{args.key}")


def _log(api: Callable[[str], BearerClient], args: argparse.Namespace) -> Any:
    client = api("logs")
    if args.action == "write":
        body = {"level": args.level, "message": args.message, "fields": parse_fields(args.field)}
        return client.call("POST", "/v1/logs", body)
    return client.call(
        "GET", "/v1/logs", query={"since_minutes": args.since_minutes, "limit": args.limit, "level": args.level}
    )


def _eval(api: Callable[[str], BearerClient], args: argparse.Namespace) -> Any:
    client = api("evals")
    match args.action:
        case "run":
            run = load_suite(Path(args.suite_file))
            return client.call("POST", "/v1/evals/runs", {"suite": run.suite, "cases": run.cases})
        case "runs":
            return client.call("GET", "/v1/evals/runs", query={"suite": args.suite, "limit": args.limit})
        case "show":
            return client.call("GET", f"/v1/evals/runs/{args.run_id}")
        case _:
            return client.call("GET", "/v1/evals/evaluators")


def execute(args: argparse.Namespace, settings: Settings) -> Any:
    if args.group == "bootstrap":
        return bootstrap(settings.secrets_dir, args.force)
    if not args.client:
        raise ValueError("--client or AGENTINFRA_CLIENT is required")
    if args.group == "token":
        return {"client": args.client, "access_token": get_token(settings.secrets_dir, settings.auth_url, args.client, args.force)}
    urls = {"state": settings.state_url, "logs": settings.logs_url, "evals": settings.evals_url}
    handlers = {"state": _state, "log": _log, "eval": _eval}
    token = get_token(settings.secrets_dir, settings.auth_url, args.client)

    def api(service: str) -> BearerClient:
        return BearerClient(urls[service], token)

    try:
        return handlers[args.group](api, args)
    except ApiError as error:
        if error.status != 401:
            raise
        token = get_token(settings.secrets_dir, settings.auth_url, args.client, force=True)
        return handlers[args.group](api, args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(args, load_settings())
    except ApiError as error:
        print(json.dumps({"status": error.status, "error": error.body}), file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(json.dumps({"status": 0, "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0
