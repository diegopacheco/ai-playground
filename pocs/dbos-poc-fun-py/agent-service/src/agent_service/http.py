import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass

from agent_service.runs import RunStore

type Message = Mapping[str, object]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]
type Asgi = Callable[[Message, Receive, Send], Awaitable[None]]

DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Response:
    status: int
    body: object


def base36(value: int) -> str:
    text = ""
    while True:
        value, digit = divmod(value, 36)
        text = DIGITS[digit] + text
        if value == 0:
            return text


def new_run_id() -> str:
    return f"run-{base36(time.time_ns() // 1_000_000)}"


def parse_goal(body: bytes) -> str:
    try:
        value: object = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return ""
    goal = value.get("goal") if isinstance(value, dict) else None
    return goal.strip() if isinstance(goal, str) else ""


def to_json(value: object) -> bytes:
    def encode(item: object) -> object:
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        raise TypeError(f"cannot encode {type(item).__name__}")

    return json.dumps(value, default=encode).encode()


class Api:
    def __init__(self, store: RunStore, run_id: Callable[[], str] = new_run_id) -> None:
        self.store = store
        self.run_id = run_id

    async def handle(self, method: str, path: str, body: bytes) -> Response:
        match method, [part for part in path.split("/") if part]:
            case "GET", ["health"]:
                return Response(200, {"ok": True})
            case "POST", ["runs"]:
                goal = parse_goal(body)
                if not goal:
                    return Response(400, {"error": "goal is required"})
                run_id = self.run_id()
                await asyncio.to_thread(self.store.start, run_id, goal)
                return Response(202, {"runId": run_id})
            case "GET", ["runs"]:
                return Response(200, await asyncio.to_thread(self.store.runs))
            case "GET", ["runs", run_id]:
                return Response(200, await asyncio.to_thread(self.store.state, run_id))
            case _:
                return Response(404, {"error": f"no route {method} {path}"})


async def read_body(receive: Receive) -> bytes:
    body = b""
    while True:
        message = await receive()
        chunk = message.get("body", b"")
        body += chunk if isinstance(chunk, bytes) else b""
        if not message.get("more_body"):
            return body


async def lifespan(receive: Receive, send: Send, startup: Callable[[], None], shutdown: Callable[[], None]) -> None:
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            startup()
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            shutdown()
            await send({"type": "lifespan.shutdown.complete"})
            return


def create_app(api: Api, startup: Callable[[], None], shutdown: Callable[[], None]) -> Asgi:
    async def app(scope: Message, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await lifespan(receive, send, startup, shutdown)
            return
        try:
            response = await api.handle(str(scope["method"]), str(scope["path"]), await read_body(receive))
        except Exception as error:
            response = Response(500, {"error": str(error)})
        payload = to_json(response.body)
        await send({"type": "http.response.start", "status": response.status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": payload})

    return app
