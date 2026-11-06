import os
import re
import time
from collections.abc import Callable
from urllib.parse import urlsplit

from poppy_company.agent_directory import AgentDirectory
from poppy_company.company_agent import CompanyAgent
from poppy_company.conversations import Conversations
from poppy_company.discovery import PROTOCOL_VERSION, excluded_tasks, openapi_document, poppy_document
from poppy_company.errors import PoppyError, invalid_request
from poppy_company.feedback import FeedbackBox
from poppy_company.http import Receive, Request, Scope, Send, read_body, send_json
from poppy_company.inspector import inspect
from poppy_company.json_types import JsonObject, JsonValue
from poppy_company.rooms import RoomsService
from poppy_company.session_verifier import Session, SessionVerifier

AGENT_HOST = re.compile(r"^/agent-host/([^/]+)/agent\.json$")
CONVERSATION = re.compile(r"^/poppy/conversations/([A-Za-z0-9_-]+)/(messages|events|close)$")

type Result = tuple[int, JsonValue]


class CompanyApp:
    def __init__(self, base_url: str, clock: Callable[[], float] = time.time) -> None:
        self.base_url = base_url.rstrip("/")
        self.directory = AgentDirectory(self.base_url)
        self.verifier = SessionVerifier(self.directory, self.base_url, clock)
        self.rooms = RoomsService()
        self.conversations = Conversations(CompanyAgent(self.rooms))
        self.feedback = FeedbackBox()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while (await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        request = Request(scope, await read_body(receive))
        origin = request.header("origin")
        if request.method == "OPTIONS":
            await send_json(send, 204, None, origin)
            return
        try:
            status, body = self.route(request)
        except PoppyError as error:
            status, body = error.status, error.to_json()
        await send_json(send, status, body, origin)

    def route(self, request: Request) -> Result:
        match request.method, request.path:
            case "GET", "/health":
                return 200, {"status": "ok"}
            case "GET", "/.well-known/poppy.json":
                return 200, poppy_document(self.base_url)
            case "GET", "/.well-known/poppy-excluded-tasks.json":
                return 200, excluded_tasks(self.base_url)
            case "GET", "/api/openapi.json":
                return 200, openapi_document(self.base_url)
            case "POST", "/inspector/validate":
                return 200, self.validate(request.json())
        if found := AGENT_HOST.match(request.path):
            return self.agent_host(request, found.group(1))
        return self.session_route(request, self.session(request))

    def session_route(self, request: Request, session: Session) -> Result:
        match request.method, request.path:
            case "GET", "/api/rooms":
                return 200, {"rooms": [room.to_json() for room in self.rooms.search(max_price(request), request.param("pets") == "true")]}
            case "GET", "/api/holds":
                return 200, {"user_id": session.sub, "holds": [hold.to_json() for hold in self.rooms.holds_for(session.sub)]}
            case "POST", "/api/holds":
                body = request.json()
                room_id, nights = body.get("room_id"), body.get("nights")
                if not isinstance(room_id, str) or not isinstance(nights, int):
                    raise invalid_request("room_id and nights are required")
                return 201, self.rooms.hold(session.sub, room_id, nights).to_json()
            case "POST", "/poppy/conversations":
                return 201, self.conversations.start(session, request.json())
            case "POST", "/poppy/feedback":
                report = self.feedback.submit(session.client_id, request.json())
                return 202, {"status": "received", "task": report.task}
        found = CONVERSATION.match(request.path)
        if found is None:
            raise PoppyError(404, "not_found", f"no route for {request.method} {request.path}")
        conversation_id, action = found.groups()
        match request.method, action:
            case "POST", "messages":
                return 202, self.conversations.send(session, conversation_id, request.json())
            case "GET", "events":
                return 200, self.conversations.events(session, conversation_id, request.param("cursor"))
            case "POST", "close":
                return 200, self.conversations.close(session, conversation_id)
        raise PoppyError(405, "method_not_allowed", f"{request.method} is not allowed on {request.path}")

    def session(self, request: Request) -> Session:
        if request.header("poppy-version") != PROTOCOL_VERSION:
            raise PoppyError(400, "version_unsupported", f"Poppy-Version must be {PROTOCOL_VERSION}")
        token = request.header("poppy-session")
        if not token:
            raise PoppyError(401, "invalid_session", "Poppy-Session header is required")
        return self.verifier.verify(token, request.method, f"{self.base_url}{request.path}")

    def agent_host(self, request: Request, name: str) -> Result:
        if request.method == "PUT":
            return 200, self.directory.publish(name, request.json())
        metadata = self.directory.get(name)
        if request.method != "GET" or metadata is None:
            raise PoppyError(404, "not_found", "no Personal Agent metadata at this path")
        return 200, metadata

    def validate(self, body: JsonObject) -> JsonObject:
        document, host = body.get("document"), body.get("host")
        if not isinstance(document, dict):
            raise invalid_request("document must be a JSON object")
        served_by = host if isinstance(host, str) and host else urlsplit(self.base_url).netloc
        checks = inspect(document, served_by)
        return {"host": served_by, "valid": all(check.level != "fail" for check in checks), "checks": [check.to_json() for check in checks]}


def max_price(request: Request) -> int | None:
    value = request.param("max_price")
    if value is None or value == "":
        return None
    if not value.isdigit():
        raise invalid_request("max_price must be a whole number")
    return int(value)


app = CompanyApp(os.environ.get("POPPY_BASE_URL", "http://localhost:8741"))
