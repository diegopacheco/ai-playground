import secrets
from dataclasses import dataclass, field
from datetime import datetime

from mcp_events.event_log import utc_now
from mcp_events.json_types import JsonObject

UNKNOWN_CLIENT = "unknown-client"


@dataclass
class Session:
    session_id: str
    client_name: str
    client_version: str
    protocol_version: str
    started_at: datetime = field(default_factory=utc_now)
    last_seen: datetime = field(default_factory=utc_now)
    requests: int = 0
    closed: bool = False

    def to_json(self) -> JsonObject:
        return {
            "sessionId": self.session_id[:8],
            "clientName": self.client_name,
            "clientVersion": self.client_version,
            "protocolVersion": self.protocol_version,
            "startedAt": self.started_at.isoformat(),
            "lastSeen": self.last_seen.isoformat(),
            "requests": self.requests,
            "closed": self.closed,
        }


class Sessions:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, client_name: str, client_version: str, protocol_version: str) -> Session:
        session = Session(secrets.token_hex(16), client_name, client_version, protocol_version)
        self._sessions[session.session_id] = session
        return session

    def touch(self, session_id: str | None) -> Session | None:
        session = self._sessions.get(session_id or "")
        if session is not None:
            session.last_seen = utc_now()
            session.requests += 1
        return session

    def close(self, session_id: str | None) -> None:
        session = self._sessions.get(session_id or "")
        if session is not None:
            session.closed = True

    def all(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.last_seen, reverse=True)
