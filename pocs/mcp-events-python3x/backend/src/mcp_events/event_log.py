import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from mcp_events.json_types import JsonObject

type Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class EventOccurrence:
    seq: int
    event_id: str
    name: str
    timestamp: datetime
    data: JsonObject

    def to_json(self, cursor: str | None) -> JsonObject:
        body: JsonObject = {
            "eventId": self.event_id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            "data": self.data,
        }
        if cursor is not None:
            body["cursor"] = cursor
        return body


@dataclass(frozen=True)
class ReadResult:
    events: list[EventOccurrence]
    position: int
    truncated: bool
    has_more: bool


class EventLog:
    def __init__(self, capacity: int = 500, clock: Clock = utc_now) -> None:
        self._entries: deque[EventOccurrence] = deque(maxlen=capacity)
        self._head = 0
        self._clock = clock
        self._changed = asyncio.Event()

    @property
    def head(self) -> int:
        return self._head

    def now(self) -> datetime:
        return self._clock()

    def append(self, name: str, data: JsonObject) -> EventOccurrence:
        self._head += 1
        event = EventOccurrence(self._head, f"evt_{self._head:06d}", name, self._clock(), data)
        self._entries.append(event)
        self._changed.set()
        self._changed = asyncio.Event()
        return event

    def change_signal(self) -> asyncio.Event:
        return self._changed

    def read(
        self,
        after: int,
        accept: Callable[[EventOccurrence], bool],
        limit: int | None = None,
        not_before: datetime | None = None,
    ) -> ReadResult:
        if after > self._head:
            return ReadResult([], self._head, True, False)
        oldest = self._entries[0].seq if self._entries else self._head + 1
        truncated = after < oldest - 1
        position = max(after, oldest - 1)
        events: list[EventOccurrence] = []
        for event in self._entries:
            if event.seq <= position:
                continue
            if not accept(event):
                position = event.seq
                continue
            if not_before is not None and event.timestamp < not_before:
                truncated = True
                position = event.seq
                continue
            if limit is not None and len(events) >= limit:
                return ReadResult(events, position, truncated, True)
            events.append(event)
            position = event.seq
        return ReadResult(events, self._head, truncated, False)
