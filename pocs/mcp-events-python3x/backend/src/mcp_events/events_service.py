import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta

from mcp_events import cursors
from mcp_events.catalog import Catalog, EventType, Predicate
from mcp_events.errors import unsupported_delivery
from mcp_events.event_log import EventLog, EventOccurrence
from mcp_events.json_types import JsonObject, JsonValue
from mcp_events.params import Params

SUBSCRIPTION_META = "io.modelcontextprotocol/subscriptionId"
NEXT_POLL_MS = 2000
DEFAULT_MAX_EVENTS = 50
MAX_EVENTS = 500
MAX_AGE_MS = 7 * 24 * 3600 * 1000


@dataclass(frozen=True)
class Subscription:
    event_type: EventType
    predicate: Predicate
    cursor: int | None
    not_before: datetime | None
    max_events: int

    def accepts(self, event: EventOccurrence) -> bool:
        return event.name == self.event_type.name and self.predicate(event.data)


def notification(method: str, params: JsonObject) -> JsonObject:
    return {"jsonrpc": "2.0", "method": method, "params": params}


class EventsService:
    def __init__(self, catalog: Catalog, log: EventLog, heartbeat_seconds: float = 15.0) -> None:
        self._catalog = catalog
        self._log = log
        self._heartbeat_seconds = heartbeat_seconds

    @property
    def head_cursor(self) -> str:
        return cursors.encode(self._log.head)

    def list_types(self) -> JsonObject:
        return {"events": self._catalog.descriptors()}

    def subscription(self, params: Params, mode: str) -> Subscription:
        event_type = self._catalog.get(params.required_str("name"))
        if mode not in event_type.delivery:
            raise unsupported_delivery(mode)
        predicate = event_type.subscribe(params.object("arguments"))
        cursor = params.optional_str("cursor")
        max_age = params.optional_int("maxAgeMs", 0, MAX_AGE_MS)
        max_events = params.optional_int("maxEvents", 1, MAX_EVENTS) or DEFAULT_MAX_EVENTS
        not_before = None if max_age is None else self._log.now() - timedelta(milliseconds=max_age)
        return Subscription(event_type, predicate, None if cursor is None else cursors.decode(cursor), not_before, max_events)

    def poll(self, params: Params) -> JsonObject:
        sub = self.subscription(params, "poll")
        if sub.cursor is None:
            return self._poll_result([], self._log.head, False, False)
        read = self._log.read(sub.cursor, sub.accepts, sub.max_events, sub.not_before)
        return self._poll_result(read.events, read.position, read.truncated, read.has_more)

    async def wait_for_change(self, timeout_seconds: float) -> bool:
        try:
            await asyncio.wait_for(self._log.change_signal().wait(), timeout_seconds)
        except TimeoutError:
            return False
        return True

    async def stream(self, sub: Subscription, subscription_id: JsonValue) -> AsyncIterator[JsonObject]:
        meta: JsonObject = {SUBSCRIPTION_META: subscription_id}
        position = self._log.head
        backlog: list[EventOccurrence] = []
        truncated = False
        if sub.cursor is not None:
            read = self._log.read(sub.cursor, sub.accepts, None, sub.not_before)
            backlog, position, truncated = read.events, read.position, read.truncated
        start = backlog[0].seq - 1 if backlog else position
        yield notification("notifications/events/active", {"cursor": cursors.encode(start), "truncated": truncated, "_meta": meta})
        for event in backlog:
            yield self._event_notification(event, meta)
        while True:
            signal = self._log.change_signal()
            read = self._log.read(position, sub.accepts)
            if read.truncated:
                yield notification("notifications/events/active", {"cursor": cursors.encode(read.position), "truncated": True, "_meta": meta})
            for event in read.events:
                yield self._event_notification(event, meta)
            position = read.position
            try:
                await asyncio.wait_for(signal.wait(), self._heartbeat_seconds)
            except TimeoutError:
                yield notification("notifications/events/heartbeat", {"cursor": cursors.encode(position), "_meta": meta})

    def _event_notification(self, event: EventOccurrence, meta: JsonObject) -> JsonObject:
        params = event.to_json(cursors.encode(event.seq))
        params["_meta"] = meta
        return notification("notifications/events/event", params)

    def _poll_result(self, events: list[EventOccurrence], position: int, truncated: bool, has_more: bool) -> JsonObject:
        return {
            "events": [event.to_json(None) for event in events],
            "cursor": cursors.encode(position),
            "truncated": truncated,
            "hasMore": has_more,
            "nextPollMs": NEXT_POLL_MS,
        }
