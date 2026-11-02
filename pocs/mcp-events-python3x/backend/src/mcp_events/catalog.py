from collections.abc import Callable
from dataclasses import dataclass

from mcp_events.errors import event_not_found, invalid_params
from mcp_events.json_types import JsonObject, JsonValue

DEPLOY_STATUS_CHANGED = "deploy.status_changed"
ALERT_RAISED = "alert.raised"
ENVIRONMENTS = ["dev", "staging", "prod"]
STATUSES = ["queued", "building", "testing", "deployed", "failed"]
SEVERITIES = ["P1", "P2", "P3", "P4"]

type Predicate = Callable[[JsonObject], bool]


@dataclass(frozen=True)
class EventType:
    name: str
    description: str
    delivery: tuple[str, ...]
    input_schema: JsonObject
    payload_schema: JsonObject
    predicate: Callable[[JsonObject], Predicate]

    def descriptor(self) -> JsonObject:
        return {
            "name": self.name,
            "description": self.description,
            "delivery": list(self.delivery),
            "inputSchema": self.input_schema,
            "payloadSchema": self.payload_schema,
        }

    def subscribe(self, arguments: JsonObject) -> Predicate:
        validate_arguments(self.input_schema, arguments)
        return self.predicate(arguments)


def validate_arguments(schema: JsonObject, arguments: JsonObject) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for key, value in arguments.items():
        rule = properties.get(key)
        if not isinstance(rule, dict):
            raise invalid_params(f"unknown argument {key}")
        if not isinstance(value, str):
            raise invalid_params(f"argument {key} must be a string")
        allowed = rule.get("enum")
        if isinstance(allowed, list) and value not in allowed:
            raise invalid_params(f"argument {key} must be one of {', '.join(str(a) for a in allowed)}")


def string_property(description: str, allowed: list[str] | None = None) -> JsonObject:
    prop: JsonObject = {"type": "string", "description": description}
    if allowed is not None:
        prop["enum"] = list[JsonValue](allowed)
    return prop


def object_schema(properties: JsonObject) -> JsonObject:
    return {"type": "object", "properties": properties, "additionalProperties": False}


def deploy_predicate(arguments: JsonObject) -> Predicate:
    return lambda data: all(data.get(key) == value for key, value in arguments.items())


def alert_predicate(arguments: JsonObject) -> Predicate:
    floor = SEVERITIES.index(str(arguments.get("minSeverity", "P4")))
    return lambda data: str(data.get("severity")) in SEVERITIES and SEVERITIES.index(str(data.get("severity"))) <= floor


DEPLOY_EVENT = EventType(
    name=DEPLOY_STATUS_CHANGED,
    description="Fires every time a deployment moves to a new status: queued, building, testing, deployed or failed",
    delivery=("poll", "push"),
    input_schema=object_schema(
        {
            "service": string_property("Only deployments of this service"),
            "env": string_property("Only deployments to this environment", ENVIRONMENTS),
            "status": string_property("Only this status", STATUSES),
        }
    ),
    payload_schema=object_schema(
        {
            "deployId": string_property("Deployment id"),
            "service": string_property("Service being deployed"),
            "env": string_property("Target environment", ENVIRONMENTS),
            "status": string_property("New status", STATUSES),
            "triggeredBy": string_property("MCP client that started the deployment"),
        }
    ),
    predicate=deploy_predicate,
)

ALERT_EVENT = EventType(
    name=ALERT_RAISED,
    description="Fires when a deployment fails. prod raises P1, staging P2, dev P3",
    delivery=("poll", "push"),
    input_schema=object_schema(
        {"minSeverity": string_property("Only alerts at this severity or more urgent, P1 is the most urgent", SEVERITIES)}
    ),
    payload_schema=object_schema(
        {
            "alertId": string_property("Alert id"),
            "severity": string_property("Alert severity", SEVERITIES),
            "message": string_property("What happened"),
            "deployId": string_property("Deployment that failed"),
            "triggeredBy": string_property("MCP client that started the deployment"),
        }
    ),
    predicate=alert_predicate,
)


class Catalog:
    def __init__(self, event_types: list[EventType]) -> None:
        self._types = {event_type.name: event_type for event_type in event_types}

    def get(self, name: str) -> EventType:
        event_type = self._types.get(name)
        if event_type is None:
            raise event_not_found(name)
        return event_type

    def descriptors(self) -> list[JsonValue]:
        return [event_type.descriptor() for event_type in self._types.values()]


def default_catalog() -> Catalog:
    return Catalog([DEPLOY_EVENT, ALERT_EVENT])
