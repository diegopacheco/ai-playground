from urllib.parse import urlsplit

from poppy_company.json_types import JsonObject

PROTOCOL_VERSION = "0.4"


def poppy_document(base_url: str) -> JsonObject:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "organization": {"name": "Willow Inn", "domain": urlsplit(base_url).netloc},
        "agent": {"protocols": [{"type": "poppy", "endpoint": f"{base_url}/poppy/conversations"}]},
        "apis": [{"type": "openapi", "url": f"{base_url}/api/openapi.json", "description": "Room search and room holds"}],
        "feedback": {"endpoint": f"{base_url}/poppy/feedback"},
        "excluded_tasks": {"url": f"{base_url}/.well-known/poppy-excluded-tasks.json"},
    }


def excluded_tasks(base_url: str) -> JsonObject:
    return {
        "tasks": [
            {
                "task": "Pay for a stay",
                "alternative": {
                    "instructions": "Payment is taken at the front desk on arrival.",
                    "references": [{"type": "web", "url": f"{base_url}/front-desk"}],
                },
            },
            {"task": "Change the name on a reservation"},
        ]
    }


def openapi_document(base_url: str) -> JsonObject:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Willow Inn rooms", "version": "1.0.0"},
        "servers": [{"url": base_url}],
        "paths": {
            "/api/rooms": {"get": {"summary": "Search rooms", "parameters": [
                {"name": "max_price", "in": "query", "schema": {"type": "integer"}},
                {"name": "pets", "in": "query", "schema": {"type": "boolean"}},
            ]}},
            "/api/holds": {
                "get": {"summary": "List this User's room holds"},
                "post": {"summary": "Hold a room", "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["room_id", "nights"],
                    "properties": {"room_id": {"type": "string"}, "nights": {"type": "integer", "minimum": 1, "maximum": 14}},
                }}}}},
            },
        },
    }
