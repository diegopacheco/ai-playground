from mcp_events.json_types import JsonObject

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
NOT_FOUND = -32023
UNSUPPORTED = -32026


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: JsonObject | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_json(self) -> JsonObject:
        error: JsonObject = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


def invalid_params(reason: str) -> RpcError:
    return RpcError(INVALID_PARAMS, "InvalidParams", {"reason": reason})


def event_not_found(name: str) -> RpcError:
    return RpcError(NOT_FOUND, "NotFound", {"kind": "event", "name": name})


def unsupported_delivery(mode: str) -> RpcError:
    return RpcError(UNSUPPORTED, "Unsupported", {"feature": "deliveryMode", "value": mode})
