from mcp_events.errors import invalid_params

PREFIX = "c_"


def encode(position: int) -> str:
    return f"{PREFIX}{position}"


def decode(cursor: str) -> int:
    digits = cursor.removeprefix(PREFIX)
    if digits == cursor or not digits.isdigit():
        raise invalid_params(f"unknown cursor {cursor}")
    return int(digits)
