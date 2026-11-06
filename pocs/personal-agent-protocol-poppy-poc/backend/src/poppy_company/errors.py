from poppy_company.json_types import JsonObject


class PoppyError(Exception):
    def __init__(self, status: int, code: str, description: str) -> None:
        super().__init__(description)
        self.status = status
        self.code = code
        self.description = description

    def to_json(self) -> JsonObject:
        return {"error": self.code, "error_description": self.description}


def invalid_session(description: str) -> PoppyError:
    return PoppyError(401, "invalid_session", description)


def invalid_request(description: str) -> PoppyError:
    return PoppyError(400, "invalid_request", description)
