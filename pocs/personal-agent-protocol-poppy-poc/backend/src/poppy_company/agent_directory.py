from poppy_company.errors import invalid_request, invalid_session
from poppy_company.json_types import JsonObject


class AgentDirectory:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._agents: dict[str, JsonObject] = {}

    def client_id(self, name: str) -> str:
        return f"{self._base_url}/agent-host/{name}/agent.json"

    def publish(self, name: str, metadata: JsonObject) -> JsonObject:
        if not name.isalnum() or len(name) > 40:
            raise invalid_request("agent name must be 1 to 40 letters or digits")
        if metadata.get("client_id") != self.client_id(name):
            raise invalid_request(f"client_id must be {self.client_id(name)}")
        keys = jwks_keys(metadata)
        if not keys:
            raise invalid_request("jwks.keys must list at least one key")
        self._agents[self.client_id(name)] = metadata
        return metadata

    def get(self, name: str) -> JsonObject | None:
        return self._agents.get(self.client_id(name))

    def key(self, client_id: str, kid: str) -> JsonObject:
        metadata = self._agents.get(client_id)
        if metadata is None:
            raise invalid_session("unknown Personal Agent")
        for key in jwks_keys(metadata):
            if key.get("kid") == kid:
                return key
        raise invalid_session("delegation kid is not in the Personal Agent's jwks")


def jwks_keys(metadata: JsonObject) -> list[JsonObject]:
    jwks = metadata.get("jwks")
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    return [key for key in keys if isinstance(key, dict)] if isinstance(keys, list) else []
