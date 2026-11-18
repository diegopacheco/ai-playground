from temporalio.client import Client

from agent_service.config import temporal_address, temporal_ui_url
from agent_service.http import Api, Asgi, create_app
from agent_service.runs import RunStore
from agent_service.store import TemporalRunStore


class Connection:
    def __init__(self, address: str, ui_url: str) -> None:
        self.address = address
        self.ui_url = ui_url
        self.store: RunStore | None = None

    async def connect(self) -> None:
        self.store = TemporalRunStore(await Client.connect(self.address, identity="agent-api"), self.ui_url)
        print(f"agent-api connected to temporal at {self.address}", flush=True)

    def get(self) -> RunStore:
        if self.store is None:
            raise RuntimeError("temporal client is not connected yet")
        return self.store


def build() -> Asgi:
    connection = Connection(temporal_address(), temporal_ui_url())
    return create_app(Api(connection.get), connection.connect)


app = build()
