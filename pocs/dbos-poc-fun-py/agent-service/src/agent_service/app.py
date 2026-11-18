from dbos import DBOS

from agent_service.config import database_url, read_config
from agent_service.http import Api, Asgi, create_app
from agent_service.serializer import AgentJsonSerializer
from agent_service.store import DbosRunStore


def build() -> Asgi:
    config = read_config()
    DBOS(config={"name": "dbos-agent-service", "system_database_url": database_url(), "serializer": AgentJsonSerializer()})
    print(f"agent-service using {config.provider} {config.model}", flush=True)
    return create_app(Api(DbosRunStore()), DBOS.launch, DBOS.destroy)


app = build()
