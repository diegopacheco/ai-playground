from agent_api.config import read_trigger_config
from agent_api.http import Api, Asgi, create_app
from agent_api.runs import TriggerRunStore
from agent_api.trigger_client import TriggerClient


def build() -> Asgi:
    config = read_trigger_config()
    print(f"agent-api triggering tasks on {config.api_url}", flush=True)
    return create_app(Api(TriggerRunStore(config, TriggerClient(config.api_url, config.secret_key))))


app = build()
