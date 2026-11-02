import logging
import os

from mcp_events.catalog import default_catalog
from mcp_events.deployer import Deployer
from mcp_events.event_log import EventLog
from mcp_events.events_service import EventsService
from mcp_events.http_app import McpHttpApp
from mcp_events.protocol import Protocol
from mcp_events.sessions import Sessions
from mcp_events.tools import Tools


def create_app(heartbeat_seconds: float = 15.0, step_seconds: float = 1.2) -> McpHttpApp:
    log = EventLog()
    events = EventsService(default_catalog(), log, heartbeat_seconds)
    tools = Tools(events, Deployer(log, step_seconds))
    return McpHttpApp(Protocol(events, tools, Sessions()))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
app = create_app(float(os.environ.get("MCP_EVENTS_HEARTBEAT_SECONDS", "15")), float(os.environ.get("MCP_EVENTS_STEP_SECONDS", "1.2")))
