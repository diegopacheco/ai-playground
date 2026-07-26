from __future__ import annotations

from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import config
from .indexing import configure_settings, service
from .routers import agents, annotations, chat, documents, rust, search

configure_settings()

app = FastAPI(title="llamaindex-poc", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (documents, chat, rust, search, agents, annotations):
    app.include_router(module.router)


def ollama_reachable() -> bool:
    try:
        with urlopen(f"{config.ollama_host}/api/tags", timeout=2) as response:
            return response.status == 200
    except (URLError, OSError):
        return False


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ollama": {"host": config.ollama_host, "reachable": ollama_reachable()},
        "index": service.stats(),
    }
