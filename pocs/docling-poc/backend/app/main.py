import os
import uuid
from datetime import UTC, datetime

from app import log
from app.api import Api
from app.config import Settings
from app.engine import DoclingEngine
from app.repo import InMemoryConversionRepository
from app.service import ConversionService


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def build() -> Api:
    settings = Settings.from_env(os.environ)
    log.setup(settings.service_name)
    service = ConversionService(
        engine=DoclingEngine(),
        repo=InMemoryConversionRepository(),
        clock=_now,
        new_id=_new_id,
        max_upload_bytes=settings.max_upload_bytes,
    )
    return Api(service, settings.max_upload_bytes)


app = build()
