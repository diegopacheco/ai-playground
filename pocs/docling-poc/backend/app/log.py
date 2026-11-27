import logging
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass

HEADER = "x-correlation-id"


@dataclass(frozen=True, slots=True)
class RequestContext:
    correlation_id: str
    path: str
    user_id: str | None = None


_current: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


class _ContextFilter(logging.Filter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _current.get()
        record.correlation_id = ctx.correlation_id if ctx else "-"
        record.path = ctx.path if ctx else "-"
        record.user_id = ctx.user_id if ctx and ctx.user_id else "-"
        record.service = self._service
        return True


def setup(service: str) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_ContextFilter(service))
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(service)s %(correlation_id)s %(path)s %(user_id)s %(name)s %(message)s"
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)


def bind(correlation_id: str | None, path: str) -> tuple[str, Token[RequestContext | None]]:
    cid = correlation_id or uuid.uuid4().hex
    token = _current.set(RequestContext(correlation_id=cid, path=path))
    return cid, token


def unbind(token: Token[RequestContext | None]) -> None:
    _current.reset(token)
