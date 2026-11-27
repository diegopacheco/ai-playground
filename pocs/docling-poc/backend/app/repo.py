import threading

from app.domain import Conversion


class InMemoryConversionRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Conversion] = {}

    def save(self, conversion: Conversion) -> None:
        with self._lock:
            self._items[conversion.id] = conversion

    def get(self, conversion_id: str) -> Conversion | None:
        with self._lock:
            return self._items.get(conversion_id)

    def all(self) -> tuple[Conversion, ...]:
        with self._lock:
            return tuple(self._items.values())
