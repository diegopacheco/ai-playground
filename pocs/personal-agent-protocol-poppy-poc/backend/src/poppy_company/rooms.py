from dataclasses import dataclass

from poppy_company.errors import invalid_request
from poppy_company.json_types import JsonObject

MAX_NIGHTS = 14


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    beds: int
    price_per_night: int
    pets_allowed: bool

    def to_json(self) -> JsonObject:
        return {"id": self.id, "name": self.name, "beds": self.beds, "price_per_night": self.price_per_night, "pets_allowed": self.pets_allowed}


@dataclass(frozen=True)
class Hold:
    id: str
    room: Room
    nights: int

    def to_json(self) -> JsonObject:
        return {"id": self.id, "room_id": self.room.id, "room_name": self.room.name, "nights": self.nights, "total": self.room.price_per_night * self.nights}


DEFAULT_ROOMS = (
    Room("rm_garden", "Garden Single", 1, 120, False),
    Room("rm_oak", "Oak Double", 2, 180, True),
    Room("rm_river", "River Suite", 2, 320, True),
    Room("rm_loft", "Family Loft", 4, 260, False),
)


class RoomsService:
    def __init__(self, rooms: tuple[Room, ...] = DEFAULT_ROOMS) -> None:
        self._rooms = {room.id: room for room in rooms}
        self._holds: dict[str, list[Hold]] = {}
        self._next = 0

    def search(self, max_price: int | None = None, pets: bool = False) -> list[Room]:
        return [room for room in self._rooms.values() if (max_price is None or room.price_per_night <= max_price) and (room.pets_allowed or not pets)]

    def hold(self, sub: str, room_id: str, nights: int) -> Hold:
        room = self._rooms.get(room_id)
        if room is None:
            raise invalid_request(f"unknown room {room_id}")
        if not 1 <= nights <= MAX_NIGHTS:
            raise invalid_request(f"nights must be between 1 and {MAX_NIGHTS}")
        self._next += 1
        hold = Hold(f"hld_{self._next}", room, nights)
        self._holds.setdefault(sub, []).append(hold)
        return hold

    def holds_for(self, sub: str) -> list[Hold]:
        return list(self._holds.get(sub, []))
