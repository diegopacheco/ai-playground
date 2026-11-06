from poppy_company.rooms import RoomsService

TOPICS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pay", "card", "refund", "wire"), "I can't take payments here. Payment is taken at the front desk on arrival. If you need this, please send it as feedback."),
    (("pet", "dog", "cat"), "Pets are welcome in the Oak Double and the River Suite, with no extra fee."),
    (("check-in", "check in", "checkout", "check-out", "arrive"), "Check-in is from 3 pm and check-out is until 11 am."),
    (("park",), "Parking is free for guests, in the lot behind the inn."),
    (("breakfast",), "Breakfast is served from 7 to 10 am and is included with every room."),
)


class CompanyAgent:
    def __init__(self, rooms: RoomsService) -> None:
        self._rooms = rooms

    def reply(self, text: str) -> str:
        lowered = text.lower()
        for keywords, answer in TOPICS:
            if any(keyword in lowered for keyword in keywords):
                return answer
        if any(word in lowered for word in ("room", "available", "price", "stay", "night")):
            listing = "; ".join(f"{room.name} ${room.price_per_night}/night" for room in self._rooms.search())
            return f"We have these rooms: {listing}. Your Personal Agent can hold one through the rooms API."
        return "I'm the Willow Inn agent. I can help with rooms, check-in times, pets, parking and breakfast."
