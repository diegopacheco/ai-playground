from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from typing import Literal, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class Settings(TypedDict):
    primary: str
    secondary: str
    interval: int
    timezone: str


@dataclass(frozen=True)
class Event:
    id: str
    role: Literal["Primary", "Secondary"]
    date: str
    week_start: str
    week_end: str
    summary: str
    start: str
    end: str


@dataclass(frozen=True)
class Schedule:
    events: list[Event]
    years: list[int]
    timezone: str
    through: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate(settings: Settings, today: date | None = None) -> Schedule:
    try:
        zone = ZoneInfo(settings["timezone"])
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("Choose a valid IANA time zone.") from None
    current = today or datetime.now(zone).date()
    try:
        primary = date.fromisoformat(settings["primary"])
        secondary = date.fromisoformat(settings["secondary"])
    except ValueError:
        raise ValueError("Enter valid first dates for both roles.") from None
    interval = settings["interval"]
    if not 2 <= interval <= 52:
        raise ValueError("The rotation must repeat every 2 to 52 weeks.")
    period = interval * 7
    distance = (secondary - primary).days % period
    if distance < 7 or distance > period - 7:
        raise ValueError("Primary and secondary weeks overlap. Adjust a date or the repeat interval.")
    end = date(current.year + 1, 12, 31)
    if primary > end or secondary > end:
        raise ValueError("Both first dates must be on or before the end of next year.")
    events: list[Event] = []
    anchors: list[tuple[Literal["Primary", "Secondary"], date]] = [("Primary", primary), ("Secondary", secondary)]
    for role, anchor in anchors:
        skip = max(0, (current - anchor).days // period)
        week = anchor + timedelta(days=skip * period)
        while week <= end:
            week_end = week + timedelta(days=6)
            title = f"🔥 On Call {role} {week:%d/%m}-{week_end:%d/%m} week"
            for offset in range(7):
                day = week + timedelta(days=offset)
                if current <= day <= end:
                    start = datetime.combine(day, time(4), zone)
                    finish = datetime.combine(day, time(6), zone)
                    identifier = sha256(f"onward:{role}:{day}:{zone.key}:04-06".encode()).hexdigest()[:40]
                    events.append(Event(identifier, role, day.isoformat(), week.isoformat(), week_end.isoformat(), title, start.isoformat(), finish.isoformat()))
            week += timedelta(days=period)
    return Schedule(sorted(events, key=lambda event: (event.date, event.role)), [current.year, current.year + 1], zone.key, end.isoformat())


def calendar_file(schedule: Schedule) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Onward//On Call Calendar//EN", "CALSCALE:GREGORIAN", "METHOD:PUBLISH", "X-WR-CALNAME:Onward On Call"]
    for event in schedule.events:
        start = datetime.fromisoformat(event.start).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        end = datetime.fromisoformat(event.end).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lines.extend(["BEGIN:VEVENT", f"UID:{event.id}@onward.local", f"DTSTAMP:{stamp}", f"DTSTART:{start}", f"DTEND:{end}", f"SUMMARY:{event.summary}", "END:VEVENT"])
    lines.append("END:VCALENDAR")
    folded: list[str] = []
    for line in lines:
        part = ""
        for char in line:
            if len((part + char).encode()) > 75:
                folded.append(part)
                part = " "
            part += char
        folded.append(part)
    return "\r\n".join(folded) + "\r\n"
