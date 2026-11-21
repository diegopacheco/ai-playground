import { bool, date, list, num, obj, str, strings } from "../../decode";
import { ApiError, http } from "../../http";
import { parseUser } from "../people/api";
import type { Charge, Meeting, MeetingRequest, Quote, QuoteLine } from "./types";

function parseCharge(raw: unknown): Charge {
  const o = obj(raw, "charge");
  return { userId: str(o, "userId"), amount: num(o, "amount") };
}

export function parseMeeting(raw: unknown): Meeting {
  const o = obj(raw, "meeting");
  return {
    id: str(o, "id"),
    title: str(o, "title"),
    organizerId: str(o, "organizerId"),
    attendeeIds: strings(o, "attendeeIds"),
    start: date(o, "start"),
    end: date(o, "end"),
    charges: list(o.charges ?? [], parseCharge, "charges"),
    totalCost: num(o, "totalCost"),
  };
}

function parseLine(raw: unknown): QuoteLine {
  const o = obj(raw, "quote line");
  return {
    user: parseUser(o.user),
    cost: num(o, "cost"),
    balanceBefore: num(o, "balanceBefore"),
    balanceAfter: num(o, "balanceAfter"),
    allowed: bool(o, "allowed"),
  };
}

export function parseQuote(raw: unknown): Quote {
  const o = obj(raw, "quote");
  return {
    minutes: num(o, "minutes"),
    lines: list(o.lines ?? [], parseLine, "lines"),
    totalCost: num(o, "totalCost"),
    allowed: bool(o, "allowed"),
  };
}

export function quoteFromError(err: unknown): Quote | undefined {
  if (!(err instanceof ApiError) || err.status !== 402) return undefined;
  return parseQuote(obj(err.body, "short funds").quote);
}

function body(r: MeetingRequest) {
  return { ...r, attendeeIds: [...r.attendeeIds], start: r.start.toISOString(), end: r.end.toISOString() };
}

function range(from: Date, to: Date): string {
  return `from=${encodeURIComponent(from.toISOString())}&to=${encodeURIComponent(to.toISOString())}`;
}

export const meetingsApi = {
  list: (from: Date, to: Date) => http.get(`/meetings?${range(from, to)}`, (raw) => list(raw, parseMeeting, "meetings")),
  quote: (r: MeetingRequest) => http.post("/meetings/quote", body(r), parseQuote),
  book: (r: MeetingRequest) => http.post("/meetings", body(r), parseMeeting),
  cancel: (id: string) => http.del(`/meetings/${encodeURIComponent(id)}`),
};

export { range as rangeQuery };
