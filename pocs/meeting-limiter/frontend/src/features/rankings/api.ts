import { list, num, obj } from "../../decode";
import { http } from "../../http";
import { parseMeeting, rangeQuery } from "../meetings/api";
import { parseUser } from "../people/api";
import type { Period } from "./range";
import type { Rankings, Spender } from "./types";

export const RANK_LIMIT = 10;

function parseSpender(raw: unknown): Spender {
  const o = obj(raw, "spender");
  return { user: parseUser(o.user), spent: num(o, "spent"), minutes: num(o, "minutes"), meetings: num(o, "meetings") };
}

function parseRankings(raw: unknown): Rankings {
  const o = obj(raw, "rankings");
  return {
    meetings: list(o.meetings, parseMeeting, "meetings"),
    spenders: list(o.spenders, parseSpender, "spenders"),
    timeSinks: list(o.timeSinks, parseSpender, "timeSinks"),
  };
}

export const rankingsApi = {
  get: (p: Period) => http.get(`/rankings?${rangeQuery(p.from, p.to)}&limit=${RANK_LIMIT}`, parseRankings),
};
