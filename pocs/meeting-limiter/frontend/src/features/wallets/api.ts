import { list, num, obj } from "../../decode";
import { http } from "../../http";
import { DAY_MS } from "../../time";
import { parseUser } from "../people/api";
import type { Wallet } from "./types";

function parseWallet(raw: unknown): Wallet {
  const o = obj(raw, "wallet");
  return {
    user: parseUser(o.user),
    budget: num(o, "budget"),
    spent: num(o, "spent"),
    balance: num(o, "balance"),
    minutesUsed: num(o, "minutesUsed"),
    minutesLimit: num(o, "minutesLimit"),
    meetings: num(o, "meetings"),
  };
}

export function midWeek(weekStart: Date): Date {
  return new Date(weekStart.getTime() + 3.5 * DAY_MS);
}

export const walletsApi = {
  week: (weekStart: Date) =>
    http.get(`/wallets?week=${encodeURIComponent(midWeek(weekStart).toISOString())}`, (raw) => list(raw, parseWallet, "wallets")),
};
