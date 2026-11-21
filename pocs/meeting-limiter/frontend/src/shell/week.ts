import { useNavigate, useSearch } from "@tanstack/react-router";
import { addDays, fromDateKey, startOfWeek, toDateKey } from "../time";

export type WeekSearch = { readonly week?: string };

export function parseWeekSearch(search: Record<string, unknown>): WeekSearch {
  const week = search.week;
  return typeof week === "string" && fromDateKey(week) ? { week } : {};
}

export function useWeek() {
  const { week } = useSearch({ from: "__root__" });
  const navigate = useNavigate();
  const weekStart = startOfWeek(fromDateKey(week) ?? new Date());
  const goTo = (d: Date) => navigate({ to: ".", search: (prev) => ({ ...prev, week: toDateKey(startOfWeek(d)) }) });
  return {
    weekStart,
    goTo,
    prev: () => goTo(addDays(weekStart, -7)),
    next: () => goTo(addDays(weekStart, 7)),
    thisWeek: () => goTo(new Date()),
  };
}
