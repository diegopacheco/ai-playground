import { combine } from "../../time";
import type { MeetingRequest } from "./types";

export const SLOT_MINUTES = 15;
export const MAX_MINUTES = 8 * 60;

export type MeetingDraft = {
  readonly title: string;
  readonly organizerId: string;
  readonly attendeeIds: readonly string[];
  readonly date: string;
  readonly time: string;
  readonly minutes: number;
};

export function toRequest(d: MeetingDraft): MeetingRequest | undefined {
  const start = combine(d.date, d.time);
  if (!start || d.organizerId === "" || d.minutes <= 0) return undefined;
  return {
    title: d.title.trim(),
    organizerId: d.organizerId,
    attendeeIds: d.attendeeIds.filter((id) => id !== d.organizerId),
    start,
    end: new Date(start.getTime() + d.minutes * 60000),
  };
}

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

export const TIME_OPTIONS: readonly string[] = Array.from({ length: (24 * 60) / SLOT_MINUTES }, (_, i) => {
  const m = i * SLOT_MINUTES;
  return `${pad(Math.floor(m / 60))}:${pad(m % 60)}`;
});

export const DURATION_OPTIONS: readonly number[] = Array.from({ length: MAX_MINUTES / SLOT_MINUTES }, (_, i) => (i + 1) * SLOT_MINUTES);
