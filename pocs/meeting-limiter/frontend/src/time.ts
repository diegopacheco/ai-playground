export const DAY_MS = 24 * 60 * 60 * 1000;

export function startOfDay(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}

export function addDays(d: Date, days: number): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate() + days, d.getHours(), d.getMinutes());
}

export function startOfWeek(d: Date): Date {
  const day = startOfDay(d);
  const offset = (day.getDay() + 6) % 7;
  return addDays(day, -offset);
}

export function startOfMonth(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), 1);
}

export function weekDays(weekStart: Date): readonly Date[] {
  return Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));
}

export function sameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

export function toDateKey(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function fromDateKey(key: string | undefined): Date | undefined {
  if (!key || !/^\d{4}-\d{2}-\d{2}$/.test(key)) return undefined;
  const [y, m, d] = key.split("-").map(Number);
  if (y === undefined || m === undefined || d === undefined) return undefined;
  const value = new Date(y, m - 1, d);
  return value.getMonth() === m - 1 ? value : undefined;
}

export function toTimeKey(d: Date): string {
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function combine(dateKey: string, timeKey: string): Date | undefined {
  const day = fromDateKey(dateKey);
  const match = /^(\d{2}):(\d{2})$/.exec(timeKey);
  if (!day || !match) return undefined;
  return new Date(day.getFullYear(), day.getMonth(), day.getDate(), Number(match[1]), Number(match[2]));
}

export function minutesSinceMidnight(d: Date): number {
  return d.getHours() * 60 + d.getMinutes();
}

const dayLabel = new Intl.DateTimeFormat("en-US", { weekday: "short", day: "numeric" });
const rangeLabel = new Intl.DateTimeFormat("en-US", { month: "long", day: "numeric", year: "numeric" });
const timeLabel = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit" });
const whenLabel = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric" });

export function formatDay(d: Date): string {
  return dayLabel.format(d);
}

export function formatTime(d: Date): string {
  return timeLabel.format(d);
}

export function formatWhen(start: Date, end: Date): string {
  return `${whenLabel.format(start)}, ${formatTime(start)} - ${formatTime(end)}`;
}

export function formatWeekRange(weekStart: Date): string {
  return `${rangeLabel.format(weekStart)} - ${rangeLabel.format(addDays(weekStart, 6))}`;
}
