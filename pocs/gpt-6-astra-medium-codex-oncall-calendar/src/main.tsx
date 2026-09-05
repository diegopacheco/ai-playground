import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { FormEvent } from "react";
import { importToGoogle, loadGoogle } from "./google";
import "./style.css";

export type Shift = {
  id: string;
  role: "Primary" | "Secondary";
  date: string;
  week_start: string;
  week_end: string;
  summary: string;
  start: string;
  end: string;
};
export type Schedule = {
  events: Shift[];
  years: number[];
  timezone: string;
  through: string;
};
type Settings = {
  primary: string;
  secondary: string;
  interval: string;
  timezone: string;
};
type IconName =
  | "arrow"
  | "calendar"
  | "download"
  | "clock"
  | "grid"
  | "list"
  | "check"
  | "close";
const paths: Record<IconName, string> = {
  arrow: "M5 12h14m-6-6 6 6-6 6",
  calendar: "M8 3v4m8-4v4M4 10h16M5 5h14a1 1 0 0 1 1 1v14H4V6a1 1 0 0 1 1-1Z",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
  clock: "M12 8v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  grid: "M3 3h7v7H3Zm11 0h7v7h-7ZM3 14h7v7H3Zm11 0h7v7h-7Z",
  list: "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
  check: "m5 12 4 4L19 6",
  close: "m6 6 12 12M6 18 18 6",
};
function Icon({ name }: { name: IconName }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name]} />
    </svg>
  );
}
function iso(day: Date) {
  return `${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, "0")}-${String(day.getDate()).padStart(2, "0")}`;
}
function displayDate(
  value: string,
  options: Intl.DateTimeFormatOptions = { month: "short", day: "numeric" },
) {
  return new Date(`${value}T12:00:00`).toLocaleDateString("en-US", options);
}
function initialSettings(): Settings {
  const first = new Date();
  first.setDate(first.getDate() + ((8 - first.getDay()) % 7));
  const second = new Date(first);
  second.setDate(second.getDate() + 7);
  const defaults = {
    primary: iso(first),
    secondary: iso(second),
    interval: "4",
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  };
  try {
    const saved: unknown = JSON.parse(
      localStorage.getItem("onward-settings") || "null",
    );
    if (
      saved &&
      typeof saved === "object" &&
      "primary" in saved &&
      typeof saved.primary === "string" &&
      "secondary" in saved &&
      typeof saved.secondary === "string" &&
      "interval" in saved &&
      typeof saved.interval === "string" &&
      "timezone" in saved &&
      typeof saved.timezone === "string"
    )
      return {
        primary: saved.primary,
        secondary: saved.secondary,
        interval: saved.interval,
        timezone: saved.timezone,
      };
  } catch {
    return defaults;
  }
  return defaults;
}

function App() {
  const [settings, setSettings] = useState(initialSettings);
  const [applied, setApplied] = useState(settings);
  const [schedule, setSchedule] = useState<Schedule | null>(null);
  const [year, setYear] = useState(new Date().getFullYear());
  const [view, setView] = useState<"calendar" | "list">("calendar");
  const [filter, setFilter] = useState("All shifts");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");
  const [clientId, setClientId] = useState("");
  const [googleReady, setGoogleReady] = useState(false);
  const [importing, setImporting] = useState(false);
  const modal = useRef<HTMLDialogElement>(null);
  const request = useRef(0);
  const dirty = JSON.stringify(settings) !== JSON.stringify(applied);
  const query = new URLSearchParams(applied).toString();
  async function generate(next: Settings) {
    const id = ++request.current;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(
        `/api/schedule?${new URLSearchParams(next)}`,
      );
      const result = await response.json();
      if (!response.ok)
        throw new Error(result.error || "Unable to create your schedule.");
      if (id !== request.current) return;
      const data = result as Schedule;
      setSchedule(data);
      setApplied(next);
      setYear((previous) =>
        data.years.includes(previous) ? previous : data.years[0],
      );
      try {
        localStorage.setItem("onward-settings", JSON.stringify(next));
      } catch {
        setNotice("Your browser could not save these settings.");
      }
    } catch (cause) {
      if (id === request.current)
        setError(
          cause instanceof Error
            ? cause.message
            : "The server is unavailable. Please try again.",
        );
    } finally {
      if (id === request.current) setLoading(false);
    }
  }
  useEffect(() => {
    void generate(settings);
    void fetch("/api/config")
      .then((response) => response.json())
      .then((config: { googleClientId: string }) => {
        setClientId(config.googleClientId);
        if (config.googleClientId)
          void loadGoogle()
            .then(() => setGoogleReady(true))
            .catch(() =>
              setNotice(
                "Google sign-in could not load. You can still download your calendar.",
              ),
            );
      })
      .catch(() => {});
  }, []);
  function submit(event: FormEvent) {
    event.preventDefault();
    void generate(settings);
  }
  function update(key: keyof Settings, value: string) {
    setSettings((previous) => ({ ...previous, [key]: value }));
  }
  const events = schedule?.events ?? [];
  const visible = events.filter(
    (event) =>
      event.date.startsWith(String(year)) &&
      (filter === "All shifts" || event.role === filter),
  );
  const dayRoles = new Map(visible.map((event) => [event.date, event]));
  const weeks = [
    ...new Map(
      visible.map((event) => [`${event.role}-${event.week_start}`, event]),
    ).values(),
  ];
  const next = events[0];
  const totalWeeks = new Set(
    events.map((event) => `${event.role}-${event.week_start}`),
  ).size;
  const zones = Array.from(
    new Set([settings.timezone, ...Intl.supportedValuesOf("timeZone")]),
  ).sort();
  const exportDisabled = !events.length || loading || dirty || !!error;
  function download() {
    const anchor = document.createElement("a");
    anchor.href = `/api/calendar.ics?${query}`;
    anchor.download = "onward-oncall.ics";
    anchor.click();
  }
  async function connectGoogle() {
    setImporting(true);
    setNotice("Waiting for Google authorization…");
    try {
      await importToGoogle(clientId, events, applied.timezone, setNotice);
    } catch (cause) {
      setNotice(
        cause instanceof Error
          ? cause.message
          : "Import failed. Please try again.",
      );
    } finally {
      setImporting(false);
    }
  }
  return (
    <>
      <header className="topbar">
        <a className="brand" href="/" aria-label="Onward home">
          <img src="/logo.svg" alt="" />
          onward<span className="brand-dot">.</span>
        </a>
        <span className="product-label">THE ON-CALL PLANNER</span>
        <a className="header-link" href="#rotation">
          Your rotation <Icon name="arrow" />
        </a>
      </header>
      <main>
        <section className="intro">
          <div>
            <div className="eyebrow">
              <span className="live-dot" /> LESS GUESSWORK. MORE HEADSPACE.
            </div>
            <h1>
              On-call, <span>on your terms.</span>
            </h1>
            <p>
              Two dates. Your whole rotation. Make room for everything else.
            </p>
          </div>
          <div className="intro-note">
            <Icon name="clock" />
            <span>
              A little planning.
              <br />
              <strong>A lot of peace of mind.</strong>
            </span>
          </div>
        </section>
        <div className="workspace">
          <aside>
            <form className="rotation panel" onSubmit={submit} id="rotation">
              <div className="section-number">01 / SET YOUR RHYTHM</div>
              <h2>Your rotation</h2>
              <p className="muted">
                Start with the first day of each duty week. We’ll take it from
                there.
              </p>
              <label htmlFor="primary">
                <span className="role-dot primary" />
                First primary date
              </label>
              <input
                id="primary"
                type="date"
                required
                value={settings.primary}
                onChange={(event) => update("primary", event.target.value)}
              />
              <label htmlFor="secondary">
                <span className="role-dot secondary" />
                First secondary date
              </label>
              <input
                id="secondary"
                type="date"
                required
                value={settings.secondary}
                onChange={(event) => update("secondary", event.target.value)}
              />
              <label htmlFor="interval">Each role repeats every</label>
              <div className="interval-field">
                <input
                  id="interval"
                  type="number"
                  min="2"
                  max="52"
                  required
                  value={settings.interval}
                  onChange={(event) => update("interval", event.target.value)}
                />
                <span>weeks</span>
              </div>
              <label htmlFor="timezone">Time zone</label>
              <select
                id="timezone"
                value={settings.timezone}
                onChange={(event) => update("timezone", event.target.value)}
              >
                {zones.map((zone) => (
                  <option key={zone} value={zone}>
                    {zone.replaceAll("_", " ")}
                  </option>
                ))}
              </select>
              <div className="shift-hours">
                <Icon name="clock" />
                <span>
                  7 days per duty week<strong>4:00 – 6:00 AM, every day</strong>
                </span>
              </div>
              <button
                className="button dark generate"
                disabled={loading}
                type="submit"
              >
                {loading ? "Building your schedule…" : "Generate schedule"}
                <Icon name="arrow" />
              </button>
              {dirty && (
                <p className="form-note">Generate to apply your changes.</p>
              )}
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
            </form>
            <div className="next-card">
              <div className="section-number">NEXT UP</div>
              <div className="next-heading">
                <span className={`badge ${next?.role.toLowerCase() ?? ""}`}>
                  {next ? `${next.role} duty` : "Your next shift"}
                </span>
                <span aria-hidden="true">↗</span>
              </div>
              <h3>
                {next
                  ? displayDate(next.date, { month: "long", day: "numeric" })
                  : "A clear calendar"}
              </h3>
              <p>
                {next
                  ? `${displayDate(next.date, { weekday: "long" })} · 4:00 – 6:00 AM`
                  : "Set your dates to get started."}
              </p>
              <div className="next-footer">
                <span className="live-dot" />
                {next
                  ? `Duty week ends ${displayDate(next.week_end)}`
                  : "Your next shift will appear here"}
              </div>
            </div>
            <p className="local-note">
              <Icon name="check" />
              Your rotation settings stay in this browser.
            </p>
          </aside>
          <section
            className="schedule-area"
            aria-label="Your on-call schedule"
            aria-busy={loading}
          >
            <div className="overview">
              <div>
                <span className="stat-label">THE ROAD AHEAD</span>
                <strong>
                  {totalWeeks}
                  <small>duty weeks</small>
                </strong>
              </div>
              <div>
                <span className="stat-label">PRIMARY / SECONDARY</span>
                <strong>
                  {events.filter((event) => event.role === "Primary").length}
                  <span className="stat-divider">/</span>
                  {events.filter((event) => event.role === "Secondary").length}
                  <small>days</small>
                </strong>
              </div>
              <div>
                <span className="stat-label">PLANNED THROUGH</span>
                <strong>
                  {schedule ? `Dec ${schedule.years[1]}` : "—"}
                  <span className="tiny-spark">✳</span>
                </strong>
              </div>
            </div>
            <div className="calendar-panel panel">
              <div className="schedule-heading">
                <div>
                  <div className="section-number">02 / SEE THE BIG PICTURE</div>
                  <h2>Your on-call calendar</h2>
                </div>
                <button
                  className="button google"
                  disabled={exportDisabled}
                  onClick={() => {
                    setNotice("");
                    modal.current?.showModal();
                  }}
                >
                  <span className="google-g" aria-hidden="true">
                    G
                  </span>
                  Add to Google Calendar
                  <Icon name="arrow" />
                </button>
              </div>
              <div className="calendar-toolbar">
                <div className="year-tabs" aria-label="Calendar year">
                  {(schedule?.years ?? [year, year + 1]).map((value) => (
                    <button
                      key={value}
                      aria-pressed={year === value}
                      onClick={() => setYear(value)}
                    >
                      {value}
                    </button>
                  ))}
                </div>
                <div className="view-controls">
                  <label className="sr-only" htmlFor="role-filter">
                    Filter role
                  </label>
                  <select
                    id="role-filter"
                    value={filter}
                    onChange={(event) => setFilter(event.target.value)}
                  >
                    <option>All shifts</option>
                    <option>Primary</option>
                    <option>Secondary</option>
                  </select>
                  <div className="view-switch">
                    <button
                      aria-label="Calendar view"
                      aria-pressed={view === "calendar"}
                      onClick={() => setView("calendar")}
                    >
                      <Icon name="grid" />
                    </button>
                    <button
                      aria-label="List view"
                      aria-pressed={view === "list"}
                      onClick={() => setView("list")}
                    >
                      <Icon name="list" />
                    </button>
                  </div>
                </div>
              </div>
              <div className="legend">
                <span>
                  <i className="primary" />
                  Primary
                </span>
                <span>
                  <i className="secondary" />
                  Secondary
                </span>
                <span>
                  <i className="today-key" />
                  Today
                </span>
                <span className="legend-note">
                  A year at a glance. A week at a time.
                </span>
              </div>
              {view === "calendar" ? (
                <div className="months">
                  {Array.from({ length: 12 }, (_, month) => {
                    const first = new Date(year, month, 1);
                    const offset = (first.getDay() + 6) % 7;
                    const days = new Date(year, month + 1, 0).getDate();
                    return (
                      <section
                        className="month"
                        key={month}
                        aria-label={first.toLocaleDateString("en-US", {
                          month: "long",
                          year: "numeric",
                        })}
                      >
                        <div className="month-heading">
                          <h3>
                            {first.toLocaleDateString("en-US", {
                              month: "long",
                            })}
                          </h3>
                          <span>{String(month + 1).padStart(2, "0")}</span>
                        </div>
                        <div className="month-grid">
                          {["M", "T", "W", "T", "F", "S", "S"].map(
                            (day, index) => (
                              <span className="weekday" key={`w${index}`}>
                                {day}
                              </span>
                            ),
                          )}
                          {Array.from({ length: offset }, (_, index) => (
                            <span key={`b${index}`} />
                          ))}
                          {Array.from({ length: days }, (_, index) => {
                            const day = iso(new Date(year, month, index + 1));
                            const shift = dayRoles.get(day);
                            const today = iso(new Date()) === day;
                            return (
                              <span
                                key={day}
                                className={`day ${shift?.role.toLowerCase() ?? ""} ${today ? "today" : ""}`}
                                title={`${displayDate(day, { month: "long", day: "numeric", year: "numeric" })}${shift ? ` · ${shift.role} · 4–6 AM · ${applied.timezone}` : ""}`}
                                aria-label={`${day}${shift ? ` ${shift.role} 4 to 6 AM` : ""}${today ? " today" : ""}`}
                              >
                                {index + 1}
                              </span>
                            );
                          })}
                        </div>
                      </section>
                    );
                  })}
                </div>
              ) : (
                <div className="week-list">
                  {weeks.length ? (
                    weeks.map((week) => (
                      <article
                        className="week-row"
                        key={`${week.role}-${week.week_start}`}
                      >
                        <div className={`week-icon ${week.role.toLowerCase()}`}>
                          <Icon name="calendar" />
                        </div>
                        <div>
                          <strong>
                            {displayDate(week.week_start)} –{" "}
                            {displayDate(week.week_end)}
                          </strong>
                          <p>
                            4:00 – 6:00 AM ·{" "}
                            {
                              visible.filter(
                                (event) =>
                                  event.week_start === week.week_start &&
                                  event.role === week.role,
                              ).length
                            }{" "}
                            upcoming days in {year}
                          </p>
                        </div>
                        <span className={`badge ${week.role.toLowerCase()}`}>
                          {week.role}
                        </span>
                      </article>
                    ))
                  ) : (
                    <p className="empty-state">
                      No upcoming shifts for this selection.
                    </p>
                  )}
                </div>
              )}
              <div className="calendar-footer">
                <span>
                  <Icon name="clock" />
                  All shifts in {applied.timezone.replaceAll("_", " ")}
                </span>
                <button
                  className="text-button"
                  disabled={exportDisabled}
                  onClick={download}
                >
                  <Icon name="download" />
                  Download .ics
                </button>
              </div>
            </div>
            <div className="bottom-note">
              <span>YOUR TIME, ACCOUNTED FOR.</span>
              <p>From today through next year. Past shifts are left behind.</p>
            </div>
          </section>
        </div>
      </main>
      <footer className="site-footer">
        <span>
          onward.{" "}
          <span className="muted">
            Built for the people who keep things running.
          </span>
        </span>
        <span>Plan ahead. Switch off.</span>
      </footer>
      <dialog
        aria-labelledby="export-title"
        ref={modal}
        onCancel={(event) => {
          if (importing) event.preventDefault();
        }}
      >
        <div className="modal-heading">
          <span className="section-number">TAKE YOUR SCHEDULE WITH YOU</span>
          <button
            className="icon-button"
            aria-label="Close calendar export"
            disabled={importing}
            onClick={() => modal.current?.close()}
          >
            <Icon name="close" />
          </button>
        </div>
        <h2 id="export-title">Make it official.</h2>
        <p>
          Add {events.length} daily shifts through December {schedule?.years[1]}{" "}
          to your calendar.
        </p>
        <div className="event-preview">
          <span aria-hidden="true">🔥</span>
          <div>
            <strong>{next?.summary.replace("🔥 ", "")}</strong>
            <p>4:00 – 6:00 AM · {applied.timezone}</p>
          </div>
        </div>
        {clientId ? (
          <>
            <p>
              Sign in to add each shift to your primary Google calendar.
              Repeating the import skips events already added.
            </p>
            <button
              className="button dark"
              disabled={importing || !googleReady}
              onClick={() => void connectGoogle()}
            >
              <span className="google-g">G</span>
              {importing
                ? "Adding your shifts…"
                : "Connect Google & add shifts"}
            </button>
          </>
        ) : (
          <>
            <p>
              Download your calendar file, then choose it in Google Calendar’s{" "}
              <strong>Settings → Import & export</strong>. Each duty day is a
              separate event.
            </p>
            <div className="import-actions">
              <button className="button dark" onClick={download}>
                <Icon name="download" />
                Download calendar file
              </button>
              <a
                className="button outline"
                href="https://calendar.google.com/calendar/u/0/r/settings/export"
                target="_blank"
                rel="noreferrer"
              >
                Open Google Calendar
                <Icon name="arrow" />
              </a>
            </div>
            <p className="form-note">
              Direct sign-in is available when the app owner configures Google
              Calendar.
            </p>
          </>
        )}
        {notice && (
          <p className="import-status" role="status">
            {notice}
          </p>
        )}
      </dialog>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
