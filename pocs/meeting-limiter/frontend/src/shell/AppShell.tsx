import { Link, Outlet, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { MeetingDialogs, useMeetingDialogs } from "../features/meetings/components/dialogs";
import { useWeekMeetings } from "../features/meetings/hooks";
import { useUsers } from "../features/people/hooks";
import { roleLabel } from "../features/people/types";
import { formatMoney } from "../money";
import { formatWhen } from "../time";
import { ActingUserProvider, useActingUser } from "./acting";
import { CommandPalette, type Command } from "./CommandPalette";
import { desktop, isDesktop } from "./desktop";
import { Icon } from "./icons";
import { PanelProvider } from "./panels";
import { matches, SHORTCUTS } from "./shortcuts";
import { ShortcutsModal } from "./ShortcutsModal";
import { TABS } from "./tabs";
import { Tooltip } from "./Tooltip";
import { useWeek } from "./week";

type Overlay = "none" | "palette" | "shortcuts";

export function AppShell() {
  return (
    <ActingUserProvider>
      <PanelProvider>
        <MeetingDialogs>
          <Shell />
        </MeetingDialogs>
      </PanelProvider>
    </ActingUserProvider>
  );
}

function toggleWebFullscreen(): void {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen();
}

function useToast(): string | undefined {
  const [toast, setToast] = useState<string | undefined>(undefined);
  useEffect(() => {
    desktop()?.onToast((text) => {
      setToast(text);
      window.setTimeout(() => setToast(undefined), 3500);
    });
  }, []);
  return toast;
}

function Shell() {
  const [overlay, setOverlay] = useState<Overlay>("none");
  const navigate = useNavigate();
  const week = useWeek();
  const dialogs = useMeetingDialogs();
  const users = useUsers();
  const meetings = useWeekMeetings(week.weekStart);
  const toast = useToast();
  const { userId, setUserId } = useActingUser();

  const actions: Readonly<Record<string, () => void>> = {
    search: () => setOverlay(overlay === "palette" ? "none" : "palette"),
    shortcuts: () => setOverlay(overlay === "shortcuts" ? "none" : "shortcuts"),
    "new-meeting": () => dialogs.openNew(),
    "prev-week": week.prev,
    "next-week": week.next,
    "this-week": week.thisWeek,
    ...Object.fromEntries(TABS.map((t) => [`tab-${t.id}`, () => void navigate({ to: t.path, search: (prev) => prev })])),
    ...(isDesktop() ? {} : { fullscreen: toggleWebFullscreen }),
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const hit = SHORTCUTS.find((s) => matches(e, s.combo) && actions[s.id]);
      if (!hit) return;
      e.preventDefault();
      actions[hit.id]?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const commands: readonly Command[] = [
    ...TABS.map((t) => ({ id: `tab-${t.id}`, label: t.label, hint: "Tab", icon: t.icon, shortcut: `tab-${t.id}`, run: actions[`tab-${t.id}`] ?? (() => undefined) })),
    { id: "new", label: "New meeting", hint: "Action", icon: "plus", shortcut: "new-meeting", run: () => dialogs.openNew() },
    { id: "prev", label: "Previous week", hint: "Action", icon: "chevronLeft", shortcut: "prev-week", run: week.prev },
    { id: "next", label: "Next week", hint: "Action", icon: "chevronRight", shortcut: "next-week", run: week.next },
    { id: "today", label: "This week", hint: "Action", icon: "today", shortcut: "this-week", run: week.thisWeek },
    { id: "keys", label: "Keyboard shortcuts", hint: "Help", icon: "keyboard", shortcut: "shortcuts", run: () => setOverlay("shortcuts") },
    ...(users.data ?? []).map(
      (u): Command => ({ id: `user-${u.id}`, label: `Act as ${u.name}`, hint: roleLabel(u.role), icon: "user", run: () => setUserId(u.id) }),
    ),
    ...(meetings.data ?? []).map(
      (m): Command => ({
        id: `meeting-${m.id}`,
        label: m.title,
        hint: `${formatWhen(m.start, m.end)} - ${formatMoney(m.totalCost)}`,
        icon: "calendar",
        run: () => dialogs.openDetails(m),
      }),
    ),
  ];

  return (
    <div className={`app ${isDesktop() ? "app--desktop" : ""}`}>
      <header
        className="titlebar"
        onDoubleClick={(e) => {
          if (e.target instanceof Element && e.target.closest("button, input, select, a")) return;
          desktop()?.toggleMaximize();
        }}
      >
        <img className="brand-logo" src="/logo.svg" alt="" />
        <span className="brand">Meeting Limiter</span>
        <button type="button" className="search-box" data-tip="Search" data-keys="search" onClick={() => setOverlay("palette")}>
          <Icon name="search" size={16} />
          <span>Search</span>
        </button>
        <label className="acting">
          <Icon name="user" size={16} />
          <span className="acting-label">Acting as</span>
          <select value={userId} onChange={(e) => setUserId(e.target.value)} aria-label="Acting as">
            {(users.data ?? []).map((u) => (
              <option key={u.id} value={u.id}>
                {u.name} ({roleLabel(u.role)})
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="icon-button on-dark" data-tip="Keyboard shortcuts" data-keys="shortcuts" aria-label="Keyboard shortcuts" onClick={() => setOverlay("shortcuts")}>
          <Icon name="keyboard" />
        </button>
      </header>
      <div className="body">
        <nav className="rail" aria-label="Tabs">
          {TABS.map((t) => (
            <Link
              key={t.id}
              to={t.path}
              search={(prev) => prev}
              className="rail-tab"
              activeProps={{ className: "rail-tab active" }}
              activeOptions={{ exact: true, includeSearch: false }}
              data-tip={t.label}
              data-keys={`tab-${t.id}`}
            >
              <Icon name={t.icon} size={22} />
              <span>{t.label}</span>
            </Link>
          ))}
        </nav>
        <main className="content">
          <Outlet />
        </main>
      </div>
      {overlay === "palette" && <CommandPalette commands={commands} onClose={() => setOverlay("none")} />}
      {overlay === "shortcuts" && <ShortcutsModal onClose={() => setOverlay("none")} />}
      <Tooltip />
      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}
