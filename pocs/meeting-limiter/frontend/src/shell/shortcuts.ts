import type { IconName } from "./icons";
import { TABS } from "./tabs";

export type Combo = {
  readonly mod?: boolean;
  readonly shift?: boolean;
  readonly alt?: boolean;
  readonly code: string;
  readonly label: string;
};

export type GroupId = "navigation" | "calendar" | "view" | "panels";

export type Group = {
  readonly id: GroupId;
  readonly title: string;
  readonly color: string;
  readonly icon: IconName;
};

export type Shortcut = {
  readonly id: string;
  readonly label: string;
  readonly group: GroupId;
  readonly icon: IconName;
  readonly combo: Combo;
};

export const GROUPS: readonly Group[] = [
  { id: "navigation", title: "Navigation", color: "#0f6cbd", icon: "search" },
  { id: "calendar", title: "Calendar", color: "#107c41", icon: "calendar" },
  { id: "view", title: "View", color: "#8a3ffc", icon: "window" },
  { id: "panels", title: "Panels and modals", color: "#c75b12", icon: "expand" },
];

const tabShortcuts: readonly Shortcut[] = TABS.map((t) => ({
  id: `tab-${t.id}`,
  label: `Go to ${t.label}`,
  group: "navigation",
  icon: t.icon,
  combo: { mod: true, code: `Digit${t.digit}`, label: t.digit },
}));

export const SHORTCUTS: readonly Shortcut[] = [
  { id: "search", label: "Search anything", group: "navigation", icon: "search", combo: { mod: true, code: "KeyK", label: "K" } },
  { id: "shortcuts", label: "Show shortcuts", group: "navigation", icon: "keyboard", combo: { mod: true, code: "Slash", label: "/" } },
  ...tabShortcuts,
  { id: "new-meeting", label: "New meeting", group: "calendar", icon: "plus", combo: { alt: true, code: "KeyN", label: "N" } },
  { id: "prev-week", label: "Previous week", group: "calendar", icon: "chevronLeft", combo: { alt: true, code: "ArrowLeft", label: "←" } },
  { id: "next-week", label: "Next week", group: "calendar", icon: "chevronRight", combo: { alt: true, code: "ArrowRight", label: "→" } },
  { id: "this-week", label: "This week", group: "calendar", icon: "today", combo: { alt: true, code: "KeyT", label: "T" } },
  { id: "zoom-in", label: "Zoom in", group: "view", icon: "zoomIn", combo: { mod: true, code: "Equal", label: "+" } },
  { id: "zoom-out", label: "Zoom out", group: "view", icon: "zoomOut", combo: { mod: true, code: "Minus", label: "-" } },
  { id: "zoom-reset", label: "Actual size", group: "view", icon: "zoomReset", combo: { mod: true, code: "Digit0", label: "0" } },
  { id: "fullscreen", label: "Toggle full screen", group: "view", icon: "fullscreen", combo: { mod: true, shift: true, code: "Enter", label: "Enter" } },
  { id: "screenshot", label: "Screenshot the window", group: "view", icon: "camera", combo: { mod: true, code: "KeyP", label: "P" } },
  { id: "close", label: "Collapse panel or close modal", group: "panels", icon: "escape", combo: { code: "Escape", label: "Esc" } },
];

export function keysOf(combo: Combo): readonly string[] {
  return [combo.mod ? "⌘" : "", combo.alt ? "⌥" : "", combo.shift ? "⇧" : "", combo.label].filter((k) => k !== "");
}

export function shortcutById(id: string): Shortcut | undefined {
  return SHORTCUTS.find((s) => s.id === id);
}

export function keysFor(id: string): string {
  const s = shortcutById(id);
  return s ? keysOf(s.combo).join(" ") : "";
}

export type KeyLike = {
  readonly metaKey: boolean;
  readonly ctrlKey: boolean;
  readonly shiftKey: boolean;
  readonly altKey: boolean;
  readonly code: string;
};

export function matches(e: KeyLike, combo: Combo): boolean {
  return (
    e.code === combo.code &&
    (e.metaKey || e.ctrlKey) === Boolean(combo.mod) &&
    e.shiftKey === Boolean(combo.shift) &&
    e.altKey === Boolean(combo.alt)
  );
}

export type FilteredGroup = { readonly group: Group; readonly shortcuts: readonly Shortcut[] };

export function filterShortcuts(query: string): readonly FilteredGroup[] {
  const q = query.trim().toLowerCase();
  return GROUPS.map((group) => {
    const all = SHORTCUTS.filter((s) => s.group === group.id);
    if (q === "" || group.title.toLowerCase().includes(q)) return { group, shortcuts: all };
    const hits = all.filter((s) => s.label.toLowerCase().includes(q) || keysOf(s.combo).join(" ").toLowerCase().includes(q));
    return { group, shortcuts: hits };
  }).filter((g) => g.shortcuts.length > 0);
}
