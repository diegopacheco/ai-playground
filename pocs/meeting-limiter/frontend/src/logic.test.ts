import { describe, expect, test } from "bun:test";
import { formatMinutes, formatMoney } from "./money";
import { addDays, combine, fromDateKey, startOfWeek, toDateKey } from "./time";
import { filterShortcuts, keysOf, matches, SHORTCUTS } from "./shell/shortcuts";
import { TABS } from "./shell/tabs";
import { dockSide, hidePanel, parseHidden, restorePanel } from "./shell/dock";
import { layoutDay } from "./features/meetings/layout";
import { toRequest } from "./features/meetings/request";
import { walletHealth } from "./features/wallets/types";
import { periodFor } from "./features/rankings/range";
import { filterCommands, type Command } from "./shell/CommandPalette";

describe("money", () => {
  test("cents render as dollars so wallet math never shows float noise", () => {
    expect(formatMoney(50000)).toBe("$500.00");
    expect(formatMoney(2500)).toBe("$25.00");
    expect(formatMoney(-10000)).toBe("-$100.00");
  });

  test("minutes render as hours the way the weekly caps are stated", () => {
    expect(formatMinutes(300)).toBe("5h");
    expect(formatMinutes(90)).toBe("1h 30m");
    expect(formatMinutes(15)).toBe("15m");
  });
});

describe("weeks", () => {
  test("a week starts on monday, matching the backend refill boundary", () => {
    expect(toDateKey(startOfWeek(new Date(2026, 10, 22, 23, 0)))).toBe("2026-11-16");
    expect(toDateKey(startOfWeek(new Date(2026, 10, 23, 0, 0)))).toBe("2026-11-23");
  });

  test("invalid week keys in the url fall back instead of crashing", () => {
    expect(fromDateKey("2026-02-31")).toBeUndefined();
    expect(fromDateKey("garbage")).toBeUndefined();
    expect(toDateKey(addDays(fromDateKey("2026-11-16") ?? new Date(0), 7))).toBe("2026-11-23");
  });

  test("date and time inputs combine into one local start", () => {
    expect(combine("2026-11-16", "09:30")?.getHours()).toBe(9);
    expect(combine("2026-11-16", "9:30")).toBeUndefined();
  });
});

describe("meeting request", () => {
  test("the organizer is never charged twice by also being an attendee", () => {
    const r = toRequest({ title: " Sync ", organizerId: "ana", attendeeIds: ["ana", "bruno"], date: "2026-11-16", time: "10:00", minutes: 90 });
    expect(r?.attendeeIds).toEqual(["bruno"]);
    expect(r?.title).toBe("Sync");
    expect(((r?.end.getTime() ?? 0) - (r?.start.getTime() ?? 0)) / 60000).toBe(90);
  });

  test("an incomplete draft produces no quote request", () => {
    expect(toRequest({ title: "", organizerId: "", attendeeIds: [], date: "2026-11-16", time: "10:00", minutes: 60 })).toBeUndefined();
  });
});

describe("wallet health", () => {
  test("an empty wallet blocks meetings, a low wallet only warns", () => {
    expect(walletHealth({ budget: 50000, balance: 0 })).toBe("empty");
    expect(walletHealth({ budget: 50000, balance: 10000 })).toBe("low");
    expect(walletHealth({ budget: 50000, balance: 40000 })).toBe("healthy");
  });
});

describe("rankings period", () => {
  test("month covers the whole month of the selected week", () => {
    const p = periodFor("month", new Date(2026, 10, 16));
    expect(toDateKey(p.from)).toBe("2026-11-01");
    expect(toDateKey(p.to)).toBe("2026-12-01");
  });

  test("week covers exactly seven days so it matches one wallet cycle", () => {
    const p = periodFor("week", new Date(2026, 10, 16));
    expect(toDateKey(p.to)).toBe("2026-11-23");
  });
});

describe("week layout", () => {
  const at = (h: number) => new Date(2026, 10, 16, h);
  test("overlapping meetings split the column so none is hidden", () => {
    const placed = layoutDay([
      { id: "a", start: at(9), end: at(11) },
      { id: "b", start: at(10), end: at(12) },
      { id: "c", start: at(13), end: at(14) },
    ]);
    const byId = new Map(placed.map((p) => [p.item.id, p]));
    expect(byId.get("a")?.lanes).toBe(2);
    expect(byId.get("b")?.lane).toBe(1);
    expect(byId.get("c")?.lanes).toBe(1);
  });
});

describe("shortcuts", () => {
  test("every tab is reachable by a cmd digit and the palette list", () => {
    for (const t of TABS) {
      const s = SHORTCUTS.find((x) => x.id === `tab-${t.id}`);
      expect(s?.icon).toBe(t.icon);
      expect(keysOf(s?.combo ?? { code: "", label: "" })).toEqual(["⌘", t.digit]);
    }
  });

  test("no two shortcuts share a key combo", () => {
    const combos = SHORTCUTS.map((s) => keysOf(s.combo).join("+"));
    expect(new Set(combos).size).toBe(combos.length);
  });

  test("matching needs the exact modifiers so typing letters never fires actions", () => {
    const search = SHORTCUTS.find((s) => s.id === "search")?.combo ?? { code: "", label: "" };
    expect(matches({ metaKey: true, ctrlKey: false, shiftKey: false, altKey: false, code: "KeyK" }, search)).toBe(true);
    expect(matches({ metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, code: "KeyK" }, search)).toBe(false);
  });

  test("filter keeps a whole group when its title matches, otherwise only matching rows", () => {
    const view = filterShortcuts("view");
    expect(view).toHaveLength(1);
    expect(view[0]?.shortcuts.length).toBe(SHORTCUTS.filter((s) => s.group === "view").length);
    const zoom = filterShortcuts("zoom out");
    expect(zoom.flatMap((g) => g.shortcuts.map((s) => s.id))).toEqual(["zoom-out"]);
    expect(filterShortcuts("nothing like this")).toHaveLength(0);
  });
});

describe("command palette", () => {
  test("search matches the label and the hint", () => {
    const cmds: Command[] = [
      { id: "a", label: "Wallets", hint: "Tab", icon: "wallet", run: () => undefined },
      { id: "b", label: "Act as Ana", hint: "Engineer", icon: "user", run: () => undefined },
    ];
    expect(filterCommands(cmds, "engineer").map((c) => c.id)).toEqual(["b"]);
    expect(filterCommands(cmds, "")).toHaveLength(2);
  });
});

describe("dock", () => {
  test("a panel docks to the side its center is on", () => {
    expect(dockSide({ left: 0, width: 300 }, 1200)).toBe("left");
    expect(dockSide({ left: 700, width: 300 }, 1200)).toBe("right");
  });

  test("hidden panels stack in hide order and restore removes only one", () => {
    let list = hidePanel([], { id: "a", title: "A", side: "left" });
    list = hidePanel(list, { id: "b", title: "B", side: "left" });
    expect(list.map((p) => p.id)).toEqual(["a", "b"]);
    expect(restorePanel(list, "a").map((p) => p.id)).toEqual(["b"]);
  });

  test("corrupt saved state is ignored instead of breaking the layout", () => {
    expect(parseHidden([{ id: "a", title: "A", side: "up" }, 3, { id: "b", title: "B", side: "right" }])).toEqual([{ id: "b", title: "B", side: "right" }]);
    expect(parseHidden("x")).toBeUndefined();
  });
});
