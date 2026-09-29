import test from "node:test";
import assert from "node:assert/strict";
import { SHORTCUT_GROUPS, filterGroups } from "../web/help.js";

test("matching a group title keeps the whole group", () => {
  const [tabs] = filterGroups(SHORTCUT_GROUPS, "tabs");
  assert.equal(tabs.title, "Tabs");
  assert.equal(tabs.rows.length, SHORTCUT_GROUPS.find(g => g.title === "Tabs").rows.length);
});

test("matching a key or description keeps only those rows", () => {
  const groups = filterGroups(SHORTCUT_GROUPS, "screenshot");
  assert.deepEqual(groups.map(g => g.title), ["Capture"]);
  assert.equal(groups[0].rows.length, 1);
});

test("nothing matching returns no groups so the modal shows its empty message", () => {
  assert.deepEqual(filterGroups(SHORTCUT_GROUPS, "no such shortcut"), []);
});

test("the required app shortcuts are all documented", () => {
  const keys = SHORTCUT_GROUPS.flatMap(g => g.rows.map(r => r[0].join(" ")));
  for (const k of ["⌘ K", "⌘ /", "⌘ +", "⌘ -", "⌘ 1", "⌘ P", "⌘ ⇧ ↵", "⌘ C", "⌘ V", "⌘ X"]) assert.ok(keys.includes(k), k);
});
