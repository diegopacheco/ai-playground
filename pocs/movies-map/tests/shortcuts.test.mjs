import test from "node:test"
import assert from "node:assert/strict"
import fs from "node:fs"
import vm from "node:vm"

const context = { module: { exports: {} } }
vm.runInNewContext(fs.readFileSync(new URL("../web/shortcuts.js", import.meta.url), "utf8"), context)
const { SHORTCUT_GROUPS, filterShortcutGroups } = context.module.exports

test("matching a group title keeps the whole group", () => {
  const edit = SHORTCUT_GROUPS.find(g => g.title === "Edit")
  assert.equal(filterShortcutGroups(SHORTCUT_GROUPS, "edit")[0].items.length, edit.items.length)
})

test("matching a description keeps only that row", () => {
  const out = filterShortcutGroups(SHORTCUT_GROUPS, "full screen")
  assert.equal(out.length, 1)
  assert.equal(out[0].items.length, 1)
})

test("the required app shortcuts are all documented", () => {
  const keys = SHORTCUT_GROUPS.flatMap(g => g.items.map(i => i[0]))
  for (const k of ["⌘ K", "⌘ /", "⌘ +", "⌘ -", "⌘ 1", "⌘ P", "⌘ ⇧ Enter", "⌘ C", "⌘ V", "⌘ X"]) assert.ok(keys.includes(k), k)
})

test("an unknown search returns nothing", () => {
  assert.equal(filterShortcutGroups(SHORTCUT_GROUPS, "zzzz").length, 0)
})
