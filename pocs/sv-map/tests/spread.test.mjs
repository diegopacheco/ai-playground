import test from "node:test";
import assert from "node:assert/strict";
import { spreadPoints } from "../web/spread.js";

const overlaps = (placed, gap) => placed.some((a, i) => placed.slice(i + 1).some(b => Math.hypot(a.x - b.x, a.y - b.y) < gap - 0.5));

test("crowded logos are pushed apart so every company's logo stays visible", () => {
  const crowd = Array.from({ length: 60 }, (_, i) => ({ id: i, x: 100 + (i % 3), y: 100 + (i % 5) }));
  const placed = spreadPoints(crowd, 30, 400);
  assert.equal(placed.length, 60);
  assert.equal(overlaps(placed, 30), false);
});

test("companies at the exact same address still separate", () => {
  const placed = spreadPoints([{ x: 50, y: 50 }, { x: 50, y: 50 }], 30);
  assert.ok(Math.hypot(placed[0].x - placed[1].x, placed[0].y - placed[1].y) >= 29.5);
});

test("logos that do not overlap stay exactly on their address", () => {
  const placed = spreadPoints([{ x: 0, y: 0 }, { x: 100, y: 0 }], 30);
  for (const p of placed) {
    assert.ok(Math.abs(p.x - p.anchorX) < 0.1 && Math.abs(p.y - p.anchorY) < 0.1);
  }
});

test("each placed logo remembers its true address for the leader line", () => {
  const [p] = spreadPoints([{ x: 7, y: 9, id: "a" }], 30);
  assert.deepEqual([p.anchorX, p.anchorY, p.id], [7, 9, "a"]);
});
