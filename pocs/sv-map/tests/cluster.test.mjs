import test from "node:test";
import assert from "node:assert/strict";
import { clusterPoints } from "../web/cluster.js";

test("logos closer than the radius merge into one bubble so SoMa stays readable", () => {
  const groups = clusterPoints([{ x: 10, y: 10 }, { x: 20, y: 30 }, { x: 500, y: 500 }], 48);
  assert.equal(groups.length, 2);
  const big = groups.find(g => g.members.length === 2);
  assert.deepEqual([big.x, big.y], [15, 20]);
});

test("points just across any grid line still merge, so bubbles never sit on top of each other", () => {
  assert.equal(clusterPoints([{ x: 47, y: 47 }, { x: 49, y: 49 }], 48).length, 1);
});

test("points farther apart than the radius stay separate", () => {
  assert.equal(clusterPoints([{ x: 0, y: 0 }, { x: 60, y: 0 }], 48).length, 2);
});

test("a tiny radius keeps every company separate at street zoom", () => {
  assert.equal(clusterPoints([{ x: 10, y: 10 }, { x: 12, y: 10 }, { x: 14, y: 10 }], 1).length, 3);
});

test("no points gives no clusters", () => {
  assert.deepEqual(clusterPoints([], 48), []);
});
