import { expect, test } from "bun:test";
import { FLY_BUZZ, FLY_CALM, RANGER_CALM, ZORD_ATTACK, ZORD_GUARD, ZORD_IDLE, ZORD_PALETTES, skyline } from "../src/index.js";
import type { Sprite } from "../src/index.js";

const sprites: [string, Sprite][] = [
  ["idle", ZORD_IDLE], ["attack", ZORD_ATTACK], ["guard", ZORD_GUARD],
  ["fly calm", FLY_CALM], ["fly buzz", FLY_BUZZ], ["ranger", RANGER_CALM]
];

test("every sprite is a rectangle because the renderer walks it row by row", () => {
  for (const [name, art] of sprites) {
    expect(art.height).toBeGreaterThan(0);
    for (const row of art.rows) expect(`${name}:${row.length}`).toBe(`${name}:${art.width}`);
  }
});

test("the three zord poses share one frame size so a pose swap never shifts the mech", () => {
  expect([ZORD_ATTACK.width, ZORD_ATTACK.height]).toEqual([ZORD_IDLE.width, ZORD_IDLE.height]);
  expect([ZORD_GUARD.width, ZORD_GUARD.height]).toEqual([ZORD_IDLE.width, ZORD_IDLE.height]);
});

test("both zord palettes colour every key the poses use so no pixel renders blank", () => {
  const keys = new Set([...ZORD_IDLE.rows, ...ZORD_ATTACK.rows, ...ZORD_GUARD.rows].join("").split("").filter(key => key !== "."));
  for (const palette of Object.values(ZORD_PALETTES)) for (const key of keys) expect(`${key}:${typeof palette[key]}`).toBe(`${key}:string`);
});

test("the two flies wear different colours so the cockpit cams are never confused", () => {
  expect(ZORD_PALETTES.crimson?.B).not.toBe(ZORD_PALETTES.cobalt?.B);
});

test("the skyline fills the canvas width and stays behind the fighters", () => {
  const buildings = skyline(3, 320, 10);
  expect(buildings).toHaveLength(10);
  expect(buildings.at(-1)!.x).toBeLessThan(320);
  for (const building of buildings) {
    expect(building.width).toBeGreaterThan(0);
    expect(building.height).toBeLessThanOrEqual(72);
  }
});

test("one seed draws one city so a replay looks like the same place", () => {
  expect(skyline(5, 320, 8)).toEqual(skyline(5, 320, 8));
  expect(skyline(5, 320, 8)).not.toEqual(skyline(6, 320, 8));
});
