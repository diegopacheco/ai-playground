import { describe, expect, test } from "bun:test";
import { MAX_FLIGHT, MIN_FLIGHT, type Shot, flightShot, flightTime } from "./flight";

const ground: Shot = { pos: { x: 0, y: 8, z: 9 }, look: { x: 0, y: 1, z: 0 } };
const sky: Shot = { pos: { x: 42, y: 42, z: 9 }, look: { x: 42, y: 35, z: 0 } };
const nextDoor: Shot = { pos: { x: 10, y: 8, z: 9 }, look: { x: 10, y: 1, z: 0 } };
const steps = Array.from({ length: 101 }, (_, i) => i / 100);

describe("camera flight between stations", () => {
  test("the flight starts exactly where the camera was, so there is no jump when a page is picked", () => {
    expect(flightShot(ground, sky, 0)).toEqual(ground);
  });

  test("the flight lands exactly on the next station, so the pop-up is framed the same way every time", () => {
    const end = flightShot(ground, sky, 1);
    expect(end.pos.x).toBeCloseTo(sky.pos.x, 6);
    expect(end.pos.y).toBeCloseTo(sky.pos.y, 6);
    expect(end.look.y).toBeCloseTo(sky.look.y, 6);
  });

  test("the camera arcs up between stations and never dips under the lower one, so it never flies through the ground", () => {
    for (const p of steps) expect(flightShot(nextDoor, ground, p).pos.y).toBeGreaterThanOrEqual(ground.pos.y - 1e-9);
    expect(flightShot(nextDoor, ground, 0.5).pos.y).toBeGreaterThan(ground.pos.y);
  });

  test("the camera pulls back on the way, so it never brushes past the scenery between stations", () => {
    for (const p of steps) expect(flightShot(ground, sky, p).pos.z).toBeGreaterThanOrEqual(ground.pos.z - 1e-9);
    expect(flightShot(ground, sky, 0.5).pos.z).toBeGreaterThan(ground.pos.z + 1);
  });

  test("the camera always moves forward along the way, it never backs up mid flight", () => {
    let last = -Infinity;
    for (const p of steps) {
      const x = flightShot(ground, sky, p).pos.x;
      expect(x).toBeGreaterThanOrEqual(last);
      last = x;
    }
  });

  test("a long trip into the sky takes longer than a hop next door, but never too short or too long", () => {
    expect(flightTime(ground, sky)).toBeGreaterThan(flightTime(ground, nextDoor));
    expect(flightTime(ground, ground)).toBe(MIN_FLIGHT);
    expect(flightTime(ground, { pos: { x: 900, y: 0, z: 0 }, look: ground.look })).toBe(MAX_FLIGHT);
  });
});
