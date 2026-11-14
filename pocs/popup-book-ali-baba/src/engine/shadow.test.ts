import { describe, expect, test } from "bun:test";
import { lampFrustum, penumbra, projectToScreen, SCREEN_H, SCREEN_W, shadowScale } from "./shadow";

const lamp = { x: 0, y: 2.2, z: -4.2 };

describe("shadow projection", () => {
  test("a puppet pressed against the screen casts a shadow of its own size and place", () => {
    expect(shadowScale(4.2, 0)).toBe(1);
    const p = projectToScreen({ x: 1.3, y: 0.7, z: 0 }, lamp);
    expect(p.x).toBeCloseTo(1.3);
    expect(p.y).toBeCloseTo(0.7);
  });

  test("moving the lamp closer to a puppet makes its shadow grow, which is how the captain looms", () => {
    expect(shadowScale(2, 1)).toBeGreaterThan(shadowScale(4.2, 1));
    expect(shadowScale(4.2, 2.8)).toBeGreaterThan(2.5);
  });

  test("a puppet deeper behind the screen has a softer edge than one touching it", () => {
    expect(penumbra(4.2, 1.5, 0.1)).toBeGreaterThan(penumbra(4.2, 0.1, 0.1));
    expect(penumbra(4.2, 0, 0.1)).toBe(0);
  });

  test("the lamp camera frames exactly the screen so the mask lines up with the paper", () => {
    const near = 0.05;
    const f = lampFrustum({ x: 0, y: SCREEN_H / 2, z: -4 }, near);
    expect(f.right - f.left).toBeCloseTo((SCREEN_W * near) / 4);
    expect(f.top - f.bottom).toBeCloseTo((SCREEN_H * near) / 4);
    expect(f.left).toBeCloseTo(-f.right);
  });
});
