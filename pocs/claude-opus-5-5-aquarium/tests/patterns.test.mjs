import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TANK, FISH } from '../web/catalog.mjs';
import { PATTERNS, shade, fishColor, sandHeight, noise } from '../web/patterns.mjs';

function stats(pattern) {
  let min = 1;
  let max = 0;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const v = shade(pattern, x / 64, y / 64);
      assert.ok(v >= 0 && v <= 1, `${pattern} out of range`);
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  return { min, max };
}

test('every material texture has visible detail, not a flat color', () => {
  for (const p of PATTERNS) {
    const { min, max } = stats(p);
    assert.ok(max - min > 0.15, `${p} is too flat to read as a material`);
  }
});

test('bricks show dark mortar lines between bricks', () => {
  assert.ok(shade('bricks', 0.5, 0.001) < 0.35, 'mortar at the row edge');
  assert.ok(shade('bricks', 0.1, 0.05) > 0.35, 'brick face');
});

test('noise tiles so textures repeat without seams', () => {
  for (const [x, y] of [[0.3, 0.7], [5.5, 2.25]]) assert.ok(Math.abs(noise(x, y, 8) - noise(x + 8, y + 8, 8)) < 1e-9);
});

test('fish are painted with their species markings', () => {
  const clown = FISH.find(f => f.id === 'clown');
  assert.equal(fishColor(clown, 0.45, 0), '#ffffff');
  assert.equal(fishColor(clown, 0.2, 0), clown.colors[0]);
  const neon = FISH.find(f => f.id === 'neon');
  assert.equal(fishColor(neon, 0, 0.2), neon.colors[1]);
  assert.equal(fishColor(neon, -0.5, -0.5), neon.colors[2]);
  const tang = FISH.find(f => f.id === 'bluetang');
  assert.equal(fishColor(tang, -0.9, 0), tang.colors[2]);
  for (const f of FISH) assert.match(fishColor(f, 0.9, 0.9), /^#[0-9a-f]{6}$/);
});

test('the shark has a pale belly and a darker back', () => {
  assert.notEqual(fishColor({ pattern: 'shark', colors: [] }, 0, -0.8), fishColor({ pattern: 'shark', colors: [] }, 0, 0.8));
});

test('sand slopes up toward the back and stays well below the water line', () => {
  let front = 0;
  let back = 0;
  for (let x = -0.5; x <= 0.5; x += 0.05) {
    front += sandHeight(x, TANK.depth / 2);
    back += sandHeight(x, -TANK.depth / 2);
    assert.ok(sandHeight(x, 0) > 0 && sandHeight(x, 0) < 0.1);
  }
  assert.ok(back > front);
});
