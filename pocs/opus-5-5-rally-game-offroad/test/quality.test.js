import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGovernor, tickGovernor, LEVELS, TARGET_FPS } from '../public/js/core/quality.js';

function run(gov, fpsOf, seconds) {
  const history = [];
  for (let s = 0; s < seconds; s++) {
    const fps = fpsOf(gov.level);
    for (let f = 0; f < fps; f++) tickGovernor(gov, 1 / fps);
    history.push({ level: gov.level, fps: gov.fps });
  }
  return history;
}

test('a slow machine drops quality until the game holds at least 30 FPS', () => {
  const costPerLevel = [22, 27, 33, 41, 55];
  const gov = createGovernor(0);
  const history = run(gov, (l) => costPerLevel[l], 12);
  assert.ok(history.at(-1).fps >= TARGET_FPS, `settled at ${history.at(-1).fps} fps`);
  assert.ok(gov.level >= 2);
});

test('a fast machine climbs back to Ultra quality', () => {
  const gov = createGovernor(3);
  run(gov, () => 120, 30);
  assert.equal(gov.level, 0);
  assert.equal(LEVELS[gov.level].name, 'Ultra');
});

test('quality does not flicker between two levels when one is just too slow', () => {
  const gov = createGovernor(1);
  const history = run(gov, (l) => (l === 0 ? 30 : 62), 60);
  const changes = history.filter((h, i) => i > 0 && h.level !== history[i - 1].level).length;
  assert.ok(changes <= 4, `${changes} level changes in 60s`);
});
