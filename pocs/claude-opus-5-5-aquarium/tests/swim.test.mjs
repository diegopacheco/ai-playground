import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FISH, TANK } from '../web/catalog.mjs';
import { rng, swimBox, createSwimmer, step, heading, patrol, nearestFood, canEat, sinkFood, FLEE_RADIUS, SINK_SPEED } from '../web/swim.mjs';

const inside = (p, b) => p.x >= b.minX && p.x <= b.maxX && p.y >= b.minY && p.y <= b.maxY && p.z >= b.minZ && p.z <= b.maxZ;

test('fish never leave the water, even after minutes of swimming', () => {
  const random = rng(7);
  const box = swimBox();
  for (const f of FISH) {
    const s = createSwimmer(random, box, f.band, f.speed);
    for (let i = 0; i < 60 * 120; i++) {
      step(s, 1 / 60, random, box, null);
      assert.ok(inside(s.pos, box), `${f.id} escaped at step ${i}`);
    }
  }
  assert.ok(box.maxY < TANK.water);
});

test('fish flee from a nearby shark and speed up', () => {
  const random = rng(3);
  const box = swimBox();
  const s = createSwimmer(random, box, [0.4, 0.6], 0.08);
  s.pos = { x: 0, y: 0.3, z: 0 };
  s.vel = { x: 0.05, y: 0, z: 0 };
  s.target = { x: 0.4, y: 0.3, z: 0 };
  const shark = { x: 0.08, y: 0.3, z: 0 };
  for (let i = 0; i < 30; i++) step(s, 1 / 60, random, box, shark);
  assert.ok(s.vel.x < 0, 'the fish should turn away from the shark, even against its target');
  assert.ok(Math.hypot(s.vel.x, s.vel.y, s.vel.z) > 0.08, 'fleeing fish swim faster than cruising');
});

test('a shark outside the flee radius does not scare anyone', () => {
  const a = createSwimmer(rng(9), swimBox(), [0.4, 0.6], 0.08);
  const b = structuredClone(a);
  const far = { x: a.pos.x + FLEE_RADIUS * 2, y: a.pos.y, z: a.pos.z };
  step(a, 1 / 60, rng(1), swimBox(), far);
  step(b, 1 / 60, rng(1), swimBox(), null);
  assert.deepEqual(a.pos, b.pos);
});

test('fish settle on the depth band of their species', () => {
  const random = rng(11);
  const box = swimBox();
  const mandarin = FISH.find(f => f.id === 'mandarin');
  const s = createSwimmer(random, box, mandarin.band, mandarin.speed);
  let sum = 0;
  const n = 60 * 90;
  for (let i = 0; i < n; i++) {
    step(s, 1 / 60, random, box, null);
    sum += (s.pos.y - box.minY) / (box.maxY - box.minY);
  }
  assert.ok(sum / n < 0.45, 'a bottom dweller should mostly stay low');
});

test('the heading faces the direction of travel', () => {
  assert.ok(Math.abs(heading({ x: 1, y: 0, z: 0 }).yaw) < 1e-9);
  assert.ok(Math.abs(heading({ x: 0, y: 0, z: -1 }).yaw - Math.PI / 2) < 1e-9);
  assert.ok(heading({ x: 1, y: 5, z: 0 }).pitch <= 0.5);
});

test('the shark patrols inside its box and faces where it goes', () => {
  const box = swimBox(0.13);
  for (let t = 0; t < 200; t += 0.1) {
    const p = patrol(t, box);
    assert.ok(inside(p.pos, box), `shark left the box at t=${t}`);
    const next = patrol(t + 0.01, box).pos;
    const dx = next.x - p.pos.x;
    const dz = next.z - p.pos.z;
    assert.ok(dx * p.vel.x + dz * p.vel.z > 0, 'velocity must point along the path');
  }
});

test('a hungry fish swims to the food and gets close enough to eat it', () => {
  const random = rng(5);
  const box = swimBox();
  const s = createSwimmer(random, box, [0.4, 0.6], 0.08);
  s.pos = { x: -0.3, y: 0.3, z: -0.1 };
  const food = { x: 0.2, y: 0.35, z: 0.1 };
  let ate = false;
  for (let i = 0; i < 60 * 15 && !ate; i++) {
    step(s, 1 / 60, random, box, null, food);
    ate = canEat(s.pos, food);
  }
  assert.ok(ate, 'the fish never reached the food');
});

test('fish go for the closest flake and ignore food out of reach', () => {
  const pos = { x: 0, y: 0.3, z: 0 };
  const near = { x: 0.05, y: 0.3, z: 0 };
  const far = { x: 0.3, y: 0.3, z: 0 };
  assert.equal(nearestFood(pos, [far, near]), near);
  assert.equal(nearestFood(pos, [far], 0.1), null);
  assert.equal(nearestFood(pos, []), null);
});

test('the shark scares fish away even from food', () => {
  const random = rng(8);
  const box = swimBox();
  const s = createSwimmer(random, box, [0.4, 0.6], 0.08);
  s.pos = { x: 0, y: 0.3, z: 0 };
  s.vel = { x: 0, y: 0, z: 0 };
  const food = { x: 0.1, y: 0.3, z: 0 };
  const shark = { x: 0.03, y: 0.3, z: 0 };
  for (let i = 0; i < 30; i++) step(s, 1 / 60, random, box, shark, food);
  assert.ok(s.vel.x < 0);
});

test('flakes sink slowly, stop on the sand and start counting their time there', () => {
  const f = { x: 0, y: 0.5, z: 0, phase: 0, rest: 0 };
  sinkFood(f, 1, 0.05);
  assert.ok(Math.abs(f.y - (0.5 - SINK_SPEED)) < 1e-9);
  for (let i = 0; i < 100; i++) sinkFood(f, 1, 0.05);
  assert.equal(f.y, 0.05);
  assert.ok(f.rest > 0);
});
