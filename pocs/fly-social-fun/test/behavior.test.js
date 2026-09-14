import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRandom } from '../server/random.js';
import { chooseDestination, resolveSwat, actionWeights } from '../server/behavior.js';
import { SPOTS } from '../server/world.js';

const FOOD = new Set(SPOTS.filter((spot) => spot.kind === 'food').map((spot) => spot.id));

function fakeFly(overrides = {}) {
  return {
    spot: 'mug',
    hunger: 0,
    traits: { chatty: 0.5, social: 0.5, reckless: 0.5, romantic: 0.5 },
    following: new Set(),
    ...overrides,
  };
}

function destinations(fly, snacks, draws = 2000) {
  const rng = createRandom(42);
  const counts = {};
  for (let i = 0; i < draws; i++) {
    const spot = chooseDestination(fly, snacks, rng);
    counts[spot] = (counts[spot] || 0) + 1;
  }
  return counts;
}

const share = (counts, predicate) => {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const hits = Object.entries(counts).filter(([spot]) => predicate(spot)).reduce((a, [, n]) => a + n, 0);
  return hits / total;
};

test('starving flies go looking for food instead of sightseeing', () => {
  const starving = share(destinations(fakeFly({ hunger: 0.95 }), {}), (spot) => FOOD.has(spot));
  const full = share(destinations(fakeFly({ hunger: 0 }), {}), (spot) => FOOD.has(spot));
  assert.ok(starving > 0.9, `starving flies picked food ${starving}`);
  assert.ok(full < 0.75, `full flies should still wander, picked food ${full}`);
});

test('a dropped snack pulls hungry flies to that spot', () => {
  const hungry = fakeFly({ hunger: 0.8 });
  const without = share(destinations(hungry, {}), (spot) => spot === 'pizza');
  const withSnack = share(destinations(hungry, { pizza: 10 }), (spot) => spot === 'pizza');
  assert.ok(withSnack > 0.5, `snack spot share ${withSnack}`);
  assert.ok(withSnack > without * 2, 'the snack must clearly beat the normal pull of the spot');
});

test('careful flies never fly into the spider web', () => {
  const careful = fakeFly({ hunger: 0.2, traits: { chatty: 0.5, social: 0.5, reckless: 0, romantic: 0.5 } });
  const counts = destinations(careful, {});
  assert.equal(counts.web, undefined);
});

test('a fly never picks the spot it is already on', () => {
  const counts = destinations(fakeFly({ spot: 'banana', hunger: 1 }), {});
  assert.equal(counts.banana, undefined);
});

test('a fly can only follow flies it has met at its own spot', () => {
  const lonely = actionWeights(fakeFly(), { neighbors: [], recentBuzzes: [], atFood: false });
  const follow = lonely.find((entry) => entry.value === 'follow');
  const flirt = lonely.find((entry) => entry.value === 'flirt');
  assert.equal(follow.weight, 0);
  assert.equal(flirt.weight, 0);
});

test('a swat only splits the flies it was aimed at into killed and survivors', () => {
  const targets = Array.from({ length: 40 }, (_, i) => ({ id: `fly-${i}` }));
  const { killed, survivors } = resolveSwat(targets, createRandom(7));
  assert.equal(killed.length + survivors.length, targets.length);
  assert.ok(killed.length > 0 && survivors.length > 0, 'a swatter is dangerous but flies can dodge');
});
