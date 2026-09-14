import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createSimulation, SimulationError } from '../server/simulation.js';

function record(simulation) {
  const events = [];
  simulation.subscribe((event) => events.push(event));
  return events;
}

function run(simulation, ticks) {
  for (let i = 0; i < ticks; i++) simulation.tick();
}

test('the social graph is built from flies meeting at the same spot', () => {
  const simulation = createSimulation({ seed: 11, actChance: 0.6 });
  let checked = 0;
  simulation.subscribe((event) => {
    if (event.type !== 'follow') return;
    const flies = new Map(simulation.snapshot().flies.map((fly) => [fly.id, fly]));
    assert.equal(flies.get(event.flyId).spot, flies.get(event.targetId).spot);
    checked++;
  });
  run(simulation, 150);
  assert.ok(checked > 10, `expected many follows, saw ${checked}`);
});

test('dead flies never buzz, like or follow again', () => {
  const simulation = createSimulation({ seed: 5, actChance: 0.5 });
  const events = record(simulation);
  const dead = new Set();
  for (let i = 0; i < 400; i++) {
    simulation.tick();
    if (i % 25 === 0) ['banana', 'pizza', 'trash'].forEach(simulation.swat);
  }
  for (const event of events) {
    if (event.type === 'death') dead.add(event.flyId);
    if (event.type === 'buzz') assert.ok(!dead.has(event.buzz.flyId), `${event.buzz.handle} buzzed after dying`);
    if (event.type === 'like' || event.type === 'follow') assert.ok(!dead.has(event.flyId), 'a dead fly interacted');
  }
  assert.ok(dead.size > 5, `the swatter should have killed several flies, killed ${dead.size}`);
});

test('a swat kills only flies at the swatted spot and the survivors flee', () => {
  const simulation = createSimulation({ seed: 21 });
  run(simulation, 20);
  const before = simulation.snapshot().flies;
  const onPizza = new Set(before.filter((fly) => fly.spot === 'pizza').map((fly) => fly.id));
  const events = record(simulation);
  const result = simulation.swat('pizza');
  const deaths = events.filter((event) => event.type === 'death');
  for (const death of deaths) assert.ok(onPizza.has(death.flyId), 'a fly far from the swatter died');
  assert.equal(result.killed, deaths.length);
  const stillOnPizza = simulation.snapshot().flies.filter((fly) => fly.spot === 'pizza');
  assert.equal(stillOnPizza.length, 0, 'survivors should not hang around a swatter');
});

test('the colony recovers from a massacre and never overflows the kitchen', () => {
  const simulation = createSimulation({ seed: 9, minFlies: 6, maxFlies: 20 });
  for (let i = 0; i < 300; i++) {
    for (const spot of ['banana', 'pizza', 'trash', 'mug', 'lamp', 'window', 'web']) simulation.swat(spot);
    simulation.tick();
    const alive = simulation.snapshot().stats.alive;
    assert.ok(alive <= 20, `population ${alive} passed the max`);
  }
  run(simulation, 12);
  assert.ok(simulation.snapshot().stats.alive >= 6);
});

test('flies die of old age once they reach their lifespan', () => {
  const simulation = createSimulation({ seed: 2, startFlies: 8 });
  const events = record(simulation);
  run(simulation, 34 * 12 * 1.25);
  const agedOut = events.filter((event) => event.type === 'death' && event.cause === 'age');
  assert.ok(agedOut.length >= 8, 'every starting fly should be gone by the longest possible lifespan');
  for (const death of agedOut) assert.ok(death.ageDays <= Math.ceil(34 * 1.2));
});

test('mutual crushes lay eggs that hatch into new flies with parents', () => {
  const simulation = createSimulation({ seed: 4, actChance: 0.7 });
  const events = record(simulation);
  run(simulation, 500);
  const hatchedWithParents = events.filter((event) => event.type === 'hatch' && event.fly.parents.length === 2);
  assert.ok(hatchedWithParents.length > 0, 'love should find a way');
});

test('the feed stays capped and newest first', () => {
  const simulation = createSimulation({ seed: 8, feedLimit: 30, actChance: 0.8 });
  run(simulation, 120);
  const { feed } = simulation.snapshot();
  assert.ok(feed.length <= 30);
  for (let i = 1; i < feed.length; i++) assert.ok(feed[i - 1].tick >= feed[i].tick);
});

test('snapshots never leak internal sets that break JSON', () => {
  const simulation = createSimulation({ seed: 1 });
  run(simulation, 50);
  const json = JSON.parse(JSON.stringify(simulation.snapshot()));
  assert.ok(json.feed.every((buzz) => !('likedBy' in buzz)));
  assert.ok(json.flies.every((fly) => typeof fly.followers === 'number'));
});

test('human actions reject spots that do not exist or cannot hold food', () => {
  const simulation = createSimulation({ seed: 1 });
  assert.throws(() => simulation.swat('fridge'), SimulationError);
  assert.throws(() => simulation.dropSnack('window'), SimulationError);
  assert.equal(simulation.dropSnack('banana').spot, 'banana');
});
