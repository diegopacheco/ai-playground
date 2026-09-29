import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRace, updateEntry, standings, positionOf, formatTime } from '../public/js/core/race.js';
import { TRACKS } from '../public/js/core/tracks.js';
import { createWorld, createSim, stepSim } from '../public/js/core/sim.js';
import { CARS } from '../public/js/core/vehicle.js';

const fakeTrack = { count: 1000 };

function lap(race, entry) {
  for (let i = 10; i < 1000; i += 10) updateEntry(race, entry, i);
  updateEntry(race, entry, 5);
}

test('a lap only counts after passing the far side of the loop', () => {
  const race = createRace(fakeTrack, [0]);
  const e = race.entries[0];
  updateEntry(race, e, 900);
  updateEntry(race, e, 5);
  assert.equal(e.lap, 0, 'reversing over the line and back is not a lap');
  lap(race, e);
  assert.equal(e.lap, 1);
});

test('the race is 3 laps and records every lap time', () => {
  const race = createRace(fakeTrack, [0]);
  const e = race.entries[0];
  for (let k = 0; k < 3; k++) {
    race.time += 60 + k;
    lap(race, e);
  }
  assert.equal(race.laps, 3);
  assert.ok(e.finished);
  assert.deepEqual(e.lapTimes, [60, 61, 62]);
  assert.equal(e.finishTime, 183);
});

test('standings rank by distance covered, and finishers by finish time', () => {
  const race = createRace(fakeTrack, [0, 1, 2]);
  const [a, b, c] = race.entries;
  const advance = (e, to) => {
    for (let i = 0; i <= to; i += 10) updateEntry(race, e, i);
  };
  advance(a, 300);
  advance(b, 600);
  advance(c, 100);
  assert.deepEqual(standings(race).map((e) => e.id), [1, 0, 2]);
  assert.equal(positionOf(race, 2), 3);
});

test('lap times are shown as m:ss.cc', () => {
  assert.equal(formatTime(83.456), '1:23.46');
  assert.equal(formatTime(5), '0:05.00');
});

for (const def of TRACKS) {
  test(`${def.city}: 3 CPU rivals finish 3 laps in rain without getting stuck off the road`, () => {
    const world = createWorld(def);
    const sim = createSim({ world, weather: 'rain', specs: [CARS[1], CARS[3], CARS[5], CARS[0]], playerIndex: -1 });
    let off = 0;
    let frames = 0;
    while (sim.race.time < 600 && !sim.race.entries.every((e) => e.finished)) {
      stepSim(sim, 1 / 30);
      frames++;
      for (const car of sim.cars) if (car.proj.dist > world.track.halfWidth + 3) off++;
    }
    assert.ok(sim.race.entries.every((e) => e.finished), 'every CPU finished');
    assert.ok(off / (frames * 4) < 0.1, `off road ${(100 * off / (frames * 4)).toFixed(1)}%`);
  });
}

test('the player car stays on the grid during the countdown, then the race clock starts', () => {
  const world = createWorld(TRACKS[0]);
  const sim = createSim({ world, weather: 'clear', specs: [CARS[0], CARS[1], CARS[2], CARS[3]] });
  const start = { x: sim.cars[3].x, z: sim.cars[3].z };
  for (let t = 0; t < 2.5; t += 1 / 60) stepSim(sim, 1 / 60, { throttle: 1, brake: 0, steer: 0 });
  assert.equal(sim.phase, 'countdown');
  assert.ok(Math.hypot(sim.cars[3].x - start.x, sim.cars[3].z - start.z) < 0.05);
  for (let t = 0; t < 2; t += 1 / 60) stepSim(sim, 1 / 60, { throttle: 1, brake: 0, steer: 0 });
  assert.equal(sim.phase, 'racing');
  assert.ok(sim.race.time > 0);
});
