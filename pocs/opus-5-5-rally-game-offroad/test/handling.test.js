import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TRACKS } from '../public/js/core/tracks.js';
import { createSim, stepSim } from '../public/js/core/sim.js';
import { CARS, SURFACES, WEATHER } from '../public/js/core/vehicle.js';
import { paceNotes, nextNote } from '../public/js/core/pacenotes.js';
import { worldFor } from './geo-fixture.js';
import { createKeyboardDriver } from './keyboard-driver.js';

const worlds = Object.fromEntries(TRACKS.map((def) => [def.id, worldFor(def)]));

function keyboardLaps(def, weather, spec, laps = 1) {
  const world = worlds[def.id];
  const sim = createSim({ world, weather, specs: [spec], playerIndex: 0, laps });
  const drive = createKeyboardDriver(world.track, paceNotes(world.track, SURFACES.mud.grip * WEATHER[weather].grip));
  let input = { throttle: 0, brake: 0, steer: 0, handbrake: false };
  let off = 0;
  let frames = 0;
  while (!sim.race.entries[0].finished && sim.race.time < 240) {
    stepSim(sim, 1 / 60, input);
    input = { ...drive(sim.cars[0], 1 / 60) };
    frames++;
    if (sim.cars[0].proj.dist > world.track.halfWidth) off++;
  }
  return { finished: sim.race.entries[0].finished, off: off / frames };
}

for (const def of TRACKS) {
  for (const [weather, spec] of [['clear', CARS[4]], ['snow', CARS[0]]]) {
    test(`${def.city}, ${weather}, ${spec.name}: a keyboard player who brakes for the pace notes keeps it on the road`, () => {
      const r = keyboardLaps(def, weather, spec);
      assert.ok(r.finished, 'completed the lap');
      assert.ok(r.off < 0.05, `off road ${(r.off * 100).toFixed(1)}% of the lap`);
    });
  }
}

test('pace notes call a tight bend as a low number and a sweeper as a high number, in the right direction', () => {
  const track = worlds.la.track;
  const notes = paceNotes(track, SURFACES.mud.grip);
  const tight = notes.reduce((a, b) => (a.radius < b.radius ? a : b));
  const wide = notes.reduce((a, b) => (a.radius > b.radius ? a : b));
  assert.ok(tight.severity < wide.severity);
  assert.ok(tight.speed < wide.speed);
  for (const n of notes) assert.equal(n.dir, track.curvature[n.i] > 0 ? 'LEFT' : 'RIGHT');
});

test('the next pace note is found even when the corner starts before the finish line', () => {
  const track = worlds.sf.track;
  const notes = paceNotes(track, SURFACES.mud.grip);
  for (let i = 0; i < track.count; i += 5) {
    const next = nextNote(track, notes, i);
    assert.ok(next && next.distance < track.length / 2 + 400, `index ${i} next note ${next?.distance}`);
  }
});

test('at speed the steering asks for a share of the grip instead of snapping to the limit, so tapping a key does not throw the car sideways', async () => {
  const { createCar, stepCar } = await import('../public/js/core/vehicle.js');
  const env = { heightAt: () => 0, grip: SURFACES.mud.grip, rolling: SURFACES.mud.rolling, bump: 0 };
  const yawAfter = (steer) => {
    const car = createCar(CARS[4], 0, 0, 0, 0);
    car.vz = 40;
    for (let t = 0; t < 1; t += 1 / 120) stepCar(car, { throttle: 0.3, brake: 0, steer, handbrake: false }, env, 1 / 120);
    return Math.abs(car.yawRate);
  };
  assert.ok(yawAfter(0.3) < yawAfter(1) * 0.5);
});
