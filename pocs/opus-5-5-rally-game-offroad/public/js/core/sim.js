import { buildTrack, gridSlots, project, puddleAt } from './tracks.js';
import { buildTerrain } from './terrain.js';
import { placeProps, obstaclesNear } from './props.js';
import { createCar, stepCar, SURFACES, WEATHER } from './vehicle.js';
import { createDriver, driveAI } from './ai.js';
import { createRace, updateEntry, standings, LAPS } from './race.js';
import { clamp, createNoise2D } from './math.js';

export const STEP = 1 / 120;
export const COUNTDOWN = 3;
const CAR_RADIUS = 1.9;
const IDLE_INPUT = { throttle: 0, brake: 0, steer: 0, handbrake: false };

export function createWorld(trackDef) {
  const track = buildTrack(trackDef);
  const terrain = buildTerrain(track);
  const props = placeProps(track, terrain);
  return { track, terrain, props };
}

export function createSim({ world, weather, specs, playerIndex = specs.length - 1, laps = LAPS }) {
  const { track, terrain, props } = world;
  const slots = gridSlots(track, specs.length);
  const cars = specs.map((spec, k) => {
    const car = createCar(spec, slots[k].x, slots[k].z, slots[k].heading, terrain.heightAt(slots[k].x, slots[k].z));
    car.id = k;
    car.hint = slots[k].i;
    car.proj = project(track, car.x, car.z, car.hint);
    car.surface = SURFACES.mud;
    car.puddle = null;
    return car;
  });
  const skills = [0.93, 0.9, 0.87, 0.84, 0.8];
  const drivers = specs.map((_, k) => createDriver(skills[k % skills.length], ((k % 3) - 1) * 2.2));
  const race = createRace(track, specs.map((_, k) => k), laps);
  const bumpNoise = createNoise2D(track.def.seed + 5);
  return {
    world, weather, cars, drivers, race, playerIndex,
    phase: 'countdown', countdown: COUNTDOWN, events: [],
    bumpNoise, accumulator: 0,
  };
}

function envFor(sim, car) {
  const { track, terrain } = sim.world;
  const proj = project(track, car.x, car.z, car.hint);
  car.hint = proj.i;
  car.proj = proj;
  const onRoad = proj.dist < track.halfWidth;
  car.puddle = onRoad ? puddleAt(track, car.x, car.z) : null;
  const surface = car.puddle ? SURFACES.puddle : onRoad ? SURFACES.mud : SURFACES.offroad;
  car.surface = surface;
  const beyond = Math.max(0, proj.dist - track.halfWidth - 30);
  const rough = onRoad ? 1 : 2.2;
  return {
    heightAt: terrain.heightAt,
    grip: surface.grip * WEATHER[sim.weather].grip,
    rolling: surface.rolling + beyond * 0.03 + (sim.weather === 'snow' && !onRoad ? 0.04 : 0),
    bump: sim.bumpNoise(car.x * 0.35, car.z * 0.35) * rough * Math.min(1, Math.abs(car.u) / 8),
  };
}

function collideCars(sim) {
  const cars = sim.cars;
  for (let a = 0; a < cars.length; a++) {
    for (let b = a + 1; b < cars.length; b++) {
      const A = cars[a];
      const B = cars[b];
      const dx = B.x - A.x;
      const dz = B.z - A.z;
      const d = Math.hypot(dx, dz);
      const min = CAR_RADIUS * 2;
      if (d >= min || d === 0) continue;
      const nx = dx / d;
      const nz = dz / d;
      const push = (min - d) / 2;
      A.x -= nx * push;
      A.z -= nz * push;
      B.x += nx * push;
      B.z += nz * push;
      const rel = (B.vx - A.vx) * nx + (B.vz - A.vz) * nz;
      if (rel < 0) {
        const j = -rel * 0.6;
        const ma = A.spec.mass;
        const mb = B.spec.mass;
        A.vx -= nx * j * (mb / (ma + mb));
        A.vz -= nz * j * (mb / (ma + mb));
        B.vx += nx * j * (ma / (ma + mb));
        B.vz += nz * j * (ma / (ma + mb));
        if (-rel > 2) sim.events.push({ type: 'hit', x: A.x, z: A.z, strength: -rel, cars: [a, b] });
      }
    }
  }
}

function collideObstacles(sim, car, index) {
  for (const o of obstaclesNear(sim.world.props, car.x, car.z)) {
    const dx = car.x - o.x;
    const dz = car.z - o.z;
    const d = Math.hypot(dx, dz);
    const min = o.r + CAR_RADIUS * 0.8;
    if (d >= min || d === 0) continue;
    const nx = dx / d;
    const nz = dz / d;
    car.x = o.x + nx * min;
    car.z = o.z + nz * min;
    const vn = car.vx * nx + car.vz * nz;
    if (vn < 0) {
      car.vx -= nx * vn * 1.3;
      car.vz -= nz * vn * 1.3;
      car.vx *= 0.7;
      car.vz *= 0.7;
      if (-vn > 2) sim.events.push({ type: 'hit', x: car.x, z: car.z, strength: -vn, cars: [index] });
    }
  }
}

function fixedStep(sim, playerInput, dt) {
  const racing = sim.phase === 'racing' || sim.phase === 'finished';
  sim.race.time += racing ? dt : 0;
  const mu = WEATHER[sim.weather].grip * SURFACES.mud.grip;
  sim.cars.forEach((car, k) => {
    const env = envFor(sim, car);
    let input = IDLE_INPUT;
    const entry = sim.race.entries[k];
    if (racing) {
      if (k === sim.playerIndex && !entry.finished) input = playerInput;
      else input = driveAI(sim.drivers[k], car, sim.world.track, car.proj, sim.cars.filter((c) => c !== car), mu, dt);
    } else if (k === sim.playerIndex) {
      input = { throttle: playerInput.throttle, brake: 0, steer: 0, handbrake: true };
    }
    const held = racing ? null : { x: car.x, z: car.z, heading: car.heading };
    stepCar(car, input, env, dt);
    if (held) {
      Object.assign(car, held);
      car.vx = car.vz = car.u = car.lat = car.yawRate = 0;
    }
    car.input = input;
    const dirtRate = car.surface === SURFACES.offroad ? 0.004 : 0.012;
    car.dirt = clamp(car.dirt + Math.abs(car.u) * dt * dirtRate * (sim.weather === 'snow' ? 0.4 : 1) * 0.1, 0, 1);
    if (car.impact > 3) sim.events.push({ type: 'land', x: car.x, z: car.z, strength: car.impact, cars: [k] });
    collideObstacles(sim, car, k);
  });
  collideCars(sim);
  sim.cars.forEach((car, k) => updateEntry(sim.race, sim.race.entries[k], car.proj.i));
}

export function stepSim(sim, frameDt, playerInput = IDLE_INPUT) {
  const dt = Math.min(frameDt, 0.1);
  sim.events = [];
  if (sim.phase === 'countdown') {
    sim.countdown -= dt;
    if (sim.countdown <= 0) sim.phase = 'racing';
  }
  sim.accumulator += dt;
  let steps = 0;
  while (sim.accumulator >= STEP && steps < 12) {
    fixedStep(sim, playerInput, STEP);
    sim.accumulator -= STEP;
    steps++;
  }
  if (steps === 12) sim.accumulator = 0;
  if (sim.phase === 'racing' && sim.race.entries[sim.playerIndex]?.finished) sim.phase = 'finished';
  return sim;
}

export function resetCarToTrack(sim, k) {
  const car = sim.cars[k];
  const { track, terrain } = sim.world;
  const i = car.proj.i;
  car.x = track.xs[i];
  car.z = track.zs[i];
  car.heading = track.heading[i];
  car.y = terrain.heightAt(car.x, car.z) + 0.3;
  car.vx = car.vz = car.vy = car.yawRate = car.u = car.lat = 0;
  car.air = false;
}

export function standingsOf(sim) {
  return standings(sim.race);
}
