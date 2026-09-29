import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CARS, createCar, stepCar, speedKmh, SURFACES, WEATHER } from '../public/js/core/vehicle.js';

const flat = (grip = SURFACES.mud.grip) => ({ heightAt: () => 0, grip, rolling: SURFACES.mud.rolling, bump: 0 });
const DT = 1 / 120;

function drive(car, input, env, seconds) {
  for (let t = 0; t < seconds; t += DT) stepCar(car, input, env, DT);
  return car;
}

test('the garage has 7 distinct off-road vehicles, led by the Jeep', () => {
  assert.equal(CARS.length, 7);
  assert.equal(new Set(CARS.map((c) => c.id)).size, 7);
  assert.match(CARS[0].name, /Jeep/);
});

test('full throttle accelerates a Jeep to highway speed within 12 seconds on mud', () => {
  const car = drive(createCar(CARS[0], 0, 0, 0, 0), { throttle: 1, brake: 0, steer: 0 }, flat(), 12);
  assert.ok(speedKmh(car) > 100, `${speedKmh(car).toFixed(0)} km/h`);
});

test('the automatic gearbox upshifts under load and never passes redline', () => {
  const car = createCar(CARS[4], 0, 0, 0, 0);
  let maxRpm = 0;
  for (let t = 0; t < 15; t += DT) {
    stepCar(car, { throttle: 1, brake: 0, steer: 0 }, flat(), DT);
    maxRpm = Math.max(maxRpm, car.rpm);
  }
  assert.ok(car.gear >= 4, `gear ${car.gear}`);
  assert.ok(maxRpm <= 6400);
});

test('the gearbox downshifts when the car slows so it can pull out of the next corner', () => {
  const car = drive(createCar(CARS[0], 0, 0, 0, 0), { throttle: 1, brake: 0, steer: 0 }, flat(), 12);
  const high = car.gear;
  drive(car, { throttle: 0, brake: 1, steer: 0 }, flat(), 3);
  assert.ok(car.gear < high);
});

test('top speed is bounded by aero drag and the limiter', () => {
  for (const spec of CARS) {
    const car = drive(createCar(spec, 0, 0, 0, 0), { throttle: 1, brake: 0, steer: 0 }, flat(1), 60);
    assert.ok(car.u < spec.topSpeed + 4, `${spec.id} ${car.u.toFixed(1)}`);
  }
});

test('holding brake from standstill reverses the car, like an automatic 4x4', () => {
  const car = drive(createCar(CARS[0], 0, 0, 0, 0), { throttle: 0, brake: 1, steer: 0 }, flat(), 3);
  assert.equal(car.gear, -1);
  assert.ok(car.u < -2);
});

test('snow reduces cornering grip, so the same corner at the same speed turns wider', () => {
  const turned = (grip) => {
    const car = createCar(CARS[0], 0, 0, 0, 0);
    car.vz = 25;
    drive(car, { throttle: 0.5, brake: 0, steer: 1 }, flat(grip), 1.5);
    return Math.abs(car.heading);
  };
  const snow = turned(SURFACES.mud.grip * WEATHER.snow.grip);
  const clear = turned(SURFACES.mud.grip * WEATHER.clear.grip);
  assert.ok(snow < clear * 0.9, `snow ${snow.toFixed(2)} clear ${clear.toFixed(2)}`);
});

test('the handbrake kicks the tail out for rally hairpins', () => {
  const run = (handbrake) => {
    const car = createCar(CARS[0], 0, 0, 0, 0);
    car.vz = 18;
    for (let t = 0; t < 0.8; t += DT) stepCar(car, { throttle: 0, brake: 0, steer: 0.6, handbrake }, flat(), DT);
    return Math.abs(car.heading);
  };
  assert.ok(run(true) > run(false) * 1.2);
});

test('steering right turns the car right', () => {
  const car = createCar(CARS[0], 0, 0, 0, 0);
  car.vz = 15;
  drive(car, { throttle: 0.3, brake: 0, steer: 1 }, flat(), 1);
  assert.ok(car.x < -1, `x ${car.x.toFixed(2)}`);
});

test('a car launched off a crest flies, then lands with an impact', () => {
  const ramp = { heightAt: (x, z) => (z < 20 ? z * 0.25 : Math.max(0, 5 - (z - 20) * 2)), grip: 0.8, rolling: 0.03, bump: 0 };
  const car = createCar(CARS[6], 0, 0, 0, 0);
  car.vz = 25;
  let flew = false;
  let landed = false;
  for (let t = 0; t < 3; t += DT) {
    stepCar(car, { throttle: 1, brake: 0, steer: 0 }, ramp, DT);
    flew = flew || car.air;
    landed = landed || car.impact > 0;
  }
  assert.ok(flew && landed);
});
