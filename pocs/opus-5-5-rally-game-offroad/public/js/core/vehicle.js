import { clamp } from './math.js';

export const G = 9.81;

export const CARS = [
  { id: 'wrangler', name: 'Jeep Wrangler Rubicon', style: 'wrangler', mass: 2050, torque: 470, grip: 1.0, finalDrive: 4.1, topSpeed: 50, length: 4.3, width: 1.9, height: 1.85, wheelbase: 2.46, track: 1.62, wheelR: 0.42, clearance: 0.58, cylinders: 6, accent: 'Trail rated legend, short wheelbase, flicks through hairpins.' },
  { id: 'bronco', name: 'Ford Bronco Raptor', style: 'bronco', mass: 2300, torque: 595, grip: 1.02, finalDrive: 4.0, topSpeed: 53, length: 4.8, width: 2.2, height: 1.95, wheelbase: 2.95, track: 1.77, wheelR: 0.46, clearance: 0.62, cylinders: 6, accent: 'Twin turbo V6, wide stance, long-travel suspension.' },
  { id: 'cruiser', name: 'Toyota Land Cruiser 70', style: 'cruiser', mass: 2250, torque: 430, grip: 0.98, finalDrive: 4.3, topSpeed: 46, length: 4.9, width: 1.87, height: 1.94, wheelbase: 2.98, track: 1.55, wheelR: 0.4, clearance: 0.54, cylinders: 8, accent: 'Unkillable V8 diesel workhorse, stable on rough ground.' },
  { id: 'defender', name: 'Land Rover Defender 110', style: 'defender', mass: 2350, torque: 550, grip: 1.03, finalDrive: 3.9, topSpeed: 52, length: 5.0, width: 2.0, height: 1.97, wheelbase: 3.02, track: 1.7, wheelR: 0.42, clearance: 0.56, cylinders: 6, accent: 'Air suspension and torque vectoring, calm and precise.' },
  { id: 'raptor', name: 'Ford F-150 Raptor R', style: 'raptor', mass: 2700, torque: 868, grip: 1.0, finalDrive: 4.1, topSpeed: 58, length: 5.9, width: 2.2, height: 2.0, wheelbase: 3.7, track: 1.83, wheelR: 0.46, clearance: 0.6, cylinders: 8, accent: 'Supercharged V8 desert truck, brutal on the straights.' },
  { id: 'hummer', name: 'Hummer H1 Alpha', style: 'hummer', mass: 3100, torque: 700, grip: 1.06, finalDrive: 4.3, topSpeed: 45, length: 4.7, width: 2.2, height: 1.9, wheelbase: 3.3, track: 1.82, wheelR: 0.45, clearance: 0.52, cylinders: 8, accent: 'Portal axles and huge track width, a tank that never tips.' },
  { id: 'trophy', name: 'Baja Trophy Truck', style: 'trophy', mass: 1950, torque: 780, grip: 1.08, finalDrive: 3.8, topSpeed: 62, length: 5.6, width: 2.3, height: 1.9, wheelbase: 3.35, track: 2.05, wheelR: 0.5, clearance: 0.7, cylinders: 8, accent: 'Tube frame race truck with 30 inches of wheel travel.' },
];

export const GEARS = [3.8, 2.3, 1.55, 1.15, 0.9, 0.72];
const IDLE = 850;
const REDLINE = 6400;
const REVERSE_RATIO = 3.5;

export const SURFACES = {
  mud: { grip: 0.78, rolling: 0.035, name: 'mud' },
  offroad: { grip: 0.62, rolling: 0.07, name: 'offroad' },
  puddle: { grip: 0.55, rolling: 0.16, name: 'puddle' },
};

export const WEATHER = {
  clear: { grip: 1.0 },
  rain: { grip: 0.84 },
  snow: { grip: 0.66 },
};

export function createCar(spec, x, z, heading, y) {
  return {
    spec, x, z, y, heading,
    vx: 0, vz: 0, vy: 0, yawRate: 0,
    u: 0, lat: 0,
    gear: 1, rpm: IDLE, shiftTimer: 0,
    air: false, impact: 0,
    pitch: 0, roll: 0, bodyPitch: 0, bodyRoll: 0, bodyPitchV: 0, bodyRollV: 0,
    wheelSpin: 0, slip: 0, accelLong: 0, accelLat: 0,
    wheelAngle: 0, steerAngle: 0, wheelComp: [0, 0, 0, 0],
    braking: false, dirt: 0,
  };
}

function torqueAt(spec, rpm) {
  const n = rpm / REDLINE;
  return spec.torque * clamp(1 - (n - 0.55) * (n - 0.55) * 1.6, 0.45, 1);
}

function wheelPositions(car) {
  const s = car.spec;
  const sh = Math.sin(car.heading);
  const ch = Math.cos(car.heading);
  const out = [];
  for (const [lx, lz] of [[s.track / 2, s.wheelbase / 2], [-s.track / 2, s.wheelbase / 2], [s.track / 2, -s.wheelbase / 2], [-s.track / 2, -s.wheelbase / 2]]) {
    out.push([car.x + ch * lx + sh * lz, car.z - sh * lx + ch * lz]);
  }
  return out;
}

function updateGearbox(car, input, dt) {
  const s = car.spec;
  car.shiftTimer = Math.max(0, car.shiftTimer - dt);
  if (car.gear === -1) {
    if (input.throttle > 0.05 && car.u > -0.8) car.gear = 1;
  } else if (input.brake > 0.05 && input.throttle < 0.05 && car.u < 0.6) {
    car.gear = -1;
  }
  const ratio = car.gear === -1 ? REVERSE_RATIO : GEARS[car.gear - 1];
  const wheelRpm = (Math.abs(car.u) / s.wheelR) * 60 / (Math.PI * 2);
  let rpm = wheelRpm * ratio * s.finalDrive;
  if (car.gear >= 1 && car.shiftTimer === 0) {
    if (rpm > 5700 && car.gear < GEARS.length) {
      car.gear++;
      car.shiftTimer = 0.28;
    } else if (rpm < 2800 && car.gear > 1) {
      car.gear--;
      car.shiftTimer = 0.18;
    }
  }
  const pedal = car.gear === -1 ? input.brake : input.throttle;
  const launch = car.gear <= 1 ? IDLE + pedal * 3200 : IDLE;
  rpm = Math.max(rpm, launch, IDLE);
  if (car.air) rpm = Math.max(rpm, IDLE + pedal * 5200);
  rpm = Math.min(rpm + car.wheelSpin * 1800, REDLINE);
  car.rpm += (rpm - car.rpm) * clamp(dt * 12, 0, 1);
  return ratio;
}

export function stepCar(car, input, env, dt) {
  const s = car.spec;
  const fx = Math.sin(car.heading);
  const fz = Math.cos(car.heading);
  const rx = -Math.cos(car.heading);
  const rz = Math.sin(car.heading);
  let u = car.vx * fx + car.vz * fz;
  let lat = car.vx * rx + car.vz * rz;
  car.u = u;

  const ratio = updateGearbox(car, input, dt);
  const mu = env.grip * s.grip;
  const traction = car.air ? 0 : 1;

  const wheels = wheelPositions(car);
  const hs = wheels.map(([x, z]) => env.heightAt(x, z));
  const front = (hs[0] + hs[1]) / 2;
  const rear = (hs[2] + hs[3]) / 2;
  const left = (hs[0] + hs[2]) / 2;
  const right = (hs[1] + hs[3]) / 2;
  const slopeF = (front - rear) / s.wheelbase;
  const slopeR = (right - left) / s.track;

  let aLong = 0;
  car.braking = false;
  if (traction) {
    const limit = mu * G * 0.95;
    let drive = 0;
    if (car.shiftTimer === 0) {
      if (car.gear >= 1) drive = input.throttle * torqueAt(s, car.rpm) * ratio * s.finalDrive * 0.86 / s.wheelR / s.mass;
      else drive = -input.brake * torqueAt(s, car.rpm) * ratio * s.finalDrive * 0.5 / s.wheelR / s.mass;
    }
    car.wheelSpin = clamp((Math.abs(drive) - limit) / limit, 0, 1);
    drive = clamp(drive, -limit, limit);
    let brake = 0;
    if (car.gear >= 1 && input.brake > 0) brake = input.brake * limit;
    if (car.gear === -1 && input.throttle > 0) brake = input.throttle * limit;
    if (input.handbrake) brake = Math.max(brake, limit * 0.45);
    car.braking = brake > 0.5;
    const rolling = env.rolling * G * Math.sign(u) * Math.min(1, Math.abs(u));
    const aero = (0.33 * s.width * s.height / s.mass) * u * Math.abs(u) + (Math.abs(u) > s.topSpeed ? (Math.abs(u) - s.topSpeed) * 0.6 * Math.sign(u) : 0);
    aLong = drive - rolling - aero - G * slopeF * 0.9;
    const before = u;
    u += aLong * dt;
    if (brake > 0) {
      const dv = brake * dt;
      u = Math.abs(u) <= dv ? 0 : u - Math.sign(u) * dv;
      if (Math.sign(before) !== Math.sign(u) && before !== 0) u = 0;
    }
  }

  let latGrip = mu * G * (input.handbrake ? 0.42 : 1);
  if (traction) {
    lat -= G * slopeR * 0.5 * dt;
    const want = -lat * clamp(12 * dt, 0, 1);
    const cap = latGrip * dt;
    lat += clamp(want, -cap, cap);
  }
  car.slip = clamp(Math.abs(lat) / 6, 0, 1);

  const speedFactor = 1 + Math.abs(u) / 20;
  const steerAngle = (-input.steer * 0.6) / speedFactor;
  car.steerAngle += (steerAngle - car.steerAngle) * clamp(dt * 10, 0, 1);
  if (traction) {
    let yawTarget = (u * Math.tan(car.steerAngle)) / s.wheelbase;
    const yawMax = (mu * G * 1.2) / Math.max(Math.abs(u), 2);
    yawTarget = clamp(yawTarget, -yawMax, yawMax);
    if (input.handbrake && Math.abs(u) > 4) yawTarget *= 1.7;
    car.yawRate += (yawTarget - car.yawRate) * clamp(dt * 7 * clamp(mu, 0.4, 1.2), 0, 1);
  } else {
    car.yawRate *= 1 - clamp(dt * 3, 0, 1);
  }

  car.accelLong = aLong;
  car.accelLat = u * car.yawRate;
  car.vx = fx * u + rx * lat;
  car.vz = fz * u + rz * lat;
  car.u = u;
  car.lat = lat;
  car.heading += car.yawRate * dt;
  car.x += car.vx * dt;
  car.z += car.vz * dt;

  const ground = (hs[0] + hs[1] + hs[2] + hs[3]) / 4;
  car.impact = 0;
  if (car.air) {
    car.vy -= G * dt;
    car.y += car.vy * dt;
    if (car.y <= ground) {
      car.impact = Math.max(0, -car.vy);
      car.y = ground;
      car.vy = 0;
      car.air = false;
    }
  } else {
    const ballistic = car.y + car.vy * dt - 0.5 * G * dt * dt;
    if (ground < ballistic - 0.002 && car.vy > 1.0) {
      car.air = true;
      car.vy -= G * dt;
      car.y = ballistic;
    } else {
      car.vy = clamp((ground - car.y) / dt, -25, 25);
      car.y = ground;
    }
  }

  if (!car.air) {
    car.pitch += (-Math.atan(slopeF) - car.pitch) * clamp(dt * 14, 0, 1);
    car.roll += (Math.atan((left - right) / s.track) - car.roll) * clamp(dt * 14, 0, 1);
  } else {
    car.pitch += (0.18 - car.pitch) * clamp(dt * 0.9, 0, 1);
  }
  const k = 90;
  const c = 11;
  const tp = clamp(aLong * 0.011, -0.07, 0.07) + (env.bump || 0) * 0.01;
  const tr = clamp(car.accelLat * 0.012, -0.09, 0.09);
  car.bodyPitchV += ((tp - car.bodyPitch) * k - car.bodyPitchV * c) * dt;
  car.bodyRollV += ((tr - car.bodyRoll) * k - car.bodyRollV * c) * dt;
  car.bodyPitch += car.bodyPitchV * dt;
  car.bodyRoll += car.bodyRollV * dt;
  if (car.impact > 2) car.bodyPitchV -= car.impact * 0.08;

  for (let w = 0; w < 4; w++) car.wheelComp[w] = clamp(hs[w] - ground, -0.25, 0.25);
  car.wheelAngle += (u / s.wheelR) * dt + (car.wheelSpin * 30 * dt * Math.sign(car.gear));
  return car;
}

export function speedKmh(car) {
  return Math.abs(car.u) * 3.6;
}
