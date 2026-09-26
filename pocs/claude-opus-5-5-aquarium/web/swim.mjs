import { TANK } from './catalog.mjs';

export const FLEE_RADIUS = 0.22;

export const EAT_RADIUS = 0.018;

export const SINK_SPEED = 0.03;

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function swimBox(margin = 0.06) {
  return {
    minX: -TANK.width / 2 + margin,
    maxX: TANK.width / 2 - margin,
    minY: 0.1,
    maxY: TANK.water - margin,
    minZ: -TANK.depth / 2 + margin,
    maxZ: TANK.depth / 2 - margin
  };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function pickTarget(random, box, band) {
  const h = box.maxY - box.minY;
  return {
    x: lerp(box.minX, box.maxX, random()),
    y: box.minY + h * lerp(band[0], band[1], random()),
    z: lerp(box.minZ, box.maxZ, random())
  };
}

export function createSwimmer(random, box, band, speed) {
  const pos = pickTarget(random, box, band);
  return { pos, vel: { x: speed * (random() - 0.5), y: 0, z: speed * (random() - 0.5) }, target: pickTarget(random, box, band), band, speed, fleeing: 0 };
}

function length(v) {
  return Math.hypot(v.x, v.y, v.z);
}

function scaled(v, s) {
  const l = length(v) || 1;
  return { x: (v.x / l) * s, y: (v.y / l) * s, z: (v.z / l) * s };
}

export function step(s, dt, random, box, threat, food = null) {
  const goal = food || s.target;
  const toTarget = { x: goal.x - s.pos.x, y: goal.y - s.pos.y, z: goal.z - s.pos.z };
  let desired = scaled(toTarget, s.speed * (food ? 1.8 : 1));
  let turn = 1.6;
  if (threat) {
    const away = { x: s.pos.x - threat.x, y: s.pos.y - threat.y, z: s.pos.z - threat.z };
    const d = length(away);
    if (d < FLEE_RADIUS) {
      const w = 1 - d / FLEE_RADIUS;
      const flee = scaled(away, s.speed * 2.6);
      desired = { x: lerp(desired.x, flee.x, w), y: lerp(desired.y, flee.y, w), z: lerp(desired.z, flee.z, w) };
      turn = 5;
      s.fleeing = 1;
    }
  }
  const k = Math.min(1, dt * turn);
  s.vel.x += (desired.x - s.vel.x) * k;
  s.vel.y += (desired.y - s.vel.y) * k * 0.6;
  s.vel.z += (desired.z - s.vel.z) * k;
  s.pos.x += s.vel.x * dt;
  s.pos.y += s.vel.y * dt;
  s.pos.z += s.vel.z * dt;
  for (const [axis, lo, hi] of [['x', box.minX, box.maxX], ['y', box.minY, box.maxY], ['z', box.minZ, box.maxZ]]) {
    if (s.pos[axis] < lo) {
      s.pos[axis] = lo;
      s.vel[axis] = Math.abs(s.vel[axis]) * 0.5;
    } else if (s.pos[axis] > hi) {
      s.pos[axis] = hi;
      s.vel[axis] = -Math.abs(s.vel[axis]) * 0.5;
    }
  }
  if ((!food && length(toTarget) < 0.05) || (s.fleeing && random() < dt)) {
    s.target = pickTarget(random, box, s.band);
    s.fleeing = 0;
  }
  return s;
}

export function nearestFood(pos, foods, reach = 0.5) {
  let best = null;
  let bestD = reach;
  for (const f of foods) {
    const d = Math.hypot(f.x - pos.x, f.y - pos.y, f.z - pos.z);
    if (d < bestD) {
      best = f;
      bestD = d;
    }
  }
  return best;
}

export function canEat(pos, food) {
  return Math.hypot(food.x - pos.x, food.y - pos.y, food.z - pos.z) < EAT_RADIUS;
}

export function sinkFood(f, dt, floor) {
  if (f.y > floor) {
    f.y = Math.max(floor, f.y - SINK_SPEED * dt);
    f.x += Math.sin(f.y * 90 + f.phase) * 0.01 * dt;
  } else {
    f.rest += dt;
  }
  return f;
}

export function heading(vel) {
  const flat = Math.hypot(vel.x, vel.z);
  return {
    yaw: Math.atan2(-vel.z, vel.x),
    pitch: Math.max(-0.5, Math.min(0.5, Math.atan2(vel.y, flat || 1e-6)))
  };
}

export function patrol(t, box) {
  const cx = (box.minX + box.maxX) / 2;
  const cz = (box.minZ + box.maxZ) / 2;
  const a = (box.maxX - box.minX) * 0.38;
  const b = (box.maxZ - box.minZ) * 0.34;
  const w = 0.32;
  const y = box.minY + (box.maxY - box.minY) * (0.55 + 0.15 * Math.sin(t * 0.45));
  return {
    pos: { x: cx + a * Math.cos(t * w), y, z: cz + b * Math.sin(t * w) },
    vel: { x: -a * w * Math.sin(t * w), y: 0.15 * 0.45 * Math.cos(t * 0.45) * (box.maxY - box.minY), z: b * w * Math.cos(t * w) }
  };
}
