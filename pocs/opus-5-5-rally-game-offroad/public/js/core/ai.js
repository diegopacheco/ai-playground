import { clamp, wrapAngle } from './math.js';
import { rightAt } from './tracks.js';
import { G } from './vehicle.js';

const JUMP_SPEED = 21;

export function createDriver(skill, lane) {
  return { skill, lane, laneShift: 0, stuckTime: 0, reverseTime: 0, reverseSteer: 0 };
}

function targetSpeed(track, i, u, mu, skill) {
  const decel = mu * G * 0.75;
  let best = Infinity;
  const steps = Math.ceil((u * u) / (2 * decel) / track.spacing) + 12;
  for (let k = 0; k < Math.min(steps, 140); k++) {
    const j = (i + k) % track.count;
    const v = track.jumps.includes(j) ? JUMP_SPEED : track.vmaxUnit[j] * Math.sqrt(mu) * skill;
    const allowed = Math.sqrt(v * v + 2 * decel * k * track.spacing);
    if (allowed < best) best = allowed;
  }
  return best;
}

export function driveAI(driver, car, track, proj, others, mu, dt) {
  if (driver.reverseTime > 0) {
    driver.reverseTime -= dt;
    return { throttle: 0, brake: 1, steer: driver.reverseSteer, handbrake: false };
  }
  let shift = 0;
  for (const o of others) {
    const dx = o.x - car.x;
    const dz = o.z - car.z;
    const ahead = dx * Math.sin(car.heading) + dz * Math.cos(car.heading);
    const side = dx * -Math.cos(car.heading) + dz * Math.sin(car.heading);
    if (ahead > 0 && ahead < 14 && Math.abs(side) < 2.6) shift = side > 0 ? -3 : 3;
  }
  driver.laneShift += (shift - driver.laneShift) * clamp(dt * 1.5, 0, 1);
  const lane = clamp(driver.lane + driver.laneShift, -track.halfWidth + 1.8, track.halfWidth - 1.8);

  const u = Math.max(car.u, 0);
  const look = Math.round((7 + u * 0.55) / track.spacing);
  const j = (proj.i + look) % track.count;
  const [rx, rz] = rightAt(track, j);
  const tx = track.xs[j] + rx * lane;
  const tz = track.zs[j] + rz * lane;
  const desired = Math.atan2(tx - car.x, tz - car.z);
  const err = wrapAngle(desired - car.heading);
  const steer = clamp(-err * 2.6, -1, 1);

  const vt = targetSpeed(track, proj.i, u, mu, driver.skill);
  let throttle = 0;
  let brake = 0;
  if (car.u < vt - 0.5) throttle = clamp((vt - car.u) / 4, 0.35, 1) * (Math.abs(err) > 0.9 ? 0.5 : 1);
  else if (car.u > vt + 1.2) brake = clamp((car.u - vt) / 5, 0.2, 1);

  if (car.u < 1.2 && throttle > 0 && !car.air) driver.stuckTime += dt;
  else driver.stuckTime = Math.max(0, driver.stuckTime - dt * 2);
  if (driver.stuckTime > 2.2) {
    driver.stuckTime = 0;
    driver.reverseTime = 1.3;
    driver.reverseSteer = err > 0 ? 1 : -1;
  }
  return { throttle, brake, steer, handbrake: false };
}
