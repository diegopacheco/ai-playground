import { height, normal, COURSE_LENGTH, START_Z } from './terrain.js';

const G = 9.81;
const MU_BASE = 0.042;
const DRAG_UPRIGHT = 0.0042;
const DRAG_TUCK = 0.0021;
const DRAG_AIR = 0.0026;
const SIDECUT = 11.5;
const MAX_EDGE = 1.0;
const EDGE_IN = 3.4;
const EDGE_OUT = 5.6;
const GRIP_BASE = 0.5;
const GRIP_EDGE = 1.15;
const SKID_MU = 0.45;
const SCRUB_MU = 0.2;
const BRAKE_MU = 0.95;
const POP_MAX = 4.7;
const CHARGE_TIME = 0.5;
const SPIN_MAX = 4.3;
const SPIN_ACCEL = 9.5;
const HARD_LANDING = 12.5;
const BAD_ANGLE = 1.15;

const n = { x: 0, y: 1, z: 0 };

export class Rider {
  constructor(scenery) {
    this.trees = scenery.trees;
    this.rocks = scenery.rocks;
    this.buckets = new Map();
    for (const t of this.trees) {
      const k = Math.floor(t.z / 8);
      if (!this.buckets.has(k)) this.buckets.set(k, []);
      this.buckets.get(k).push(t);
    }
    this.reset();
  }

  reset() {
    this.x = 0;
    this.z = START_Z;
    this.y = height(0, START_Z);
    this.vx = 0;
    this.vy = 0;
    this.vz = 4;
    this.yaw = 0;
    this.edge = 0;
    this.pitch = 0;
    this.crouch = 0;
    this.charge = 0;
    this.airborne = false;
    this.grabbing = false;
    this.crashTime = 0;
    this.tumble = 0;
    this.skid = 0;
    this.speed = 4;
    this.topSpeed = 0;
    this.time = 0;
    this.score = 0;
    this.airTime = 0;
    this.lastAirTime = 0;
    this.totalAir = 0;
    this.spinAccum = 0;
    this.spinDir = 0;
    this.grabTime = 0;
    this.carveTime = 0;
    this.finished = false;
    this.finishTime = 0;
    this.events = [];
    this.impact = 0;
    this.slip = 0;
    this.crashes = 0;
    this.tricks = 0;
    this.stall = 0;
  }

  say(text, kind) {
    this.events.push({ text, kind, t: this.time });
  }

  step(dt, input) {
    if (this.finished) return;
    this.time += dt;

    const crashed = this.crashTime > 0;
    if (crashed) {
      this.crashTime -= dt;
      this.tumble += dt * 9;
      if (this.crashTime <= 0) {
        this.tumble = 0;
        this.edge = 0;
      }
    }

    const th = height(this.x, this.z);
    const gap = this.y - th;
    normal(this.x, this.z, n);

    const dhx = (height(this.x + 0.4, this.z) - height(this.x - 0.4, this.z)) / 0.8;
    const dhz = (height(this.x, this.z + 0.4) - height(this.x, this.z - 0.4)) / 0.8;
    const surfaceRate = this.vx * dhx + this.vz * dhz;

    let onGround = gap <= 0.02;
    if (!onGround && gap < 0.22 && this.vy - surfaceRate < 1.0) onGround = true;

    if (this.airborne && onGround) this.land(n);
    if (!this.airborne && !onGround) this.takeOff();

    if (onGround) this.ground(dt, input, crashed);
    else this.air(dt, input, crashed);

    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.z += this.vz * dt;

    const nh = height(this.x, this.z);
    if (this.y < nh) {
      this.y = nh;
      const vn = this.vx * n.x + this.vy * n.y + this.vz * n.z;
      if (vn < 0) {
        this.vx -= n.x * vn;
        this.vy -= n.y * vn;
        this.vz -= n.z * vn;
      }
    }

    this.speed = Math.hypot(this.vx, this.vy, this.vz);
    if (this.speed > this.topSpeed) this.topSpeed = this.speed;
    this.collide();

    if (this.z >= COURSE_LENGTH) {
      this.finished = true;
      this.finishTime = this.time;
    }
  }

  ground(dt, input, crashed) {
    this.airborne = false;
    const gn = Math.max(G * n.y, 1.5);

    const hold = Math.min(MAX_EDGE, 0.18 + Math.hypot(this.vx, this.vz) * 0.06);
    const target = crashed ? 0 : input.steer * hold;
    const rate = Math.abs(target) > Math.abs(this.edge) ? EDGE_IN : EDGE_OUT;
    this.edge += Math.max(-rate * dt, Math.min(rate * dt, target - this.edge));
    this.edge = Math.max(-hold, Math.min(hold, this.edge));

    let fx = Math.sin(this.yaw);
    let fz = Math.cos(this.yaw);
    const d = fx * n.x + fz * n.z;
    fx -= n.x * d;
    let fy = -n.y * d;
    fz -= n.z * d;
    const fl = 1 / Math.max(Math.hypot(fx, fy, fz), 1e-6);
    fx *= fl;
    fy *= fl;
    fz *= fl;

    const rx = n.y * fz - n.z * fy;
    const ry = n.z * fx - n.x * fz;
    const rz = n.x * fy - n.y * fx;

    let vf = this.vx * fx + this.vy * fy + this.vz * fz;
    let vl = this.vx * rx + this.vy * ry + this.vz * rz;

    const gd = -G * n.y;
    let ax = -n.x * gd;
    let ay = -G - n.y * gd;
    let az = -n.z * gd;

    const sinE = Math.sin(Math.abs(this.edge));
    const grip = gn * (GRIP_BASE + GRIP_EDGE * sinE) * Math.min(1, 0.32 + Math.abs(vf) / 7);
    const wanted = (vf * Math.sin(this.edge)) / SIDECUT;
    const need = Math.abs(vf * wanted);

    let omega = wanted;
    let spare = grip - need;
    if (spare < 0) {
      omega = wanted * (grip / Math.max(need, 1e-6));
      spare = 0;
    }
    if (crashed) omega = 0;
    this.yaw += omega * dt;

    const slow = crashed ? 0 : Math.max(0, 1 - Math.abs(vf) / 6);
    if (slow > 0) {
      this.yaw += input.steer * 0.9 * slow * dt;
      if (!input.steer) {
        const fall = Math.atan2(ax, az);
        const diff = ((fall - this.yaw + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
        if (Math.abs(diff) < 2.4) this.yaw += diff * 2.2 * slow * slow * dt;
      }
    }

    const slipMag = Math.abs(vl);
    this.slip = slipMag;
    const kinetic = SKID_MU * gn * Math.min(1, slipMag / 1.6);
    const brake = input.brake && !crashed ? BRAKE_MU * gn : 0;
    const lateralStop = Math.min(slipMag / dt, spare + kinetic + brake);
    vl -= Math.sign(vl) * lateralStop * dt;

    this.skid += (Math.min(1, slipMag / 3.2) - this.skid) * Math.min(1, dt * 6);

    const dragK = input.tuck && !crashed ? DRAG_TUCK : DRAG_UPRIGHT;
    const roll = MU_BASE * gn;
    const scrub = SCRUB_MU * gn * Math.min(1, slipMag / 2.5) + (input.brake && !crashed ? 0.5 * gn : 0);
    const crashDrag = crashed ? 2.6 * gn : 0;
    const along = vf > 0 ? -1 : 1;
    vf += along * (roll + scrub + crashDrag) * dt;
    vf -= vf * Math.abs(vf) * dragK * dt;
    if (vf < 0.05 && along < 0) vf = Math.max(vf, 0);

    this.vx = fx * vf + rx * vl + ax * dt;
    this.vy = fy * vf + ry * vl + ay * dt;
    this.vz = fz * vf + rz * vl + az * dt;

    const vn = this.vx * n.x + this.vy * n.y + this.vz * n.z;
    if (vn < 0) {
      this.vx -= n.x * vn;
      this.vy -= n.y * vn;
      this.vz -= n.z * vn;
    }

    this.pitch += (Math.asin(Math.max(-1, Math.min(1, -fy))) - this.pitch) * Math.min(1, dt * 8);

    if (!crashed && input.jump) {
      this.charge = Math.min(CHARGE_TIME, this.charge + dt);
      this.crouch += (1 - this.crouch) * Math.min(1, dt * 10);
    } else if (this.charge > 0) {
      const pop = POP_MAX * (0.35 + 0.65 * (this.charge / CHARGE_TIME));
      this.vx += n.x * pop;
      this.vy += n.y * pop;
      this.vz += n.z * pop;
      this.charge = 0;
      this.y += 0.05;
    } else {
      this.crouch += ((input.tuck ? 0.85 : 0.34 + Math.abs(this.edge) * 0.25) - this.crouch) * Math.min(1, dt * 6);
    }

    if (Math.hypot(this.vx, this.vz) < 0.7 && !crashed) {
      this.stall += dt;
      if (this.stall > 0.8) {
        this.vz += 2.4 * dt;
        this.vx -= Math.sign(this.x) * 0.5 * dt;
      }
    } else {
      this.stall = 0;
    }

    if (!crashed && Math.abs(this.edge) > 0.6 && this.skid < 0.3 && this.speed > 12) {
      this.carveTime += dt;
      if (this.carveTime > 1.4) {
        this.carveTime = 0;
        this.score += 60;
        this.say('CLEAN CARVE +60', 'carve');
      }
    } else {
      this.carveTime = 0;
    }
  }

  air(dt, input, crashed) {
    this.airborne = true;
    this.airTime += dt;

    if (!crashed) {
      const want = input.steer * SPIN_MAX;
      this.spinDir += Math.max(-SPIN_ACCEL * dt, Math.min(SPIN_ACCEL * dt, want - this.spinDir));
      this.grabbing = input.grab;
      if (this.grabbing) this.grabTime += dt;
      this.crouch += ((this.grabbing ? 1 : 0.25) - this.crouch) * Math.min(1, dt * 7);
    } else {
      this.spinDir = 5;
      this.grabbing = false;
    }

    this.yaw += this.spinDir * dt;
    this.spinAccum += Math.abs(this.spinDir) * dt;
    this.edge += (0 - this.edge) * Math.min(1, dt * 3);
    this.pitch += (0 - this.pitch) * Math.min(1, dt * 2);
    this.skid *= Math.max(0, 1 - dt * 3);

    this.vy -= G * dt;
    const v = Math.hypot(this.vx, this.vy, this.vz);
    const k = DRAG_AIR * v * dt;
    this.vx -= this.vx * k;
    this.vy -= this.vy * k;
    this.vz -= this.vz * k;
    this.charge = Math.min(CHARGE_TIME, input.jump ? this.charge : 0);
  }

  takeOff() {
    this.airborne = true;
    this.airTime = 0;
    this.spinAccum = 0;
    this.grabTime = 0;
    this.spinDir = 0;
  }

  land(nrm) {
    const vn = this.vx * nrm.x + this.vy * nrm.y + this.vz * nrm.z;
    this.impact = Math.max(0, -vn);
    this.lastAirTime = this.airTime;
    this.totalAir += this.airTime;

    const heading = Math.atan2(this.vx, this.vz);
    let off = Math.abs(((this.yaw - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
    const backwards = off > Math.PI / 2;
    if (backwards) off = Math.PI - off;

    const loss = Math.min(0.5, this.impact * 0.022);
    this.vx *= 1 - loss;
    this.vz *= 1 - loss;

    this.airborne = false;
    this.spinDir = 0;

    if (this.crashTime > 0) return;

    if (this.impact > HARD_LANDING || off > BAD_ANGLE) {
      this.crash(this.impact > HARD_LANDING ? 'HARD LANDING' : 'SKETCHY LANDING');
      return;
    }

    if (this.airTime > 0.35) {
      const spins = Math.floor((this.spinAccum + 0.32) / (Math.PI / 2));
      const degrees = spins * 90;
      let points = Math.round(this.airTime * 45 + degrees * 1.7);
      let label = degrees >= 180 ? `${degrees}` : 'AIR';
      if (this.grabTime > 0.25) {
        points = Math.round(points * 1.35 + 40);
        label += ' GRAB';
      }
      if (backwards && degrees < 180) label = 'SWITCH LANDING';
      this.score += points;
      this.tricks++;
      this.say(`${label} +${points}`, 'trick');
    }
  }

  crash(reason) {
    if (this.crashTime > 0) return;
    this.crashTime = 1.7;
    this.crashes++;
    this.vx *= 0.3;
    this.vz *= 0.3;
    this.vy *= 0.3;
    this.skid = 1;
    this.score = Math.max(0, this.score - 50);
    this.say(reason, 'crash');
  }

  collide() {
    if (this.crashTime > 0) return;
    const k = Math.floor(this.z / 8);
    for (let i = k - 1; i <= k + 1; i++) {
      const list = this.buckets.get(i);
      if (!list) continue;
      for (const t of list) {
        const dx = t.x - this.x;
        const dz = t.z - this.z;
        if (dx * dx + dz * dz < 1.6 && this.y - t.y < t.h) {
          this.crash('TREE!');
          return;
        }
      }
    }
    for (const r of this.rocks) {
      const dz = r.z - this.z;
      if (dz > 4 || dz < -4) continue;
      const dx = r.x - this.x;
      if (dx * dx + dz * dz < r.r * r.r && this.y - r.y < r.r && this.speed > 9) {
        this.crash('ROCK!');
        return;
      }
    }
  }
}
