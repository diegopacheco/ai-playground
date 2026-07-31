import { height, normal, PISTE_HALF, COURSE_LENGTH } from './terrain.js';

const SUN = { x: -0.52, y: 0.44, z: 0.73 };
const SNOW_LIT = [255, 249, 236];
const SNOW_SHADE = [146, 165, 209];
const ROCK_LIT = [146, 126, 111];
const ROCK_SHADE = [58, 60, 78];
const FOG = [233, 229, 230];
const FAR = 340;

const tmpN = { x: 0, y: 1, z: 0 };

function mix(a, b, t) {
  return a + (b - a) * t;
}

function hash(i, s) {
  let h = Math.imul(i | 0, 668265263) ^ Math.imul(s | 0, 374761393);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function buildRange(seed, count, base, spread) {
  const pts = [];
  const main = [];
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2 + (hash(i, seed) - 0.5) * 0.08;
    const peak = i % 2 === 1;
    const h = peak
      ? base * (0.4 + hash(i, seed + 7) * 0.6) * (1 + spread * (hash(i, seed + 3) - 0.5))
      : base * (0.04 + hash(i, seed + 11) * 0.14);
    main.push({ a, h, peak });
  }
  for (let i = 0; i < main.length; i++) {
    const p = main[i];
    const q = main[(i + 1) % main.length];
    pts.push({ a: p.a, h: p.h, peak: p.peak });
    let da = q.a - p.a;
    if (da < 0) da += Math.PI * 2;
    for (let k = 1; k < 4; k++) {
      const t = k / 4;
      const jag = (hash(i * 17 + k, seed + 23) - 0.42) * base * (p.peak ? 0.16 : 0.1);
      pts.push({ a: p.a + da * t, h: Math.max(0, p.h + (q.h - p.h) * t + jag), peak: false });
    }
  }
  return pts;
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.camX = 0;
    this.camY = 0;
    this.camZ = 0;
    this.camYaw = 0;
    this.shake = 0;
    this.ready = false;
    this.ranges = [
      { pts: buildRange(3, 22, 0.31, 0.6), par: 0.18, snow: [241, 241, 250], rock: [188, 192, 213], lit: [255, 246, 231] },
      { pts: buildRange(9, 18, 0.23, 0.75), par: 0.34, snow: [226, 230, 243], rock: [152, 155, 180], lit: [255, 238, 213] },
      { pts: buildRange(17, 14, 0.14, 0.8), par: 0.55, snow: [208, 215, 233], rock: [112, 115, 142], lit: [252, 224, 192] }
    ];
    this.puff = this.makePuff();
    this.resize();
  }

  makePuff() {
    const c = document.createElement('canvas');
    c.width = 64;
    c.height = 64;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255,255,255,0.95)');
    grad.addColorStop(0.4, 'rgba(250,251,255,0.55)');
    grad.addColorStop(1, 'rgba(240,246,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
    return c;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    this.w = Math.floor(window.innerWidth);
    this.h = Math.floor(window.innerHeight);
    this.canvas.width = Math.floor(this.w * dpr);
    this.canvas.height = Math.floor(this.h * dpr);
    this.canvas.style.width = this.w + 'px';
    this.canvas.style.height = this.h + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.sky = null;
  }

  updateCamera(rider, dt) {
    const speed = rider.speed;
    let heading = rider.yaw;
    if (speed > 2.5) {
      const vh = Math.atan2(rider.vx, rider.vz);
      const diff = ((vh - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
      heading += diff * (rider.airborne ? 0.85 : 0.45);
    }
    if (rider.crashTime > 0) heading = this.camYaw;

    const dyaw = ((heading - this.camYaw + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    this.camYaw += dyaw * (this.ready ? 1 - Math.exp(-dt * 4.5) : 1);

    const dist = 7.4 + speed * 0.085;
    const lift = 3.0 + speed * 0.022 + (rider.airborne ? 0.6 : 0);
    const tx = rider.x - Math.sin(this.camYaw) * dist;
    const tz = rider.z - Math.cos(this.camYaw) * dist;
    const ty = rider.y + lift;

    const kp = this.ready ? 1 - Math.exp(-dt * 6) : 1;
    this.camX = mix(this.camX, tx, kp);
    this.camY = mix(this.camY, ty, this.ready ? 1 - Math.exp(-dt * 4) : 1);
    this.camZ = mix(this.camZ, tz, kp);
    const floor = height(this.camX, this.camZ) + 1.7;
    if (this.camY < floor) this.camY = floor;
    this.ready = true;

    this.shake = Math.max(0, this.shake - dt * 3);

    const lookY = rider.y + 1.25;
    const fx = rider.x + Math.sin(this.camYaw) * 6 - this.camX;
    const fy = lookY - this.camY;
    const fz = rider.z + Math.cos(this.camYaw) * 6 - this.camZ;
    const fl = 1 / Math.hypot(fx, fy, fz);
    this.fx = fx * fl;
    this.fy = fy * fl;
    this.fz = fz * fl;

    const rl = 1 / Math.hypot(this.fz, this.fx);
    this.rx = this.fz * rl;
    this.ry = 0;
    this.rz = -this.fx * rl;

    this.ux = this.fy * this.rz - this.fz * this.ry;
    this.uy = this.fz * this.rx - this.fx * this.rz;
    this.uz = this.fx * this.ry - this.fy * this.rx;

    const fov = 1.05 + Math.min(0.3, speed * 0.0075);
    this.focal = (this.h * 0.5) / Math.tan(fov * 0.5);
    this.cx = this.w * 0.5 + (Math.random() - 0.5) * this.shake * 14;
    this.cy = this.h * 0.5 + (Math.random() - 0.5) * this.shake * 14;
  }

  proj(x, y, z, out) {
    const dx = x - this.camX;
    const dy = y - this.camY;
    const dz = z - this.camZ;
    const cz = dx * this.fx + dy * this.fy + dz * this.fz;
    if (cz < 0.4) {
      out.ok = false;
      return out;
    }
    const s = this.focal / cz;
    out.x = this.cx + (dx * this.rx + dy * this.ry + dz * this.rz) * s;
    out.y = this.cy - (dx * this.ux + dy * this.uy + dz * this.uz) * s;
    out.z = cz;
    out.s = s;
    out.ok = true;
    return out;
  }

  horizonY() {
    const l = Math.hypot(this.fx, this.fz) || 1;
    const dx = this.fx / l;
    const dz = this.fz / l;
    return this.cy - ((dx * this.ux + dz * this.uz) / (dx * this.fx + dz * this.fz)) * this.focal;
  }

  drawSky() {
    const ctx = this.ctx;
    if (!this.sky) {
      const g = ctx.createLinearGradient(0, 0, 0, this.h);
      g.addColorStop(0, '#4b649c');
      g.addColorStop(0.22, '#7d8dba');
      g.addColorStop(0.44, '#ac9ab6');
      g.addColorStop(0.62, '#dcae9c');
      g.addColorStop(0.78, '#f4cfa4');
      g.addColorStop(1, '#fbe6c6');
      this.sky = g;
    }
    ctx.fillStyle = this.sky;
    ctx.fillRect(0, 0, this.w, this.h);

    const hy = this.horizonY();
    const sunX = this.cx - Math.tan(Math.max(-1.2, Math.min(1.2, -0.62 - this.camYaw))) * this.focal;
    const sunY = hy - this.h * 0.14;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 7; i++) {
      const t = i / 7;
      const y = hy - this.h * (0.16 + t * 0.5);
      const alpha = 0.04 + 0.06 * (1 - t);
      const cw = this.w * (0.3 + hash(i, 5) * 0.5);
      const cxp = ((sunX + (hash(i, 9) - 0.5) * this.w * 1.6) % (this.w * 2)) - this.w * 0.2;
      ctx.fillStyle = `rgba(255,${210 + i * 5},${180 + i * 6},${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.ellipse(cxp, y, cw, this.h * (0.012 + hash(i, 3) * 0.016), 0, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    const glow = ctx.createRadialGradient(sunX, sunY, 6, sunX, sunY, this.h * 0.62);
    glow.addColorStop(0, 'rgba(255,250,232,0.95)');
    glow.addColorStop(0.08, 'rgba(255,238,200,0.55)');
    glow.addColorStop(0.34, 'rgba(255,222,178,0.22)');
    glow.addColorStop(1, 'rgba(255,214,170,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, this.w, this.h);

    for (const range of this.ranges) this.drawRange(hy, range);

    const band = ctx.createLinearGradient(0, hy - 1, 0, hy + this.h * 0.2);
    band.addColorStop(0, 'rgb(247,240,230)');
    band.addColorStop(1, `rgb(${FOG[0]},${FOG[1]},${FOG[2]})`);
    ctx.fillStyle = band;
    ctx.fillRect(0, hy - 1, this.w, this.h - hy + 1);
  }

  drawRange(hy, range) {
    const ctx = this.ctx;
    const pts = range.pts;
    const vis = [];
    for (let i = 0; i < pts.length; i++) {
      let d = ((pts[i].a - this.camYaw * range.par + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
      if (Math.abs(d) > 1.32) continue;
      vis.push({
        x: this.cx + Math.tan(d) * this.focal,
        y: hy - pts[i].h * this.h,
        peak: pts[i].peak,
        h: pts[i].h
      });
    }
    if (vis.length < 2) return;
    vis.sort((a, b) => a.x - b.x);

    let top = hy;
    for (const p of vis) if (p.y < top) top = p.y;

    ctx.beginPath();
    ctx.moveTo(vis[0].x, hy + 4);
    for (const p of vis) ctx.lineTo(p.x, p.y);
    ctx.lineTo(vis[vis.length - 1].x, hy + 4);
    ctx.closePath();
    const body = ctx.createLinearGradient(0, top, 0, hy + 4);
    body.addColorStop(0, `rgb(${range.rock[0]},${range.rock[1]},${range.rock[2]})`);
    body.addColorStop(0.55, `rgb(${mix(range.rock[0], FOG[0], 0.45) | 0},${mix(range.rock[1], FOG[1], 0.45) | 0},${mix(range.rock[2], FOG[2], 0.45) | 0})`);
    body.addColorStop(1, `rgb(${mix(range.rock[0], FOG[0], 0.92) | 0},${mix(range.rock[1], FOG[1], 0.92) | 0},${mix(range.rock[2], FOG[2], 0.92) | 0})`);
    ctx.fillStyle = body;
    ctx.fill();

    for (let i = 1; i < vis.length - 1; i++) {
      const p = vis[i];
      if (!p.peak) continue;
      const l = vis[i - 1];
      const r = vis[i + 1];
      const drop = (hy - p.y) * 0.42;

      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(l.x, l.y);
      ctx.lineTo(mix(l.x, p.x, 0.5), l.y + drop * 0.8);
      ctx.lineTo(p.x, p.y + drop);
      ctx.closePath();
      ctx.fillStyle = `rgba(${(range.rock[0] * 0.72) | 0},${(range.rock[1] * 0.74) | 0},${(range.rock[2] * 0.86) | 0},0.5)`;
      ctx.fill();

      const capL = { x: mix(p.x, l.x, 0.4), y: mix(p.y, l.y, 0.4) };
      const capR = { x: mix(p.x, r.x, 0.46), y: mix(p.y, r.y, 0.46) };
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(capR.x, capR.y);
      ctx.lineTo(mix(capR.x, p.x, 0.35), mix(capR.y, p.y, 0.3) + drop * 0.16);
      ctx.lineTo(mix(p.x, capL.x, 0.45), mix(p.y, capL.y, 0.5) + drop * 0.1);
      ctx.lineTo(capL.x, capL.y);
      ctx.closePath();
      ctx.fillStyle = `rgb(${range.snow[0]},${range.snow[1]},${range.snow[2]})`;
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(capR.x, capR.y);
      ctx.lineTo(mix(p.x, capR.x, 0.3), mix(p.y, capR.y, 0.8));
      ctx.closePath();
      ctx.fillStyle = `rgb(${range.lit[0]},${range.lit[1]},${range.lit[2]})`;
      ctx.fill();
    }
  }

  terrainColor(nx, ny, nz, x, z, dist, occ) {
    let lit = nx * SUN.x + ny * SUN.y + nz * SUN.z;
    lit = Math.max(0, Math.min(1, 0.12 + lit * 0.95)) * (1 - 0.5 * occ);
    const rocky = ny < 0.7 ? Math.min(1, (0.7 - ny) / 0.2) : 0;
    let r = mix(SNOW_SHADE[0], SNOW_LIT[0], lit);
    let g = mix(SNOW_SHADE[1], SNOW_LIT[1], lit);
    let b = mix(SNOW_SHADE[2], SNOW_LIT[2], lit);
    if (rocky > 0) {
      r = mix(r, mix(ROCK_SHADE[0], ROCK_LIT[0], lit), rocky);
      g = mix(g, mix(ROCK_SHADE[1], ROCK_LIT[1], lit), rocky);
      b = mix(b, mix(ROCK_SHADE[2], ROCK_LIT[2], lit), rocky);
    } else if (Math.abs(x) < PISTE_HALF) {
      const cord = Math.sin(x * 1.15) * 3 + Math.sin(z * 0.85) * 2;
      r += cord;
      g += cord;
      b += cord * 0.5;
    }
    const f = Math.min(1, dist / FAR);
    const ff = f * f * (1.7 - 0.7 * f);
    return `rgb(${mix(r, FOG[0], ff) | 0},${mix(g, FOG[1], ff) | 0},${mix(b, FOG[2], ff) | 0})`;
  }

  drawTerrain(rider, scenery) {
    const bands = [
      [2.5, -18, 52, 48],
      [5, 52, 130, 92],
      [9, 130, 232, 160],
      [15, 232, FAR + 30, 250]
    ];
    const px = rider.x;
    const pz = rider.z;
    const props = scenery.props;
    let ptr = props.length - 1;
    while (ptr >= 0 && props[ptr].z > this.camZ + FAR) ptr--;

    const a = { ok: false };
    const b = { ok: false };
    const c = { ok: false };
    const d = { ok: false };
    const ctx = this.ctx;

    for (let bi = bands.length - 1; bi >= 0; bi--) {
      const [step, z0, z1, halfW] = bands[bi];
      const startZ = Math.floor((pz + z0) / step) * step;
      const endZ = Math.floor((pz + z1) / step) * step;
      const startX = Math.floor((px - halfW) / step) * step;
      const cols = Math.ceil((halfW * 2) / step);
      let far = null;
      let near = new Float32Array(cols + 1);
      for (let i = 0; i <= cols; i++) near[i] = height(startX + i * step, endZ + step);

      for (let z = endZ; z >= startZ; z -= step) {
        far = near;
        near = new Float32Array(cols + 1);
        for (let i = 0; i <= cols; i++) near[i] = height(startX + i * step, z);

        for (let i = 0; i < cols; i++) {
          const x = startX + i * step;
          const h00 = near[i];
          const h10 = near[i + 1];
          const h01 = far[i];
          const h11 = far[i + 1];
          this.proj(x, h00, z, a);
          if (!a.ok) continue;
          this.proj(x + step, h10, z, b);
          if (!b.ok) continue;
          this.proj(x + step, h11, z + step, c);
          if (!c.ok) continue;
          this.proj(x, h01, z + step, d);
          if (!d.ok) continue;
          if (
            (a.x < 0 && b.x < 0 && c.x < 0 && d.x < 0) ||
            (a.x > this.w && b.x > this.w && c.x > this.w && d.x > this.w) ||
            (a.y > this.h && b.y > this.h && c.y > this.h && d.y > this.h)
          ) {
            continue;
          }
          let gx = (h10 + h11 - h00 - h01) / (2 * step);
          let gz = (h01 + h11 - h00 - h10) / (2 * step);
          let occ = 0;
          if (bi < 2) {
            const mx = x + step * 0.5;
            const mz = z + step * 0.5;
            const grain = bi === 0 ? 1 : 0.55;
            const w1 = Math.cos(mx * 0.26 + mz * 0.13);
            const w2 = Math.cos(mx * 0.11 - mz * 0.31);
            const w3 = Math.sin(mx * 0.44 - mz * 0.05);
            gx += (w1 * 0.109 + w2 * 0.033 - w3 * 0.053) * grain;
            gz += (w1 * 0.055 - w2 * 0.093 + w3 * 0.006) * grain;
            const mh = (h00 + h10 + h01 + h11) * 0.25;
            const s1 = height(mx - 3.2, mz + 4.5) - mh - 2.8;
            const s2 = height(mx - 8.1, mz + 11.4) - mh - 7.1;
            occ = Math.max(0, Math.min(1, Math.max(s1, s2) * 0.5));
          }
          const inv = 1 / Math.hypot(gx, 1, gz);
          ctx.fillStyle = this.terrainColor(-gx * inv, inv, -gz * inv, x, z, (a.z + c.z) * 0.5, occ);
          ctx.strokeStyle = ctx.fillStyle;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.lineTo(c.x, c.y);
          ctx.lineTo(d.x, d.y);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }

        while (ptr >= 0 && props[ptr].z >= z) {
          const p = props[ptr--];
          if (p.kind === 0) this.drawTree(p.o);
          else if (p.kind === 1) this.drawRock(p.o);
          else this.drawBanner(p.o);
        }
      }
    }
  }

  drawTrail(trail) {
    const ctx = this.ctx;
    const a = { ok: false };
    const b = { ok: false };
    const c = { ok: false };
    const d = { ok: false };
    for (let i = trail.length - 1; i > 0; i--) {
      const p = trail[i];
      const q = trail[i - 1];
      const dz = p.z - this.camZ;
      const dx = p.x - this.camX;
      if (dx * dx + dz * dz > 4900) continue;
      const w = 0.16 + p.skid * 0.55;
      this.proj(p.x - p.rx * w, p.y + 0.02, p.z - p.rz * w, a);
      this.proj(p.x + p.rx * w, p.y + 0.02, p.z + p.rz * w, b);
      this.proj(q.x + q.rx * w, q.y + 0.02, q.z + q.rz * w, c);
      this.proj(q.x - q.rx * w, q.y + 0.02, q.z - q.rz * w, d);
      if (!a.ok || !b.ok || !c.ok || !d.ok) continue;
      ctx.fillStyle = `rgba(126,150,196,${(0.16 + p.skid * 0.2).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineTo(c.x, c.y);
      ctx.lineTo(d.x, d.y);
      ctx.closePath();
      ctx.fill();
    }
  }

  drawTree(t) {
    const ctx = this.ctx;
    const base = { ok: false };
    this.proj(t.x, t.y, t.z, base);
    if (!base.ok) return;
    const top = { ok: false };
    this.proj(t.x, t.y + t.h, t.z, top);
    if (!top.ok) return;
    const wpx = t.w * base.s;
    if (wpx < 0.6) return;
    const fog = Math.min(0.9, (base.z / FAR) ** 1.5);
    const r = mix(26 + t.tone * 14, FOG[0], fog) | 0;
    const g = mix(48 + t.tone * 22, FOG[1], fog) | 0;
    const b = mix(52 + t.tone * 16, FOG[2], fog) | 0;
    ctx.strokeStyle = `rgba(72,54,44,${(1 - fog).toFixed(2)})`;
    ctx.lineWidth = Math.max(1, wpx * 0.2);
    ctx.beginPath();
    ctx.moveTo(base.x, base.y);
    ctx.lineTo(top.x, top.y + (base.y - top.y) * 0.25);
    ctx.stroke();
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    for (let i = 0; i < 3; i++) {
      const f0 = 0.16 + (i / 3) * 0.76;
      const f1 = Math.min(1, f0 + 0.42);
      const y0 = mix(base.y, top.y, f0);
      const y1 = mix(base.y, top.y, f1);
      const x0 = mix(base.x, top.x, f0);
      const x1 = mix(base.x, top.x, f1);
      const spread = wpx * (1.25 - i * 0.3);
      ctx.beginPath();
      ctx.moveTo(x0 - spread, y0);
      ctx.lineTo(x0 + spread, y0);
      ctx.lineTo(x1, y1);
      ctx.closePath();
      ctx.fill();
    }
    if (fog < 0.6) {
      ctx.fillStyle = `rgba(255,238,214,${(0.45 - fog * 0.5).toFixed(2)})`;
      const xa = mix(base.x, top.x, 0.72);
      ctx.beginPath();
      ctx.moveTo(xa - wpx * 0.45, mix(base.y, top.y, 0.72));
      ctx.lineTo(xa - wpx * 0.05, mix(base.y, top.y, 0.72));
      ctx.lineTo(top.x, top.y);
      ctx.closePath();
      ctx.fill();
    }
  }

  drawRock(r) {
    const ctx = this.ctx;
    const p = { ok: false };
    this.proj(r.x, r.y + r.r * 0.35, r.z, p);
    if (!p.ok) return;
    const rad = r.r * p.s;
    if (rad < 1) return;
    const fog = Math.min(0.85, (p.z / FAR) ** 1.5);
    ctx.fillStyle = `rgb(${mix(74, FOG[0], fog) | 0},${mix(72, FOG[1], fog) | 0},${mix(84, FOG[2], fog) | 0})`;
    ctx.beginPath();
    ctx.moveTo(p.x - rad, p.y + rad * 0.5);
    ctx.lineTo(p.x - rad * 0.35, p.y - rad * 0.75);
    ctx.lineTo(p.x + rad * 0.4, p.y - rad * 0.5);
    ctx.lineTo(p.x + rad, p.y + rad * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = `rgba(255,240,215,${(0.72 - fog * 0.7).toFixed(2)})`;
    ctx.beginPath();
    ctx.moveTo(p.x - rad * 0.35, p.y - rad * 0.75);
    ctx.lineTo(p.x + rad * 0.4, p.y - rad * 0.5);
    ctx.lineTo(p.x + rad * 0.15, p.y - rad * 0.2);
    ctx.closePath();
    ctx.fill();
  }

  drawBanner(n) {
    const ctx = this.ctx;
    const left = { ok: false };
    const right = { ok: false };
    const lt = { ok: false };
    const rt = { ok: false };
    this.proj(-PISTE_HALF + 2, n.y + 0.2, n.z, left);
    this.proj(PISTE_HALF - 2, n.y + 0.2, n.z, right);
    this.proj(-PISTE_HALF + 2, n.y + 5.2, n.z, lt);
    this.proj(PISTE_HALF - 2, n.y + 5.2, n.z, rt);
    if (!left.ok || !right.ok || !lt.ok || !rt.ok) return;
    ctx.strokeStyle = 'rgba(58,60,74,0.85)';
    ctx.lineWidth = Math.max(1, 0.16 * left.s);
    ctx.beginPath();
    ctx.moveTo(left.x, left.y);
    ctx.lineTo(lt.x, lt.y);
    ctx.moveTo(right.x, right.y);
    ctx.lineTo(rt.x, rt.y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(214,84,52,0.94)';
    ctx.beginPath();
    ctx.moveTo(lt.x, lt.y);
    ctx.lineTo(rt.x, rt.y);
    ctx.lineTo(rt.x, rt.y + 1.1 * rt.s);
    ctx.lineTo(lt.x, lt.y + 1.1 * lt.s);
    ctx.closePath();
    ctx.fill();
  }

  drawFinish() {
    const ctx = this.ctx;
    const y = height(0, COURSE_LENGTH);
    const a = { ok: false };
    const b = { ok: false };
    this.proj(-PISTE_HALF, y, COURSE_LENGTH, a);
    this.proj(PISTE_HALF, y, COURSE_LENGTH, b);
    if (!a.ok || !b.ok) return;
    const w = Math.abs(b.x - a.x);
    const cell = w / 26;
    for (let i = 0; i < 26; i++) {
      for (let j = 0; j < 2; j++) {
        ctx.fillStyle = (i + j) % 2 ? '#fdf6e8' : '#2b2b38';
        ctx.fillRect(Math.min(a.x, b.x) + i * cell, mix(a.y, b.y, i / 26) + j * cell, cell + 1, cell + 1);
      }
    }
  }

  drawRider(rider) {
    const ctx = this.ctx;
    const cp = Math.cos(rider.pitch);
    const fx = Math.sin(rider.yaw) * cp;
    const fy = -Math.sin(rider.pitch);
    const fz = Math.cos(rider.yaw) * cp;

    if (rider.airborne || rider.crashTime > 0) {
      tmpN.x = 0;
      tmpN.y = 1;
      tmpN.z = 0;
    } else {
      normal(rider.x, rider.z, tmpN);
    }
    const ux = tmpN.x;
    const uy = tmpN.y;
    const uz = tmpN.z;

    let rx = uy * fz - uz * fy;
    let ry = uz * fx - ux * fz;
    let rz = ux * fy - uy * fx;
    const rl = 1 / Math.max(Math.hypot(rx, ry, rz), 1e-6);
    rx *= rl;
    ry *= rl;
    rz *= rl;

    const roll = rider.edge + (rider.crashTime > 0 ? Math.sin(rider.tumble) * 1.2 : 0);
    const cr = Math.cos(roll);
    const sr = Math.sin(roll);
    const bux = ux * cr - rx * sr;
    const buy = uy * cr - ry * sr;
    const buz = uz * cr - rz * sr;
    const brx = rx * cr + ux * sr;
    const bry = ry * cr + uy * sr;
    const brz = rz * cr + uz * sr;

    const bx = rider.x + bux * 0.05;
    const by = rider.y + buy * 0.05;
    const bz = rider.z + buz * 0.05;

    const P = (dx, dy, dz, out) =>
      this.proj(bx + fx * dx + brx * dy + bux * dz, by + fy * dx + bry * dy + buy * dz, bz + fz * dx + brz * dy + buz * dz, out);

    const c0 = { ok: false };
    const c1 = { ok: false };
    const c2 = { ok: false };
    const c3 = { ok: false };
    P(0.78, 0.14, 0, c0);
    P(0.78, -0.14, 0, c1);
    P(-0.78, -0.14, 0, c2);
    P(-0.78, 0.14, 0, c3);
    const centre = { ok: false };
    P(0, 0, 0, centre);
    if (!centre.ok || !c0.ok || !c1.ok || !c2.ok || !c3.ok) return;
    const s = centre.s;

    ctx.save();
    ctx.beginPath();
    ctx.ellipse(centre.x, centre.y + 0.05 * s, 0.8 * s, 0.22 * s, 0, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(92,110,150,0.3)';
    ctx.fill();

    const crouch = rider.crouch;
    const lean = Math.sin(roll) * 0.5;
    const hipZ = 0.62 - crouch * 0.24;
    const shZ = hipZ + 0.5 - crouch * 0.1;
    const hip = { ok: false };
    const sho = { ok: false };
    const head = { ok: false };
    const pack = { ok: false };
    const fA = { ok: false };
    const fB = { ok: false };
    const hand0 = { ok: false };
    const hand1 = { ok: false };
    P(0.02, lean * 0.7, hipZ, hip);
    P(0.1, lean, shZ, sho);
    P(0.13, lean * 1.05, shZ + 0.3, head);
    P(-0.1, lean * 1.05, shZ + 0.02, pack);
    P(0.3, 0, 0.06, fA);
    P(-0.3, 0, 0.06, fB);
    const out = rider.grabbing ? 0.12 : 0.34;
    const down = rider.grabbing ? -0.5 : -0.2;
    P(0.3, lean + out, shZ + down, hand0);
    P(-0.16, lean - out * 0.8, shZ + down * 0.5, hand1);
    if (!hip.ok || !sho.ok || !head.ok) {
      ctx.restore();
      return;
    }

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (fA.ok && fB.ok) {
      ctx.strokeStyle = '#20232e';
      ctx.lineWidth = Math.max(2, 0.15 * s);
      ctx.beginPath();
      ctx.moveTo(fA.x, fA.y);
      ctx.lineTo(hip.x, hip.y);
      ctx.moveTo(fB.x, fB.y);
      ctx.lineTo(hip.x, hip.y);
      ctx.stroke();
    }

    if (pack.ok) {
      ctx.fillStyle = '#2c2f3c';
      ctx.beginPath();
      ctx.ellipse(pack.x, pack.y, 0.19 * s, 0.24 * s, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    const jw = 0.19 * s;
    ctx.beginPath();
    ctx.moveTo(hip.x - jw * 0.8, hip.y);
    ctx.lineTo(hip.x + jw * 0.8, hip.y);
    ctx.lineTo(sho.x + jw, sho.y);
    ctx.lineTo(sho.x - jw, sho.y);
    ctx.closePath();
    const jacket = ctx.createLinearGradient(sho.x - jw, sho.y, sho.x + jw, hip.y);
    jacket.addColorStop(0, '#f27a2a');
    jacket.addColorStop(0.55, '#e05316');
    jacket.addColorStop(1, '#a8360f');
    ctx.fillStyle = jacket;
    ctx.fill();

    if (hand0.ok && hand1.ok) {
      ctx.strokeStyle = '#e2601c';
      ctx.lineWidth = Math.max(1.4, 0.105 * s);
      ctx.beginPath();
      ctx.moveTo(hand0.x, hand0.y);
      ctx.lineTo(sho.x, sho.y);
      ctx.lineTo(hand1.x, hand1.y);
      ctx.stroke();
      ctx.fillStyle = '#23262f';
      ctx.beginPath();
      ctx.arc(hand0.x, hand0.y, Math.max(1.2, 0.07 * s), 0, Math.PI * 2);
      ctx.arc(hand1.x, hand1.y, Math.max(1.2, 0.07 * s), 0, Math.PI * 2);
      ctx.fill();
    }

    const hr = Math.max(2.2, 0.16 * s);
    ctx.fillStyle = '#1c1f28';
    ctx.beginPath();
    ctx.arc(head.x, head.y, hr, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,236,196,0.85)';
    ctx.beginPath();
    ctx.arc(head.x, head.y - hr * 0.35, hr, Math.PI * 1.15, Math.PI * 1.75);
    ctx.fill();
    const gog = ctx.createLinearGradient(head.x - hr, head.y, head.x + hr, head.y);
    gog.addColorStop(0, '#8fd8f2');
    gog.addColorStop(1, '#3f7fa8');
    ctx.fillStyle = gog;
    ctx.beginPath();
    ctx.ellipse(head.x + hr * 0.25, head.y + hr * 0.05, hr * 0.62, hr * 0.4, 0.1, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(c0.x, c0.y);
    ctx.lineTo(c1.x, c1.y);
    ctx.lineTo(c2.x, c2.y);
    ctx.lineTo(c3.x, c3.y);
    ctx.closePath();
    const deck = ctx.createLinearGradient(c0.x, c0.y, c2.x, c2.y);
    deck.addColorStop(0, '#fdf3e2');
    deck.addColorStop(0.45, '#2f8fd0');
    deck.addColorStop(1, '#15304d');
    ctx.fillStyle = deck;
    ctx.fill();
    ctx.strokeStyle = 'rgba(18,24,34,0.9)';
    ctx.lineWidth = Math.max(1, 0.035 * s);
    ctx.stroke();
    ctx.restore();
  }

  drawParticles(list, soft) {
    const ctx = this.ctx;
    const p = { ok: false };
    for (let i = 0; i < list.length; i++) {
      const q = list[i];
      const fluffy = q.soft === undefined ? soft : q.soft;
      this.proj(q.x, q.y, q.z, p);
      if (!p.ok || p.z < (fluffy ? 2.2 : 1.2)) continue;
      const r = Math.min(q.r * p.s, fluffy ? 60 : 4);
      if (r < 0.5) continue;
      const alpha = (q.life / q.max) * q.a;
      if (fluffy) {
        ctx.globalAlpha = Math.min(1, alpha);
        ctx.drawImage(this.puff, p.x - r, p.y - r, r * 2, r * 2);
      } else {
        ctx.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  drawSpeedLines(speed) {
    const t = Math.max(0, (speed - 16) / 22);
    if (t <= 0) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = Math.min(0.42, t * 0.42);
    const g = ctx.createRadialGradient(
      this.cx,
      this.cy,
      this.h * 0.34,
      this.cx,
      this.cy,
      this.h * (0.95 - t * 0.18)
    );
    g.addColorStop(0, 'rgba(255,255,255,0)');
    g.addColorStop(1, 'rgba(255,241,222,0.95)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, this.w, this.h);
    ctx.restore();
  }

  render(rider, scenery, particles, snow, trail, dt) {
    this.updateCamera(rider, dt);
    this.drawSky();
    this.drawTerrain(rider, scenery);
    this.drawTrail(trail);
    if (rider.z > COURSE_LENGTH - 260) this.drawFinish();
    this.drawParticles(particles, true);
    this.drawRider(rider);
    this.drawParticles(snow, false);
    this.drawSpeedLines(rider.speed);
  }
}
