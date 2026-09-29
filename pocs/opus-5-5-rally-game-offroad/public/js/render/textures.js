import * as THREE from 'three';
import { createNoise2D, fbm, mulberry32, clamp } from '../core/math.js';

function canvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function finish(c, repeat = true, color = true) {
  const t = new THREE.CanvasTexture(c);
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (color) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function heightToNormal(heights, w, h, strength) {
  const c = canvas(w, h);
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const l = heights[y * w + ((x - 1 + w) % w)];
      const r = heights[y * w + ((x + 1) % w)];
      const u = heights[((y - 1 + h) % h) * w + x];
      const d = heights[((y + 1) % h) * w + x];
      let nx = (l - r) * strength;
      let ny = (u - d) * strength;
      const len = Math.hypot(nx, ny, 1);
      const i = (y * w + x) * 4;
      img.data[i] = ((nx / len) * 0.5 + 0.5) * 255;
      img.data[i + 1] = ((ny / len) * 0.5 + 0.5) * 255;
      img.data[i + 2] = (1 / len) * 255;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return finish(c, true, false);
}

function tileNoise(noise, x, y, w, scale, h = w) {
  const sx = scale;
  const sy = scale * (h / w);
  const a = fbm(noise, (x / w) * sx, (y / h) * sy, 4);
  const b = fbm(noise, ((x - w) / w) * sx, (y / h) * sy, 4);
  const c = fbm(noise, (x / w) * sx, ((y - h) / h) * sy, 4);
  const d = fbm(noise, ((x - w) / w) * sx, ((y - h) / h) * sy, 4);
  const tx = x / w;
  const ty = y / h;
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

export function mudRoadTextures(seed, weather) {
  const wet = weather !== 'clear';
  const snow = weather === 'snow';
  const W = 256;
  const H = 512;
  const noise = createNoise2D(seed);
  const rand = mulberry32(seed);
  const heights = new Float32Array(W * H);
  const col = canvas(W, H);
  const rough = canvas(W, H);
  const ctx = col.getContext('2d');
  const rctx = rough.getContext('2d');
  const img = ctx.createImageData(W, H);
  const rimg = rctx.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = x / W;
      const n = tileNoise(noise, x, y, W, 6, H) * 0.6 + tileNoise(noise, x, y, W, 20, H) * 0.4;
      const rut = Math.exp(-((u - 0.3) ** 2) / 0.0018) + Math.exp(-((u - 0.7) ** 2) / 0.0018);
      const tread = rut * (rand() < 0.08 ? 1 : 0);
      const edge = Math.min(u, 1 - u);
      const puddle = clamp((tileNoise(noise, x, y, W, 3, H) - 0.28) * 6, 0, 1) * (wet ? 1 : 0.35);
      const h = n * 0.6 - rut * 0.7 + tread * 0.15 - puddle * 0.3 + (edge < 0.06 ? 0.4 * (0.06 - edge) * 10 : 0);
      heights[y * W + x] = h;
      const base = 0.62 + n * 0.5 - rut * 0.25;
      const gravel = rand() < 0.035 ? 0.35 : 0;
      const i = (y * W + x) * 4;
      const wetDark = wet ? 0.72 : 1;
      const slush = snow ? clamp(0.25 + n * 1.4 - rut * 0.9 - puddle, 0, 0.9) : 0;
      img.data[i] = clamp((96 * base + gravel * 120) * wetDark * (1 - puddle * 0.35) * (1 - slush) + 232 * slush, 0, 255);
      img.data[i + 1] = clamp((72 * base + gravel * 110) * wetDark * (1 - puddle * 0.3) * (1 - slush) + 236 * slush, 0, 255);
      img.data[i + 2] = clamp((52 * base + gravel * 100) * wetDark * (1 - puddle * 0.2) * (1 - slush) + 240 * slush, 0, 255);
      img.data[i + 3] = 255;
      const r = clamp((wet ? 0.55 : 0.92) - puddle * 0.7 - rut * (wet ? 0.2 : 0.05), 0.04, 1) * 255;
      rimg.data[i] = r;
      rimg.data[i + 1] = r;
      rimg.data[i + 2] = r;
      rimg.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  rctx.putImageData(rimg, 0, 0);
  return { map: finish(col), roughnessMap: finish(rough, true, false), normalMap: heightToNormal(heights, W, H, 5) };
}

export function groundDetailTextures(seed) {
  const S = 256;
  const noise = createNoise2D(seed + 3);
  const rand = mulberry32(seed + 9);
  const heights = new Float32Array(S * S);
  const c = canvas(S, S);
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(S, S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const n = tileNoise(noise, x, y, S, 8);
      const blade = rand();
      const v = 0.78 + n * 0.35 + (blade > 0.85 ? 0.12 : blade < 0.1 ? -0.12 : 0);
      heights[y * S + x] = n + blade * 0.25;
      const i = (y * S + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = clamp(v * 255, 0, 255);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return { map: finish(c), normalMap: heightToNormal(heights, S, S, 3) };
}

export function waterNormal(seed) {
  const S = 256;
  const noise = createNoise2D(seed + 77);
  const heights = new Float32Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) heights[y * S + x] = tileNoise(noise, x, y, S, 10);
  return heightToNormal(heights, S, S, 6);
}

export function camoTexture(base) {
  const S = 512;
  const c = canvas(S, S);
  const ctx = c.getContext('2d');
  const noise = createNoise2D(4242);
  const noise2 = createNoise2D(4343);
  const color = new THREE.Color(base);
  const shades = [color.clone().multiplyScalar(1), color.clone().multiplyScalar(0.62), new THREE.Color('#3b3a2a'), new THREE.Color('#8a7f5c')];
  const img = ctx.createImageData(S, S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const a = tileNoise(noise, x, y, S, 5);
      const b = tileNoise(noise2, x, y, S, 7);
      const k = a > 0.12 ? 1 : b > 0.15 ? 2 : a < -0.18 ? 3 : 0;
      const s = shades[k];
      const i = (y * S + x) * 4;
      img.data[i] = s.r * 255;
      img.data[i + 1] = s.g * 255;
      img.data[i + 2] = s.b * 255;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const t = finish(c);
  t.repeat.set(2, 2);
  return t;
}

export function carbonTexture(base) {
  const S = 128;
  const c = canvas(S, S);
  const ctx = c.getContext('2d');
  const color = new THREE.Color(base);
  const dark = color.clone().multiplyScalar(0.25);
  const light = color.clone().multiplyScalar(0.55);
  const cell = 16;
  for (let y = 0; y < S; y += cell) {
    for (let x = 0; x < S; x += cell) {
      const flip = ((x + y) / cell) % 2 === 0;
      const g = flip ? ctx.createLinearGradient(x, y, x + cell, y) : ctx.createLinearGradient(x, y, x, y + cell);
      g.addColorStop(0, `#${dark.getHexString()}`);
      g.addColorStop(0.5, `#${light.getHexString()}`);
      g.addColorStop(1, `#${dark.getHexString()}`);
      ctx.fillStyle = g;
      ctx.fillRect(x, y, cell, cell);
    }
  }
  const t = finish(c);
  t.repeat.set(10, 10);
  return t;
}

export function tireTexture() {
  const c = canvas(64, 256);
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#1b1a19';
  ctx.fillRect(0, 0, 64, 256);
  for (let y = 0; y < 256; y += 16) {
    ctx.fillStyle = '#0c0c0c';
    ctx.fillRect(4, y, 24, 7);
    ctx.fillRect(36, y + 8, 24, 7);
    ctx.fillStyle = '#2a2826';
    ctx.fillRect(0, y + 3, 3, 10);
    ctx.fillRect(61, y + 11, 3, 10);
  }
  const t = finish(c);
  t.repeat.set(1, 3);
  return t;
}

export function softDot() {
  const c = canvas(64, 64);
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.4, 'rgba(255,255,255,0.7)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  return finish(c, false);
}

export function grassBlades(tint) {
  const c = canvas(128, 128);
  const ctx = c.getContext('2d');
  const rand = mulberry32(99);
  const base = new THREE.Color(tint);
  for (let k = 0; k < 70; k++) {
    const x = 10 + rand() * 108;
    const h = 50 + rand() * 75;
    const lean = (rand() - 0.5) * 30;
    const shade = base.clone().multiplyScalar(0.6 + rand() * 0.7);
    ctx.strokeStyle = `#${shade.getHexString()}`;
    ctx.lineWidth = 1.5 + rand() * 2;
    ctx.beginPath();
    ctx.moveTo(x, 128);
    ctx.quadraticCurveTo(x + lean * 0.3, 128 - h * 0.6, x + lean, 128 - h);
    ctx.stroke();
  }
  return finish(c, false);
}

export function windowGrid(seed) {
  const c = canvas(64, 256);
  const ctx = c.getContext('2d');
  const rand = mulberry32(seed);
  ctx.fillStyle = '#39424c';
  ctx.fillRect(0, 0, 64, 256);
  for (let y = 4; y < 256; y += 8) {
    for (let x = 4; x < 64; x += 8) {
      const lit = rand();
      ctx.fillStyle = lit > 0.7 ? '#e9d9a8' : lit > 0.35 ? '#7f95a8' : '#28313a';
      ctx.fillRect(x, y, 5, 5);
    }
  }
  return finish(c);
}

export function waterfallTexture() {
  const c = canvas(64, 256);
  const ctx = c.getContext('2d');
  const rand = mulberry32(5);
  for (let x = 0; x < 64; x++) {
    const a = 0.35 + rand() * 0.6;
    const g = ctx.createLinearGradient(0, 0, 0, 256);
    g.addColorStop(0, `rgba(255,255,255,${a})`);
    g.addColorStop(0.5, `rgba(230,240,248,${a * 0.7})`);
    g.addColorStop(1, `rgba(255,255,255,${a})`);
    ctx.fillStyle = g;
    ctx.fillRect(x, 0, 1, 256);
  }
  const t = finish(c);
  t.wrapS = THREE.ClampToEdgeWrapping;
  return t;
}
