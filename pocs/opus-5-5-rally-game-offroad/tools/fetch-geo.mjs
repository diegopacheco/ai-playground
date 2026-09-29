import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PLACES, NEAR, FAR, toLatLon } from '../public/js/core/geo.js';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const OUT = join(ROOT, 'public', 'geo');
const UA = { 'User-Agent': 'california-offroad-rally/1.0 (terrain build)' };
const DEM_URL = (z, x, y) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
const IMG_URL = (z, x, y) => `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/${z}/${y}/${x}.jpg`;

function tileXY(lat, lon, z) {
  const n = 2 ** z;
  const x = ((lon + 180) / 360) * n;
  const r = (lat * Math.PI) / 180;
  const y = ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n;
  return [x, y];
}

export function decodePng(buf) {
  let i = 8;
  const idat = [];
  let width = 0;
  let height = 0;
  let channels = 3;
  while (i < buf.length) {
    const len = buf.readUInt32BE(i);
    const type = buf.toString('ascii', i + 4, i + 8);
    const data = buf.subarray(i + 8, i + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      channels = { 2: 3, 6: 4 }[data[9]];
      if (data[8] !== 8 || !channels) throw new Error('unsupported png');
    }
    if (type === 'IDAT') idat.push(data);
    i += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = Buffer.alloc(height * stride);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? out[y * stride + x - channels] : 0;
      const b = y > 0 ? out[(y - 1) * stride + x] : 0;
      const c = x >= channels && y > 0 ? out[(y - 1) * stride + x - channels] : 0;
      let v = line[x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      out[y * stride + x] = v & 255;
    }
  }
  return { width, height, channels, data: out };
}

async function cached(url, file) {
  if (existsSync(file)) return readFile(file);
  for (let attempt = 0; attempt < 4; attempt++) {
    const r = await fetch(url, { headers: UA });
    if (r.ok) {
      const b = Buffer.from(await r.arrayBuffer());
      await writeFile(file, b);
      return b;
    }
    await new Promise((res) => setTimeout(res, 800 * (attempt + 1)));
  }
  throw new Error(`failed ${url}`);
}

async function demGrid(place, level, cacheDir) {
  const tiles = new Map();
  const heights = new Int16Array(level.n * level.n);
  for (let gz = 0; gz < level.n; gz++) {
    for (let gx = 0; gx < level.n; gx++) {
      const x = -level.half + (gx * 2 * level.half) / (level.n - 1);
      const z = -level.half + (gz * 2 * level.half) / (level.n - 1);
      const [lat, lon] = toLatLon(place, x, z);
      const [tx, ty] = tileXY(lat, lon, level.demZoom);
      const key = `${Math.floor(tx)}_${Math.floor(ty)}`;
      if (!tiles.has(key)) {
        const buf = await cached(DEM_URL(level.demZoom, Math.floor(tx), Math.floor(ty)), join(cacheDir, `dem_${level.demZoom}_${key}.png`));
        tiles.set(key, decodePng(buf));
      }
      const img = tiles.get(key);
      const px = Math.min(255, Math.floor((tx % 1) * 256));
      const py = Math.min(255, Math.floor((ty % 1) * 256));
      const o = (py * 256 + px) * img.channels;
      const h = img.data[o] * 256 + img.data[o + 1] + img.data[o + 2] / 256 - 32768;
      heights[gz * level.n + gx] = Math.round(h * 4);
    }
  }
  return heights;
}

async function imagery(place, level, dir) {
  const [x0, y0] = tileXY(...toLatLon(place, -level.half, -level.half), level.imgZoom);
  const [x1, y1] = tileXY(...toLatLon(place, level.half, level.half), level.imgZoom);
  const tiles = [];
  for (let ty = Math.floor(y0); ty <= Math.floor(y1); ty++) {
    for (let tx = Math.floor(x0); tx <= Math.floor(x1); tx++) {
      const name = `${level.name}_${tx}_${ty}.jpg`;
      await cached(IMG_URL(level.imgZoom, tx, ty), join(dir, name));
      tiles.push({ name, x: tx, y: ty });
    }
  }
  return { zoom: level.imgZoom, x0: Math.floor(x0), y0: Math.floor(y0), cols: Math.floor(x1) - Math.floor(x0) + 1, rows: Math.floor(y1) - Math.floor(y0) + 1, tiles: tiles.map((t) => t.name) };
}

for (const place of PLACES) {
  const dir = join(OUT, place.id);
  const cacheDir = join(ROOT, '.geo-cache');
  await mkdir(dir, { recursive: true });
  await mkdir(cacheDir, { recursive: true });
  const manifest = { id: place.id, imagery: {} };
  for (const level of [NEAR, FAR]) {
    const heights = await demGrid(place, level, cacheDir);
    await writeFile(join(dir, `${level.name}.bin`), Buffer.from(heights.buffer));
    manifest.imagery[level.name] = await imagery(place, level, dir);
  }
  await writeFile(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2));
  console.log(`${place.id}: done`);
}
