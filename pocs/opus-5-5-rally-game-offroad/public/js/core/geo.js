export const PLACES = [
  { id: 'sf', lat: 37.8030, lon: -122.4620 },
  { id: 'la', lat: 34.1420, lon: -118.2860 },
  { id: 'tahoe', lat: 38.9370, lon: -120.0420 },
  { id: 'yosemite', lat: 37.7390, lon: -119.6050 },
];

export const NEAR = { name: 'near', half: 800, n: 401, demZoom: 15, imgZoom: 15 };
export const FAR = { name: 'far', half: 12000, n: 385, demZoom: 12, imgZoom: 13 };

const M_PER_DEG_LAT = 110540;
const M_PER_DEG_LON = 111320;

export function placeOf(id) {
  return PLACES.find((p) => p.id === id);
}

export function toLatLon(place, x, z) {
  const lat = place.lat - z / M_PER_DEG_LAT;
  const lon = place.lon + x / (M_PER_DEG_LON * Math.cos((place.lat * Math.PI) / 180));
  return [lat, lon];
}

export function toLocal(place, lat, lon) {
  return [(lon - place.lon) * M_PER_DEG_LON * Math.cos((place.lat * Math.PI) / 180), -(lat - place.lat) * M_PER_DEG_LAT];
}

export function mercatorUV(place, x, z, imagery) {
  const [lat, lon] = toLatLon(place, x, z);
  const n = 2 ** imagery.zoom;
  const tx = ((lon + 180) / 360) * n;
  const r = (lat * Math.PI) / 180;
  const ty = ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n;
  return [(tx - imagery.x0) / imagery.cols, 1 - (ty - imagery.y0) / imagery.rows];
}

function grid(level, raw) {
  const h = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) h[i] = raw[i] / 4;
  const cell = (2 * level.half) / (level.n - 1);
  return {
    level,
    heights: h,
    at(x, z) {
      const fx = Math.min(Math.max((x + level.half) / cell, 0), level.n - 1.0001);
      const fz = Math.min(Math.max((z + level.half) / cell, 0), level.n - 1.0001);
      const x0 = Math.floor(fx);
      const z0 = Math.floor(fz);
      const tx = fx - x0;
      const tz = fz - z0;
      const i = z0 * level.n + x0;
      const a = h[i] + (h[i + 1] - h[i]) * tx;
      const b = h[i + level.n] + (h[i + level.n + 1] - h[i + level.n]) * tx;
      return a + (b - a) * tz;
    },
  };
}

export function createGeo(place, nearRaw, farRaw) {
  const near = grid(NEAR, nearRaw);
  const far = grid(FAR, farRaw);
  const inner = NEAR.half - 40;
  return {
    place,
    near,
    far,
    elevation(x, z) {
      const d = Math.max(Math.abs(x), Math.abs(z));
      if (d <= inner) return near.at(x, z);
      if (d >= NEAR.half) return far.at(x, z);
      const t = (d - inner) / 40;
      return near.at(x, z) * (1 - t) + far.at(x, z) * t;
    },
  };
}
