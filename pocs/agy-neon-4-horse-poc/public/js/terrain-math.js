function fract(x) {
  return x - Math.floor(x);
}

export function hash2d(ix, iz) {
  const n = Math.sin(ix * 127.1 + iz * 311.7) * 43758.5453123;
  return fract(n);
}

export function valueNoise2d(x, z) {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uz = fz * fz * (3 - 2 * fz);
  const a = hash2d(ix, iz);
  const b = hash2d(ix + 1, iz);
  const c = hash2d(ix, iz + 1);
  const d = hash2d(ix + 1, iz + 1);
  return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}

export function fbm2d(x, z, octaves = 4) {
  let value = 0;
  let amplitude = 0.5;
  let frequency = 1;
  let norm = 0;
  for (let i = 0; i < octaves; i += 1) {
    value += amplitude * valueNoise2d(x * frequency, z * frequency);
    norm += amplitude;
    amplitude *= 0.5;
    frequency *= 2.03;
  }
  return value / norm;
}

export function getTrailCenterX(worldZ) {
  return Math.sin(worldZ * 0.011) * 5.8 + Math.sin(worldZ * 0.027 + 1.1) * 2.1;
}

export function getTrailSlopeX(worldZ) {
  const eps = 0.35;
  return (getTrailCenterX(worldZ + eps) - getTrailCenterX(worldZ - eps)) / (2 * eps);
}

export function getTrailMask(worldX, worldZ) {
  const dist = Math.abs(worldX - getTrailCenterX(worldZ));
  const trailHalfWidth = 2.15;
  const blendWidth = 2.6;
  if (dist <= trailHalfWidth) {
    return 1;
  }
  if (dist >= trailHalfWidth + blendWidth) {
    return 0;
  }
  const t = (dist - trailHalfWidth) / blendWidth;
  return 1 - t * t * (3 - 2 * t);
}

export function getTerrainHeight(worldX, worldZ) {
  const trailX = getTrailCenterX(worldZ);
  const lateral = Math.abs(worldX - trailX);
  const valleyRoll =
    Math.sin(worldZ * 0.016) * 0.85 +
    Math.cos(worldZ * 0.0075 + 0.6) * 1.15;
  const meadowDetail = (fbm2d(worldX * 0.022, worldZ * 0.022, 4) - 0.5) * 4.2;
  const foothillStart = Math.max(0, lateral - 16);
  const foothills = Math.pow(foothillStart * 0.052, 1.72) * 11.5;
  const ridgeRoughness =
    foothillStart > 0
      ? (fbm2d(worldX * 0.045 + 9.2, worldZ * 0.045 - 4.7, 3) - 0.4) * Math.min(10, foothillStart * 0.25)
      : 0;
  const trailFlatten = getTrailMask(worldX, worldZ);
  const microBump = (valueNoise2d(worldX * 0.32, worldZ * 0.32) - 0.5) * 0.09 * (1 - trailFlatten * 0.72);
  return valleyRoll + meadowDetail * (1 - trailFlatten * 0.82) + foothills + ridgeRoughness + microBump;
}

export function getTerrainNormal(worldX, worldZ) {
  const eps = 0.4;
  const hL = getTerrainHeight(worldX - eps, worldZ);
  const hR = getTerrainHeight(worldX + eps, worldZ);
  const hD = getTerrainHeight(worldX, worldZ - eps);
  const hU = getTerrainHeight(worldX, worldZ + eps);
  const nx = hL - hR;
  const ny = 2 * eps;
  const nz = hD - hU;
  const len = Math.hypot(nx, ny, nz) || 1;
  return { x: nx / len, y: ny / len, z: nz / len };
}

export function getMountainRidgeHeight(angleRad, radiusNorm) {
  const dirX = Math.cos(angleRad);
  const dirZ = Math.sin(angleRad);
  const major = fbm2d(dirX * 3.1 + 4.2, dirZ * 3.1 - 1.7, 5);
  const crag = 1 - Math.abs(fbm2d(dirX * 7.8 - 2.3, dirZ * 7.8 + 5.1, 4) * 2 - 1);
  const twinBellA = Math.exp(-Math.pow((angleRad - 1.36) * 4.5, 2)) * 42;
  const twinBellB = Math.exp(-Math.pow((angleRad - 1.76) * 4.8, 2)) * 38;
  const westPeak = Math.exp(-Math.pow((angleRad - 2.55) * 3.4, 2)) * 34;
  const eastPeak = Math.exp(-Math.pow((angleRad - 0.58) * 3.8, 2)) * 31;
  const profile = Math.sin(Math.min(1, Math.max(0, radiusNorm)) * Math.PI);
  return (18 + major * 58 + crag * 24 + twinBellA + twinBellB + westPeak + eastPeak) * profile;
}
