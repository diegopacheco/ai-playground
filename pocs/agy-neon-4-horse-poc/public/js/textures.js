import * as THREE from 'three';

function createSeededRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function makeCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return { canvas, ctx: canvas.getContext('2d') };
}

export function createCoatTextures(coat) {
  const width = 512;
  const height = 512;
  const { canvas: colorCanvas, ctx } = makeCanvas(width, height);
  const { canvas: bumpCanvas, ctx: bctx } = makeCanvas(width, height);
  const rng = createSeededRng(417);

  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, coat.bodyHighlight);
  grad.addColorStop(0.35, coat.bodyPrimary);
  grad.addColorStop(0.76, coat.bodyPrimary);
  grad.addColorStop(1, coat.bodyShadow);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  bctx.fillStyle = '#808080';
  bctx.fillRect(0, 0, width, height);

  if (coat.dapple > 0.06) {
    const count = Math.floor(150 * coat.dapple + 32);
    for (let i = 0; i < count; i += 1) {
      const cx = rng() * width;
      const cy = height * (0.12 + rng() * 0.72);
      const r = 10 + rng() * 24;
      const rg = ctx.createRadialGradient(cx, cy, r * 0.15, cx, cy, r);
      rg.addColorStop(0, `rgba(255,255,255,${(coat.dapple * 0.25).toFixed(3)})`);
      rg.addColorStop(0.65, `rgba(20,20,24,${(coat.dapple * 0.12).toFixed(3)})`);
      rg.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = rg;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  for (let i = 0; i < 7200; i += 1) {
    const x = rng() * width;
    const y = rng() * height;
    const len = 6 + rng() * 15;
    const angle = (rng() - 0.5) * 0.22;
    const bright = rng() > 0.5;
    ctx.strokeStyle = bright ? 'rgba(255,244,230,0.068)' : 'rgba(18,10,6,0.072)';
    ctx.lineWidth = 0.8 + rng() * 0.9;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len);
    ctx.stroke();

    bctx.strokeStyle = bright ? 'rgba(218,218,218,0.16)' : 'rgba(55,55,55,0.16)';
    bctx.lineWidth = 1;
    bctx.beginPath();
    bctx.moveTo(x, y);
    bctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len);
    bctx.stroke();
  }

  const map = new THREE.CanvasTexture(colorCanvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.ClampToEdgeWrapping;

  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  bumpMap.wrapS = THREE.RepeatWrapping;
  bumpMap.wrapT = THREE.RepeatWrapping;

  return { map, bumpMap };
}

export function createHairTexture(baseHex) {
  const width = 128;
  const height = 256;
  const { canvas, ctx } = makeCanvas(width, height);
  const rng = createSeededRng(909);

  ctx.clearRect(0, 0, width, height);
  const base = new THREE.Color(baseHex);
  const light = base.clone().offsetHSL(0, -0.04, 0.16);
  const dark = base.clone().offsetHSL(0, 0.02, -0.1);

  for (let i = 0; i < 320; i += 1) {
    const spread = (rng() - 0.5) * 2;
    const x = width * 0.5 + spread * width * 0.34;
    const edgeFade = 1 - Math.abs(spread) * 0.55;
    const sway = (rng() - 0.5) * 18;
    const yEnd = height * (0.58 + rng() * 0.4) * edgeFade;
    const col = rng() > 0.65 ? light : rng() > 0.3 ? base : dark;
    const alpha = (0.48 + rng() * 0.45) * edgeFade;
    ctx.strokeStyle = `rgba(${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(col.b * 255)},${alpha.toFixed(2)})`;
    ctx.lineWidth = 1.0 + rng() * 1.9;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.quadraticCurveTo(x + sway * 0.5, yEnd * 0.52, x + sway, yEnd);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function createMeadowDetailTexture() {
  const size = 512;
  const { canvas, ctx } = makeCanvas(size, size);
  const rng = createSeededRng(733);

  ctx.fillStyle = '#d4d4d4';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 6200; i += 1) {
    const x = rng() * size;
    const y = rng() * size;
    const r = 1 + rng() * 3.8;
    const v = Math.floor(182 + rng() * 64);
    ctx.fillStyle = `rgba(${v},${v},${v},0.3)`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  for (let i = 0; i < 4200; i += 1) {
    const x = rng() * size;
    const y = rng() * size;
    const h = 4 + rng() * 12;
    const shade = Math.floor(155 + rng() * 88);
    ctx.strokeStyle = `rgba(${shade},${shade},${shade},0.28)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + (rng() - 0.5) * 4, y - h);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(26, 26);
  return tex;
}

export function createAspenBarkTexture() {
  const width = 128;
  const height = 256;
  const { canvas, ctx } = makeCanvas(width, height);
  const rng = createSeededRng(314);

  const grad = ctx.createLinearGradient(0, 0, width, 0);
  grad.addColorStop(0, '#dad7cd');
  grad.addColorStop(0.5, '#f3f1e9');
  grad.addColorStop(1, '#d1cec3');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < 85; i += 1) {
    const y = rng() * height;
    const x = rng() * width;
    const w = 10 + rng() * 28;
    const h = 1.5 + rng() * 4.2;
    ctx.fillStyle = `rgba(38,34,29,${(0.45 + rng() * 0.45).toFixed(2)})`;
    ctx.beginPath();
    ctx.ellipse(x, y, w * 0.5, h * 0.5, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

export function createAspenFoliageTexture() {
  const size = 256;
  const { canvas, ctx } = makeCanvas(size, size);
  const rng = createSeededRng(521);

  ctx.fillStyle = '#6c9c33';
  ctx.fillRect(0, 0, size, size);

  const greens = ['#7db03b', '#98c74a', '#5b8a2c', '#b7d457', '#6a9d34'];
  for (let i = 0; i < 420; i += 1) {
    const x = rng() * size;
    const y = rng() * size;
    const rx = 5 + rng() * 10;
    const ry = 6 + rng() * 11;
    ctx.fillStyle = greens[Math.floor(rng() * greens.length)];
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rng() * Math.PI);
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

export function createDustSpriteTexture() {
  const size = 128;
  const { canvas, ctx } = makeCanvas(size, size);
  const rg = ctx.createRadialGradient(size / 2, size / 2, 4, size / 2, size / 2, size / 2);
  rg.addColorStop(0, 'rgba(235, 204, 166, 0.85)');
  rg.addColorStop(0.45, 'rgba(218, 182, 140, 0.48)');
  rg.addColorStop(0.76, 'rgba(202, 168, 132, 0.14)');
  rg.addColorStop(1, 'rgba(202, 168, 132, 0)');
  ctx.fillStyle = rg;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function createCloudSpriteTexture() {
  const width = 256;
  const height = 128;
  const { canvas, ctx } = makeCanvas(width, height);
  const puffs = [
    [128, 72, 62, 0.85],
    [80, 78, 46, 0.78],
    [176, 78, 48, 0.76],
    [108, 52, 44, 0.9],
    [152, 54, 40, 0.86]
  ];
  for (const [cx, cy, r, alpha] of puffs) {
    const rg = ctx.createRadialGradient(cx, cy - r * 0.2, r * 0.08, cx, cy, r);
    rg.addColorStop(0, `rgba(255,255,255,${alpha})`);
    rg.addColorStop(0.65, `rgba(245,249,255,${(alpha * 0.6).toFixed(2)})`);
    rg.addColorStop(1, 'rgba(235,243,255,0)');
    ctx.fillStyle = rg;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
