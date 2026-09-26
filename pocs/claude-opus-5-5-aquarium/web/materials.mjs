import * as THREE from 'three';
import { shade } from './patterns.mjs';

const textures = new Map();

function paint(canvas, pattern, tint) {
  const size = canvas.width;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  const n = parseInt((tint || '#ffffff').slice(1), 16);
  const c = { r: ((n >> 16) & 255) / 255, g: ((n >> 8) & 255) / 255, b: (n & 255) / 255 };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const g = Math.min(1, 0.32 + 0.8 * shade(pattern, x / size, y / size));
      const i = (y * size + x) * 4;
      img.data[i] = 255 * g * c.r;
      img.data[i + 1] = 255 * g * c.g;
      img.data[i + 2] = 255 * g * c.b;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export function patternTexture(pattern) {
  if (textures.has(pattern)) return textures.get(pattern);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 512;
  const tex = new THREE.CanvasTexture(paint(canvas, pattern));
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  textures.set(pattern, tex);
  return tex;
}

export function thumbnail(def, size = 44) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  return paint(canvas, def.pattern, def.natural).toDataURL();
}

export function applyRack(material, def, hex) {
  const tex = patternTexture(def.pattern);
  material.map = tex;
  material.bumpMap = tex;
  material.bumpScale = def.bump;
  material.color.set(hex || def.natural);
  material.roughness = def.roughness;
  material.metalness = def.metalness;
  material.needsUpdate = true;
}
