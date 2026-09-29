import { placeOf, createGeo } from './core/geo.js';

async function grid(id, name) {
  const r = await fetch(`/geo/${id}/${name}.bin`);
  if (!r.ok) throw new Error(`missing terrain ${id}/${name}`);
  return new Int16Array(await r.arrayBuffer());
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`missing imagery ${src}`));
    img.src = src;
  });
}

async function mosaic(id, info) {
  const canvas = document.createElement('canvas');
  canvas.width = info.cols * 256;
  canvas.height = info.rows * 256;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  await Promise.all(info.tiles.map(async (name) => {
    const [, x, y] = name.replace('.jpg', '').split('_').slice(-3);
    const img = await loadImage(`/geo/${id}/${name}`);
    ctx.drawImage(img, (Number(x) - info.x0) * 256, (Number(y) - info.y0) * 256);
  }));
  return canvas;
}

export async function loadPlace(id) {
  const manifest = await (await fetch(`/geo/${id}/manifest.json`)).json();
  const [near, far, nearImg, farImg] = await Promise.all([grid(id, 'near'), grid(id, 'far'), mosaic(id, manifest.imagery.near), mosaic(id, manifest.imagery.far)]);
  return {
    geo: createGeo(placeOf(id), near, far),
    imagery: { near: { canvas: nearImg, info: manifest.imagery.near }, far: { canvas: farImg, info: manifest.imagery.far } },
  };
}
