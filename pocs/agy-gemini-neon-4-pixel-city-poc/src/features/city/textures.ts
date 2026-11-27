import * as THREE from "three";
import type { PinKind } from "./types";

function makeCanvas(width: number, height: number): {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
} {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2D canvas context unavailable");
  }
  ctx.imageSmoothingEnabled = false;
  return { canvas, ctx };
}

function finalizeTexture(
  canvas: HTMLCanvasElement,
  repeatX = 1,
  repeatY = 1,
): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeatX, repeatY);
  tex.needsUpdate = true;
  return tex;
}

export function createGrassTexture(repeatX = 6, repeatY = 6): THREE.CanvasTexture {
  const { canvas, ctx } = makeCanvas(32, 32);
  ctx.fillStyle = "#77aa58";
  ctx.fillRect(0, 0, 32, 32);
  const shades = ["#6fa050", "#81b561", "#639246"];
  for (let y = 0; y < 32; y += 2) {
    for (let x = 0; x < 32; x += 2) {
      const hash = (x * 17 + y * 31) % 11;
      if (hash < 3) {
        ctx.fillStyle = shades[hash] ?? "#6fa050";
        ctx.fillRect(x, y, 2, 2);
      }
    }
  }
  ctx.fillStyle = "#527d3b";
  ctx.fillRect(4, 6, 2, 4);
  ctx.fillRect(18, 22, 2, 4);
  ctx.fillRect(26, 10, 2, 3);
  return finalizeTexture(canvas, repeatX, repeatY);
}

export function createSoilTexture(repeatX = 5, repeatY = 1): THREE.CanvasTexture {
  const { canvas, ctx } = makeCanvas(64, 24);
  ctx.fillStyle = "#629448";
  ctx.fillRect(0, 0, 64, 3);
  ctx.fillStyle = "#2b2320";
  ctx.fillRect(0, 3, 64, 1);
  ctx.fillStyle = "#8e6140";
  ctx.fillRect(0, 4, 64, 7);
  ctx.fillStyle = "#764d32";
  ctx.fillRect(0, 11, 64, 7);
  ctx.fillStyle = "#5a3924";
  ctx.fillRect(0, 18, 64, 6);

  const rocks: readonly (readonly [number, number, number, number])[] = [
    [6, 7, 6, 4],
    [20, 13, 8, 5],
    [36, 8, 5, 4],
    [48, 14, 9, 5],
    [13, 19, 6, 3],
    [57, 7, 5, 4],
  ];
  for (const [rx, ry, rw, rh] of rocks) {
    ctx.fillStyle = "#483123";
    ctx.fillRect(rx - 1, ry - 1, rw + 2, rh + 2);
    ctx.fillStyle = "#88705e";
    ctx.fillRect(rx, ry, rw, rh);
    ctx.fillStyle = "#a38a76";
    ctx.fillRect(rx, ry, rw - 1, 1);
  }

  for (let x = 0; x < 64; x += 4) {
    if ((x / 4) % 2 === 0) {
      ctx.fillStyle = "#629448";
      ctx.fillRect(x, 3, 2, 2);
    }
  }
  return finalizeTexture(canvas, repeatX, repeatY);
}

export function createWaterTexture(repeatX = 4, repeatY = 4): THREE.CanvasTexture {
  const { canvas, ctx } = makeCanvas(32, 32);
  ctx.fillStyle = "#6d9eb8";
  ctx.fillRect(0, 0, 32, 32);
  ctx.fillStyle = "#5d8fa9";
  ctx.fillRect(0, 8, 32, 2);
  ctx.fillRect(0, 24, 32, 2);
  ctx.fillStyle = "#a8cde0";
  ctx.fillRect(3, 5, 7, 1);
  ctx.fillRect(18, 13, 8, 1);
  ctx.fillRect(7, 21, 6, 1);
  ctx.fillRect(21, 29, 7, 1);
  return finalizeTexture(canvas, repeatX, repeatY);
}

export function createRoadTexture(
  dashed: boolean,
  vertical: boolean,
  repeatX = 1,
  repeatY = 1,
): THREE.CanvasTexture {
  const { canvas, ctx } = makeCanvas(32, 32);
  ctx.fillStyle = "#aba29e";
  ctx.fillRect(0, 0, 32, 32);
  ctx.fillStyle = "#9d9490";
  for (let i = 0; i < 32; i += 4) {
    ctx.fillRect((i * 3) % 32, (i * 5) % 32, 2, 2);
  }
  if (dashed) {
    ctx.fillStyle = "#f2eee6";
    if (vertical) {
      ctx.fillRect(15, 2, 2, 10);
      ctx.fillRect(15, 18, 2, 10);
    } else {
      ctx.fillRect(2, 15, 10, 2);
      ctx.fillRect(18, 15, 10, 2);
    }
  }
  return finalizeTexture(canvas, repeatX, repeatY);
}

export type FacadeOptions = {
  readonly base: string;
  readonly windowColor: string;
  readonly frameColor?: string;
  readonly cols: number;
  readonly rows: number;
  readonly stripeColor?: string;
  readonly litRatio?: number;
};

export function createFacadeTexture(opts: FacadeOptions): THREE.CanvasTexture {
  const cellW = 8;
  const cellH = 8;
  const w = Math.max(16, opts.cols * cellW);
  const h = Math.max(16, opts.rows * cellH);
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.fillStyle = opts.base;
  ctx.fillRect(0, 0, w, h);

  if (opts.stripeColor) {
    ctx.fillStyle = opts.stripeColor;
    for (let y = 0; y < h; y += cellH) {
      ctx.fillRect(0, y, w, 1);
    }
  }

  const frame = opts.frameColor ?? "#2c2420";
  const litRatio = opts.litRatio ?? 0;

  for (let r = 0; r < opts.rows; r++) {
    for (let c = 0; c < opts.cols; c++) {
      const wx = c * cellW + 2;
      const wy = r * cellH + 2;
      ctx.fillStyle = frame;
      ctx.fillRect(wx - 1, wy - 1, 6, 6);
      const pseudo = ((r * 13 + c * 7) % 10) / 10;
      const isLit = litRatio > 0 && pseudo < litRatio;
      ctx.fillStyle = isLit ? "#ffe596" : opts.windowColor;
      ctx.fillRect(wx, wy, 4, 4);
      ctx.fillStyle = isLit ? "#fff5d1" : "#c7dce8";
      ctx.fillRect(wx, wy, 2, 1);
    }
  }
  return finalizeTexture(canvas, 1, 1);
}

export function createGlassTowerTexture(
  cols: number,
  rows: number,
  litRatio = 0,
): THREE.CanvasTexture {
  const w = cols * 6;
  const h = rows * 5;
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.fillStyle = "#548fa9";
  ctx.fillRect(0, 0, w, h);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * 6;
      const y = r * 5;
      ctx.fillStyle = "#3a6980";
      ctx.fillRect(x, y, 6, 1);
      ctx.fillRect(x, y, 1, 5);
      const hash = ((r * 17 + c * 11) % 10) / 10;
      if (litRatio > 0 && hash < litRatio) {
        ctx.fillStyle = "#ffe8a3";
        ctx.fillRect(x + 1, y + 1, 5, 4);
      } else if ((r + c) % 3 === 0) {
        ctx.fillStyle = "#6aa4be";
        ctx.fillRect(x + 1, y + 1, 5, 4);
      }
    }
  }
  return finalizeTexture(canvas, 1, 1);
}

export function createPyramidTexture(litRatio = 0): THREE.CanvasTexture {
  const { canvas, ctx } = makeCanvas(64, 128);
  ctx.fillStyle = "#d5dce2";
  ctx.fillRect(0, 0, 64, 128);

  for (let y = 8; y < 120; y += 5) {
    ctx.fillStyle = "#adb8c2";
    ctx.fillRect(0, y, 64, 1);
    for (let x = 2; x < 62; x += 4) {
      const hash = ((x * 7 + y * 3) % 10) / 10;
      ctx.fillStyle = litRatio > 0 && hash < litRatio ? "#ffe79e" : "#4a5b6a";
      ctx.fillRect(x, y + 1, 2, 3);
    }
  }
  return finalizeTexture(canvas, 1, 1);
}

export function createPinSpriteTexture(kind: PinKind): THREE.CanvasTexture {
  const { canvas, ctx } = makeCanvas(32, 32);
  ctx.clearRect(0, 0, 32, 32);

  ctx.fillStyle = "#2b2320";
  ctx.fillRect(5, 3, 22, 18);
  ctx.fillRect(4, 4, 24, 16);
  ctx.fillRect(13, 21, 6, 4);
  ctx.fillRect(14, 25, 4, 3);

  ctx.fillStyle = "#fcfaf5";
  ctx.fillRect(6, 4, 20, 16);
  ctx.fillRect(5, 5, 22, 14);
  ctx.fillRect(14, 20, 4, 4);

  switch (kind) {
    case "leaf": {
      ctx.fillStyle = "#4d9c47";
      ctx.fillRect(13, 7, 7, 7);
      ctx.fillRect(12, 9, 7, 6);
      ctx.fillRect(16, 6, 4, 4);
      ctx.fillStyle = "#81c86b";
      ctx.fillRect(14, 8, 3, 3);
      ctx.fillStyle = "#356f32";
      ctx.fillRect(11, 14, 2, 3);
      ctx.fillRect(13, 12, 2, 2);
      break;
    }
    case "pin": {
      ctx.fillStyle = "#d14636";
      ctx.fillRect(13, 6, 6, 7);
      ctx.fillRect(12, 7, 8, 5);
      ctx.fillRect(14, 13, 4, 3);
      ctx.fillRect(15, 16, 2, 2);
      ctx.fillStyle = "#fcfaf5";
      ctx.fillRect(15, 8, 2, 2);
      break;
    }
    case "cart": {
      ctx.fillStyle = "#525b66";
      ctx.fillRect(10, 8, 3, 2);
      ctx.fillRect(12, 9, 10, 5);
      ctx.fillStyle = "#fcfaf5";
      ctx.fillRect(14, 10, 6, 3);
      ctx.fillStyle = "#2b2320";
      ctx.fillRect(13, 15, 2, 2);
      ctx.fillRect(19, 15, 2, 2);
      break;
    }
    default: {
      const unreachable: never = kind;
      return unreachable;
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}
