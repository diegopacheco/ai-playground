import { CanvasTexture, SRGBColorSpace } from "three";

export type Ctx = CanvasRenderingContext2D;
export type Painter = (ctx: Ctx, w: number, h: number) => void;

export function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: Ctx } {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d canvas is not available");
  return { canvas, ctx };
}

export function paint(w: number, h: number, painter: Painter): HTMLCanvasElement {
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  painter(ctx, w, h);
  return canvas;
}

export function withPaperBorder(source: HTMLCanvasElement, border: number): HTMLCanvasElement {
  const pad = border + 6;
  const w = source.width + pad * 2;
  const h = source.height + pad * 2;
  const tint = makeCanvas(source.width, source.height);
  tint.ctx.drawImage(source, 0, 0);
  tint.ctx.globalCompositeOperation = "source-in";
  tint.ctx.fillStyle = "#fffaf0";
  tint.ctx.fillRect(0, 0, source.width, source.height);
  const shade = makeCanvas(source.width, source.height);
  shade.ctx.drawImage(source, 0, 0);
  shade.ctx.globalCompositeOperation = "source-in";
  shade.ctx.fillStyle = "rgba(70,45,20,0.55)";
  shade.ctx.fillRect(0, 0, source.width, source.height);
  const out = makeCanvas(w, h);
  for (let a = 0; a < Math.PI * 2; a += Math.PI / 16) {
    out.ctx.drawImage(shade.canvas, pad + Math.cos(a) * (border + 2) + 2, pad + Math.sin(a) * (border + 2) + 3);
  }
  for (let a = 0; a < Math.PI * 2; a += Math.PI / 16) {
    out.ctx.drawImage(tint.canvas, pad + Math.cos(a) * border, pad + Math.sin(a) * border);
  }
  out.ctx.drawImage(source, pad, pad);
  return out.canvas;
}

export function toTexture(canvas: HTMLCanvasElement): CanvasTexture {
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

export function blob(ctx: Ctx, cx: number, cy: number, rx: number, ry: number): void {
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
}

export function radial(ctx: Ctx, cx: number, cy: number, r: number, inner: string, outer: string): CanvasGradient {
  const g = ctx.createRadialGradient(cx - r * 0.35, cy - r * 0.4, r * 0.1, cx, cy, r);
  g.addColorStop(0, inner);
  g.addColorStop(1, outer);
  return g;
}

export function linear(ctx: Ctx, y0: number, y1: number, stops: string[]): CanvasGradient {
  const g = ctx.createLinearGradient(0, y0, 0, y1);
  stops.forEach((c, i) => g.addColorStop(i / Math.max(stops.length - 1, 1), c));
  return g;
}

export function ink(ctx: Ctx, width: number, color = "#3b2416"): void {
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.stroke();
}

export function fillInk(ctx: Ctx, fill: string | CanvasGradient, width = 6): void {
  ctx.fillStyle = fill;
  ctx.fill();
  ink(ctx, width);
}

export function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
