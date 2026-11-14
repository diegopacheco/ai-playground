import { CanvasTexture, SRGBColorSpace } from "three";

export type Ctx = CanvasRenderingContext2D;
export type Draw = (ctx: Ctx) => void;

export const PX = 240;

export function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: Ctx } {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(w));
  canvas.height = Math.max(1, Math.round(h));
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d canvas is not available");
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  return { canvas, ctx };
}

export function toTexture(canvas: HTMLCanvasElement): CanvasTexture {
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

export function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function leather(ctx: Ctx, w: number, h: number, seed: number): void {
  ctx.save();
  ctx.globalCompositeOperation = "source-atop";
  const g = ctx.createLinearGradient(0, 0, w * 0.4, h);
  g.addColorStop(0, "#6e3a1c");
  g.addColorStop(0.5, "#4a2410");
  g.addColorStop(1, "#2c1408");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  const rand = seeded(seed);
  ctx.fillStyle = "rgba(200,140,70,0.12)";
  for (let i = 0; i < (w * h) / 900; i++) ctx.fillRect(rand() * w, rand() * h, 2, 2);
  ctx.restore();
}

export function silhouette(unitsW: number, unitsH: number, worldH: number, draw: Draw, seed = 7): HTMLCanvasElement {
  const k = (worldH * PX) / unitsH;
  const { canvas, ctx } = makeCanvas(unitsW * k, unitsH * k);
  ctx.scale(k, k);
  ctx.fillStyle = "#000";
  ctx.strokeStyle = "#000";
  draw(ctx);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  leather(ctx, canvas.width, canvas.height, seed);
  return canvas;
}

export function punch(ctx: Ctx, draw: Draw): void {
  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = "#000";
  ctx.strokeStyle = "#000";
  draw(ctx);
  ctx.restore();
}

export function poly(ctx: Ctx, points: number[]): void {
  ctx.beginPath();
  ctx.moveTo(points[0] ?? 0, points[1] ?? 0);
  for (let i = 2; i < points.length; i += 2) ctx.lineTo(points[i] ?? 0, points[i + 1] ?? 0);
  ctx.closePath();
  ctx.fill();
}

export function circle(ctx: Ctx, x: number, y: number, r: number): void {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

export function oval(ctx: Ctx, x: number, y: number, rx: number, ry: number, rot = 0): void {
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, rot, 0, Math.PI * 2);
  ctx.fill();
}

export function line(ctx: Ctx, width: number, points: number[]): void {
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0] ?? 0, points[1] ?? 0);
  for (let i = 2; i < points.length; i += 2) ctx.lineTo(points[i] ?? 0, points[i + 1] ?? 0);
  ctx.stroke();
}

export function curve(ctx: Ctx, width: number, x0: number, y0: number, cx: number, cy: number, x1: number, y1: number): void {
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.quadraticCurveTo(cx, cy, x1, y1);
  ctx.stroke();
}

export function star(ctx: Ctx, x: number, y: number, r: number, points = 8, inner = 0.5, rot = 0): void {
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const a = rot + (i * Math.PI) / points - Math.PI / 2;
    const rr = i % 2 === 0 ? r : r * inner;
    ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
  }
  ctx.closePath();
  ctx.fill();
}

export function crescent(ctx: Ctx, x: number, y: number, r: number, tilt = -0.5): void {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.arc(x + Math.cos(tilt) * r * 0.42, y + Math.sin(tilt) * r * 0.42, r * 0.82, 0, Math.PI * 2, true);
  ctx.fill("evenodd");
  ctx.restore();
}

export function dotsAlong(ctx: Ctx, x0: number, y0: number, x1: number, y1: number, count: number, r: number): void {
  for (let i = 0; i < count; i++) {
    const t = count === 1 ? 0.5 : i / (count - 1);
    circle(ctx, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r);
  }
}

export function diamond(ctx: Ctx, x: number, y: number, rx: number, ry: number): void {
  poly(ctx, [x, y - ry, x + rx, y, x, y + ry, x - rx, y]);
}

export function archHole(ctx: Ctx, x: number, y: number, w: number, h: number): void {
  ctx.beginPath();
  ctx.moveTo(x - w / 2, y + h);
  ctx.lineTo(x - w / 2, y + w * 0.5);
  ctx.quadraticCurveTo(x - w / 2, y + w * 0.05, x, y);
  ctx.quadraticCurveTo(x + w / 2, y + w * 0.05, x + w / 2, y + w * 0.5);
  ctx.lineTo(x + w / 2, y + h);
  ctx.closePath();
  ctx.fill();
}

export function onionDome(ctx: Ctx, x: number, base: number, w: number, h: number): void {
  ctx.beginPath();
  ctx.moveTo(x - w / 2, base);
  ctx.bezierCurveTo(x - w * 0.75, base - h * 0.45, x - w * 0.1, base - h * 0.7, x, base - h);
  ctx.bezierCurveTo(x + w * 0.1, base - h * 0.7, x + w * 0.75, base - h * 0.45, x + w / 2, base);
  ctx.closePath();
  ctx.fill();
}

export function lettering(text: string, font: string, unitsW: number, unitsH: number, worldH: number, frame: Draw): HTMLCanvasElement {
  return silhouette(unitsW, unitsH, worldH, (ctx) => {
    frame(ctx);
    punch(ctx, (c) => {
      c.font = font;
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.fillText(text, unitsW / 2, unitsH / 2 + 1, unitsW - 44);
    });
  });
}
