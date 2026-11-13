import { type Ctx, blob, fillInk, ink, linear, paint, seeded } from "./canvas";

export type Ground = "meadow" | "forest" | "night" | "interior" | "cover";

export interface BackdropSpec {
  ground: Ground;
  page: number;
  seed: number;
}

export const SPREAD_W = 2048;
export const SPREAD_H = 1434;
const MARGIN = 70;

function paper(ctx: Ctx, w: number, h: number): void {
  ctx.fillStyle = "#f7ecd3";
  ctx.fillRect(0, 0, w, h);
  const rand = seeded(99);
  for (let i = 0; i < 4000; i++) {
    ctx.fillStyle = `rgba(120,90,40,${rand() * 0.05})`;
    ctx.fillRect(rand() * w, rand() * h, 2, 2);
  }
}

function tornRect(ctx: Ctx, w: number, h: number, rand: () => number): void {
  ctx.beginPath();
  const step = 24;
  ctx.moveTo(MARGIN, MARGIN);
  for (let x = MARGIN; x <= w - MARGIN; x += step) ctx.lineTo(x, MARGIN + rand() * 8);
  for (let y = MARGIN; y <= h - MARGIN; y += step) ctx.lineTo(w - MARGIN - rand() * 8, y);
  for (let x = w - MARGIN; x >= MARGIN; x -= step) ctx.lineTo(x, h - MARGIN - rand() * 8);
  for (let y = h - MARGIN; y >= MARGIN; y -= step) ctx.lineTo(MARGIN + rand() * 8, y);
  ctx.closePath();
}

function meadow(ctx: Ctx, w: number, h: number, grass: HTMLImageElement, ground: Ground, rand: () => number): void {
  const dusk = ground === "night";
  const pond = ground === "meadow";
  ctx.save();
  ctx.clip();
  const pattern = ctx.createPattern(grass, "repeat");
  if (pattern) {
    ctx.fillStyle = pattern;
    ctx.fillRect(0, 0, w, h);
  }
  ctx.fillStyle = dusk ? "rgba(40,30,90,0.62)" : pond ? "rgba(150,220,90,0.28)" : "rgba(30,90,50,0.35)";
  ctx.fillRect(0, 0, w, h);
  ctx.beginPath();
  ctx.moveTo(w * 0.42, -20);
  ctx.bezierCurveTo(w * 0.3, h * 0.35, w * 0.72, h * 0.55, w * 0.46, h + 20);
  ctx.lineTo(w * 0.6, h + 20);
  ctx.bezierCurveTo(w * 0.84, h * 0.55, w * 0.4, h * 0.35, w * 0.5, -20);
  ctx.closePath();
  ctx.fillStyle = dusk ? "#9b8263" : "#e2c38d";
  ctx.fill();
  ink(ctx, 6, "rgba(110,80,40,0.45)");
  for (let i = 0; i < 26; i++) {
    const x = w * 0.42 + rand() * w * 0.2;
    const y = rand() * h;
    blob(ctx, x, y, 10 + rand() * 12, 7 + rand() * 6);
    fillInk(ctx, "#cdb48a", 3);
  }
  const petals = ["#ff6fa3", "#ffffff", "#ffd23f", "#9b7bff", "#ff8a3d"];
  for (let i = 0; i < 150; i++) {
    const x = rand() * w;
    const y = rand() * h;
    const c = petals[Math.floor(rand() * petals.length)] ?? "#ffffff";
    for (let p = 0; p < 5; p++) {
      const a = (p / 5) * Math.PI * 2;
      blob(ctx, x + Math.cos(a) * 7, y + Math.sin(a) * 7, 6, 6);
      ctx.fillStyle = c;
      ctx.fill();
    }
    blob(ctx, x, y, 4, 4);
    ctx.fillStyle = "#ffcf33";
    ctx.fill();
  }
  ctx.strokeStyle = dusk ? "rgba(40,70,40,0.6)" : "rgba(40,110,40,0.55)";
  ctx.lineWidth = 3;
  for (let i = 0; i < 500; i++) {
    const x = rand() * w;
    const y = rand() * h;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.quadraticCurveTo(x + 4, y - 10, x + 2 + rand() * 6, y - 18);
    ctx.stroke();
  }
  if (!pond) {
    ctx.restore();
    return;
  }
  blob(ctx, w * 0.82, h * 0.78, 150, 70);
  ctx.fillStyle = linear(ctx, h * 0.7, h * 0.86, ["#7fd0f0", "#3c8fcc"]);
  ctx.fill();
  ink(ctx, 8, "#2f6c99");
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  blob(ctx, w * 0.8, h * 0.76, 60, 10);
  ctx.fill();
  ctx.restore();
}

function interior(ctx: Ctx, w: number, h: number, rand: () => number): void {
  ctx.save();
  ctx.clip();
  const plank = 110;
  for (let y = 0; y < h; y += plank) {
    const offset = rand() * 400;
    for (let x = -offset; x < w; x += 420) {
      const tone = 150 + rand() * 40;
      ctx.fillStyle = `rgb(${tone},${tone * 0.62},${tone * 0.35})`;
      ctx.fillRect(x, y, 420, plank);
      ctx.strokeStyle = "rgba(60,30,10,0.6)";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, 420, plank);
    }
  }
  blob(ctx, w / 2, h * 0.62, 620, 340);
  fillInk(ctx, "#5e3a8a", 10);
  for (let r = 0; r < 4; r++) {
    blob(ctx, w / 2, h * 0.62, 560 - r * 130, 300 - r * 72);
    ctx.lineWidth = 18;
    ctx.strokeStyle = r % 2 ? "#ffd23f" : "#ff9fd0";
    ctx.setLineDash([30, 20]);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function crease(ctx: Ctx, w: number, h: number): void {
  const g = ctx.createLinearGradient(w / 2 - 120, 0, w / 2 + 120, 0);
  g.addColorStop(0, "rgba(60,30,10,0)");
  g.addColorStop(0.5, "rgba(60,30,10,0.38)");
  g.addColorStop(1, "rgba(60,30,10,0)");
  ctx.fillStyle = g;
  ctx.fillRect(w / 2 - 120, 0, 240, h);
}

function pageNumbers(ctx: Ctx, w: number, h: number, page: number): void {
  ctx.font = "italic 44px 'IM Fell English', serif";
  ctx.fillStyle = "#7a5530";
  ctx.textAlign = "center";
  ctx.fillText(String(page * 2 - 1), 120, h - 22);
  ctx.fillText(String(page * 2), w - 120, h - 22);
}

function coverGround(ctx: Ctx, w: number, h: number): void {
  ctx.save();
  ctx.clip();
  ctx.fillStyle = linear(ctx, 0, h, ["#2b4a8f", "#7a3f8f"]);
  ctx.fillRect(0, 0, w, h);
  const rand = seeded(3);
  for (let i = 0; i < 220; i++) {
    ctx.fillStyle = `rgba(255,240,180,${0.4 + rand() * 0.6})`;
    const s = 2 + rand() * 6;
    ctx.save();
    ctx.translate(rand() * w, rand() * h);
    ctx.rotate(Math.PI / 4);
    ctx.fillRect(-s / 2, -s / 2, s, s);
    ctx.restore();
  }
  ctx.restore();
}

export function paintBackdrop(spec: BackdropSpec, grass: HTMLImageElement): HTMLCanvasElement {
  return paint(SPREAD_W, SPREAD_H, (ctx, w, h) => {
    const rand = seeded(spec.seed);
    paper(ctx, w, h);
    tornRect(ctx, w, h, rand);
    ctx.shadowColor = "rgba(60,30,10,0.35)";
    ctx.shadowBlur = 20;
    ctx.fillStyle = "#f7ecd3";
    ctx.fill();
    ctx.shadowBlur = 0;
    if (spec.ground === "interior") interior(ctx, w, h, rand);
    else if (spec.ground === "cover") coverGround(ctx, w, h);
    else meadow(ctx, w, h, grass, spec.ground, rand);
    tornRect(ctx, w, h, seeded(spec.seed));
    ink(ctx, 5, "rgba(90,60,30,0.6)");
    crease(ctx, w, h);
    pageNumbers(ctx, w, h, spec.page);
  });
}
