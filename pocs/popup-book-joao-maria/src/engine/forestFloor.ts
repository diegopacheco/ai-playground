import { CanvasTexture, RepeatWrapping, SRGBColorSpace } from "three";

const SIZE = 1024;
const LEAF_COLORS = ["#b5471f", "#d9822b", "#e3a83a", "#8f2d1c", "#c25e26", "#a8792c"];
const MOSS_COLORS = ["#24401f", "#2e5226", "#1b3318", "#35602b", "#284a22"];

function seeded(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function wrapped(size: number, x: number, y: number, r: number, draw: (x: number, y: number) => void): void {
  for (const dx of [-size, 0, size]) {
    for (const dy of [-size, 0, size]) {
      if (x + dx > -r && x + dx < size + r && y + dy > -r && y + dy < size + r) draw(x + dx, y + dy);
    }
  }
}

function moss(ctx: CanvasRenderingContext2D, rand: () => number): void {
  ctx.fillStyle = "#1f3a1c";
  ctx.fillRect(0, 0, SIZE, SIZE);
  for (let i = 0; i < 2600; i++) {
    const x = rand() * SIZE;
    const y = rand() * SIZE;
    const r = 6 + rand() * 26;
    ctx.fillStyle = MOSS_COLORS[Math.floor(rand() * MOSS_COLORS.length)];
    ctx.globalAlpha = 0.35 + rand() * 0.4;
    wrapped(SIZE, x, y, r, (px, py) => {
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  for (let i = 0; i < 9000; i++) {
    ctx.fillStyle = rand() > 0.5 ? "#4f7d34" : "#152a12";
    ctx.globalAlpha = 0.5;
    ctx.fillRect(rand() * SIZE, rand() * SIZE, 2, 2);
  }
  ctx.globalAlpha = 1;
}

function leaf(ctx: CanvasRenderingContext2D, x: number, y: number, length: number, angle: number, color: string): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.beginPath();
  ctx.ellipse(3, 4, length, length * 0.42, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(-length, 0);
  ctx.quadraticCurveTo(0, -length * 0.85, length, 0);
  ctx.quadraticCurveTo(0, length * 0.85, -length, 0);
  ctx.fill();
  ctx.strokeStyle = "rgba(60,25,10,0.6)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(-length * 1.25, 0);
  ctx.lineTo(length * 0.9, 0);
  for (let k = -2; k <= 2; k++) {
    const vx = (k * length) / 3;
    ctx.moveTo(vx, 0);
    ctx.lineTo(vx + length * 0.25, -length * 0.3);
    ctx.moveTo(vx, 0);
    ctx.lineTo(vx + length * 0.25, length * 0.3);
  }
  ctx.stroke();
  ctx.restore();
}

function pebble(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, shade: number, tilt: number): void {
  ctx.fillStyle = "rgba(0,0,0,0.4)";
  ctx.beginPath();
  ctx.ellipse(x + 2, y + 3, r * 1.2, r, 0, 0, Math.PI * 2);
  ctx.fill();
  const gradient = ctx.createRadialGradient(x - r * 0.4, y - r * 0.4, r * 0.1, x, y, r * 1.2);
  gradient.addColorStop(0, `rgb(${shade + 15},${shade + 15},${shade + 10})`);
  gradient.addColorStop(1, `rgb(${shade - 70},${shade - 70},${shade - 75})`);
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.ellipse(x, y, r * 1.2, r, tilt, 0, Math.PI * 2);
  ctx.fill();
}

export function forestFloorTexture(): CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
  const rand = seeded(7);
  moss(ctx, rand);
  for (let i = 0; i < 70; i++) {
    const x = rand() * SIZE;
    const y = rand() * SIZE;
    const length = 14 + rand() * 18;
    const angle = rand() * Math.PI * 2;
    const color = LEAF_COLORS[Math.floor(rand() * LEAF_COLORS.length)];
    wrapped(SIZE, x, y, length * 1.3, (px, py) => leaf(ctx, px, py, length, angle, color));
  }
  for (let i = 0; i < 26; i++) {
    const x = rand() * SIZE;
    const y = rand() * SIZE;
    const r = 5 + rand() * 6;
    const shade = 200 + Math.floor(rand() * 40);
    const tilt = rand() * Math.PI;
    wrapped(SIZE, x, y, r * 1.5, (px, py) => pebble(ctx, px, py, r, shade, tilt));
  }
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.repeat.set(5, 5);
  texture.anisotropy = 8;
  return texture;
}
