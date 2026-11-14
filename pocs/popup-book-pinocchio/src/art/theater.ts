import { type Ctx, blob, linear, paint, seeded } from "./canvas";

const GOLD = ["#fff1a8", "#e9b949", "#a8741c"];

function goldGradient(ctx: Ctx, y0: number, y1: number): CanvasGradient {
  return linear(ctx, y0, y1, GOLD);
}

function scroll(ctx: Ctx, x: number, y: number, r: number, flip: number): void {
  ctx.beginPath();
  for (let a = 0; a < Math.PI * 3.2; a += 0.1) {
    const k = r * (1 - a / (Math.PI * 3.6));
    const px = x + Math.cos(a) * k * flip;
    const py = y + Math.sin(a) * k;
    if (a === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

export function paintVelvet(): HTMLCanvasElement {
  return paint(512, 1024, (ctx, w, h) => {
    ctx.fillStyle = linear(ctx, 0, h, ["#7a0f1c", "#b51f2e", "#8e1422", "#5c0912"]);
    ctx.fillRect(0, 0, w, h);
    const rand = seeded(5);
    for (let i = 0; i < 9000; i++) {
      ctx.fillStyle = rand() > 0.5 ? "rgba(255,120,130,0.05)" : "rgba(30,0,5,0.08)";
      ctx.fillRect(rand() * w, rand() * h, 2, 6);
    }
    for (let x = 0; x < w; x += 64) {
      const g = ctx.createLinearGradient(x, 0, x + 64, 0);
      g.addColorStop(0, "rgba(0,0,0,0.25)");
      g.addColorStop(0.5, "rgba(255,140,150,0.12)");
      g.addColorStop(1, "rgba(0,0,0,0.25)");
      ctx.fillStyle = g;
      ctx.fillRect(x, 0, 64, h);
    }
    ctx.fillStyle = goldGradient(ctx, h - 70, h);
    ctx.fillRect(0, h - 62, w, 22);
    for (let x = 6; x < w; x += 14) {
      ctx.fillStyle = x % 28 === 6 ? "#f2c14e" : "#c48f2a";
      ctx.fillRect(x, h - 40, 7, 40);
    }
    ctx.fillStyle = goldGradient(ctx, 0, h);
    ctx.fillRect(w - 26, 0, 26, h - 40);
  });
}

export function paintValance(): HTMLCanvasElement {
  return paint(2048, 300, (ctx, w) => {
    const swags = 5;
    const sw = w / swags;
    ctx.fillStyle = linear(ctx, 0, 90, ["#5c0912", "#9e1a28"]);
    ctx.fillRect(0, 0, w, 90);
    for (let i = 0; i < swags; i++) {
      const x = i * sw;
      ctx.beginPath();
      ctx.moveTo(x, 80);
      ctx.quadraticCurveTo(x + sw / 2, 300, x + sw, 80);
      ctx.closePath();
      ctx.fillStyle = linear(ctx, 80, 250, ["#9e1a28", "#c22a38", "#7a0f1c"]);
      ctx.fill();
      for (let k = 1; k <= 3; k++) {
        ctx.beginPath();
        ctx.moveTo(x + 10, 80 + k * 6);
        ctx.quadraticCurveTo(x + sw / 2, 80 + k * 52, x + sw - 10, 80 + k * 6);
        ctx.strokeStyle = "rgba(40,0,6,0.35)";
        ctx.lineWidth = 6;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(x + 4, 82);
      ctx.quadraticCurveTo(x + sw / 2, 296, x + sw - 4, 82);
      ctx.strokeStyle = "#e9b949";
      ctx.lineWidth = 12;
      ctx.stroke();
      for (let t = 0.08; t < 0.95; t += 0.06) {
        const px = (1 - t) * (1 - t) * (x + 4) + 2 * (1 - t) * t * (x + sw / 2) + t * t * (x + sw - 4);
        const py = (1 - t) * (1 - t) * 82 + 2 * (1 - t) * t * 296 + t * t * 82;
        blob(ctx, px, py + 10, 7, 12);
        ctx.fillStyle = "#f2c14e";
        ctx.fill();
      }
      blob(ctx, x, 92, 26, 26);
      ctx.fillStyle = goldGradient(ctx, 66, 118);
      ctx.fill();
    }
    ctx.fillStyle = goldGradient(ctx, 0, 30);
    ctx.fillRect(0, 0, w, 26);
  });
}

export function paintFrame(w: number, h: number, vertical: boolean): HTMLCanvasElement {
  return paint(w, h, (ctx) => {
    ctx.fillStyle = linear(ctx, 0, h, ["#f4e3c1", "#e3c896", "#f0dcb2"]);
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = goldGradient(ctx, 0, h);
    ctx.lineWidth = 14;
    ctx.strokeRect(16, 16, w - 32, h - 32);
    ctx.lineWidth = 5;
    ctx.strokeRect(38, 38, w - 76, h - 76);
    ctx.strokeStyle = "#c9932f";
    ctx.lineWidth = 7;
    const count = vertical ? 6 : 10;
    for (let i = 0; i < count; i++) {
      const t = (i + 0.5) / count;
      const x = vertical ? w / 2 : t * w;
      const y = vertical ? t * h : h / 2;
      scroll(ctx, x - 6, y, 34, -1);
      scroll(ctx, x + 6, y, 34, 1);
      blob(ctx, x, y, 12, 12);
      ctx.fillStyle = "#e9b949";
      ctx.fill();
    }
  });
}

export function paintMarquee(title: string): HTMLCanvasElement {
  return paint(1400, 360, (ctx, w, h) => {
    ctx.beginPath();
    ctx.moveTo(40, h - 20);
    ctx.lineTo(40, 140);
    ctx.quadraticCurveTo(w / 2, -60, w - 40, 140);
    ctx.lineTo(w - 40, h - 20);
    ctx.closePath();
    ctx.fillStyle = goldGradient(ctx, 0, h);
    ctx.fill();
    ctx.lineWidth = 10;
    ctx.strokeStyle = "#7a4d10";
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(80, h - 50);
    ctx.lineTo(80, 160);
    ctx.quadraticCurveTo(w / 2, 0, w - 80, 160);
    ctx.lineTo(w - 80, h - 50);
    ctx.closePath();
    ctx.fillStyle = linear(ctx, 40, h, ["#2a0a3c", "#5a1450"]);
    ctx.fill();
    ctx.font = "150px Lobster";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 16;
    ctx.strokeStyle = "#3b1a05";
    ctx.strokeText(title, w / 2, h * 0.6);
    ctx.fillStyle = goldGradient(ctx, h * 0.35, h * 0.85);
    ctx.fillText(title, w / 2, h * 0.6);
  });
}

export function paintPlanks(): HTMLCanvasElement {
  return paint(1024, 512, (ctx, w, h) => {
    const rand = seeded(21);
    const rows = 8;
    for (let r = 0; r < rows; r++) {
      const y = (r * h) / rows;
      let x = -rand() * 300;
      while (x < w) {
        const len = 260 + rand() * 260;
        const tone = 120 + Math.floor(rand() * 40);
        ctx.fillStyle = `rgb(${tone},${Math.floor(tone * 0.62)},${Math.floor(tone * 0.33)})`;
        ctx.fillRect(x, y, len, h / rows);
        for (let k = 0; k < 14; k++) {
          ctx.fillStyle = "rgba(50,25,8,0.18)";
          ctx.fillRect(x, y + rand() * (h / rows), len, 1.5);
        }
        ctx.fillStyle = "rgba(30,14,4,0.8)";
        ctx.fillRect(x, y, 3, h / rows);
        x += len;
      }
      ctx.fillStyle = "rgba(30,14,4,0.85)";
      ctx.fillRect(0, y, w, 3);
    }
  });
}

export function paintDamask(): HTMLCanvasElement {
  return paint(512, 512, (ctx, w, h) => {
    ctx.fillStyle = "#3a0d16";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(201,147,47,0.35)";
    ctx.lineWidth = 5;
    for (const [x, y] of [[w / 4, h / 4], [(3 * w) / 4, (3 * h) / 4], [(3 * w) / 4, h / 4], [w / 4, (3 * h) / 4]] as const) {
      scroll(ctx, x - 8, y, 50, -1);
      scroll(ctx, x + 8, y, 50, 1);
      blob(ctx, x, y - 50, 14, 30);
      ctx.stroke();
    }
  });
}

export function paintSkirt(): HTMLCanvasElement {
  return paint(1024, 128, (ctx, w, h) => {
    ctx.fillStyle = linear(ctx, 0, h, ["#4a1a0a", "#2a0d05"]);
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = goldGradient(ctx, 0, 18);
    ctx.fillRect(0, 0, w, 14);
    for (let x = 40; x < w; x += 128) {
      ctx.beginPath();
      ctx.moveTo(x, h / 2 + 4);
      ctx.lineTo(x + 24, h / 2 - 20);
      ctx.lineTo(x + 48, h / 2 + 4);
      ctx.lineTo(x + 24, h / 2 + 28);
      ctx.closePath();
      ctx.fillStyle = "#c9932f";
      ctx.fill();
    }
  });
}

export function paintBeam(): HTMLCanvasElement {
  return paint(64, 256, (ctx, w, h) => {
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "rgba(255,240,200,0.9)");
    g.addColorStop(0.4, "rgba(255,230,180,0.35)");
    g.addColorStop(1, "rgba(255,220,160,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  });
}

export function paintHead(seed: number): HTMLCanvasElement {
  const rand = seeded(seed);
  return paint(200, 220, (ctx, w, h) => {
    ctx.fillStyle = "#0d0608";
    blob(ctx, w / 2, h - 10, 90, 70);
    ctx.fill();
    const r = 46 + rand() * 12;
    blob(ctx, w / 2, h - 95, r, r * 1.1);
    ctx.fill();
    const style = Math.floor(rand() * 4);
    if (style === 0) {
      blob(ctx, w / 2, h - 95 - r * 1.15, 22, 20);
      ctx.fill();
    } else if (style === 1) {
      ctx.fillRect(w / 2 - r * 1.2, h - 100 - r * 0.9, r * 2.4, 12);
      ctx.fillRect(w / 2 - r * 0.75, h - 120 - r * 1.2, r * 1.5, r * 0.6);
    } else if (style === 2) {
      blob(ctx, w / 2 - r, h - 95, 16, 16);
      ctx.fill();
      blob(ctx, w / 2 + r, h - 95, 16, 16);
      ctx.fill();
    }
  });
}
