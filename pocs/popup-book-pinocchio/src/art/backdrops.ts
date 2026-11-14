import { type Ctx, blob, fillInk, ink, linear, paint, seeded } from "./canvas";

export type DropId =
  | "night"
  | "workshop"
  | "sunroom"
  | "kitchen"
  | "tent"
  | "road"
  | "field"
  | "grove"
  | "carnival"
  | "storm"
  | "belly"
  | "sunrise";

const W = 2048;
const H = 1226;

function canvasGrain(ctx: Ctx, seed: number): void {
  const rand = seeded(seed);
  for (let i = 0; i < 6000; i++) {
    ctx.fillStyle = `rgba(${rand() > 0.5 ? "255,255,255" : "0,0,0"},${rand() * 0.05})`;
    ctx.fillRect(rand() * W, rand() * H, 3, 3);
  }
}

function sky(ctx: Ctx, stops: string[]): void {
  ctx.fillStyle = linear(ctx, 0, H, stops);
  ctx.fillRect(0, 0, W, H);
}

function stars(ctx: Ctx, count: number, seed: number): void {
  const rand = seeded(seed);
  for (let i = 0; i < count; i++) {
    const x = rand() * W;
    const y = rand() * H * 0.6;
    const r = 1.5 + rand() * 3.5;
    blob(ctx, x, y, r, r);
    ctx.fillStyle = `rgba(255,248,210,${0.5 + rand() * 0.5})`;
    ctx.fill();
  }
}

function moon(ctx: Ctx, x: number, y: number, r: number): void {
  const g = ctx.createRadialGradient(x, y, r * 0.6, x, y, r * 2.6);
  g.addColorStop(0, "rgba(255,245,200,0.5)");
  g.addColorStop(1, "rgba(255,245,200,0)");
  ctx.fillStyle = g;
  ctx.fillRect(x - r * 3, y - r * 3, r * 6, r * 6);
  blob(ctx, x, y, r, r);
  fillInk(ctx, "#fff4c4", 6);
  blob(ctx, x - r * 0.3, y - r * 0.2, r * 0.18, r * 0.18);
  ctx.fillStyle = "rgba(200,180,120,0.5)";
  ctx.fill();
}

function hills(ctx: Ctx, y: number, color: string, seed: number, amp = 60): void {
  const rand = seeded(seed);
  ctx.beginPath();
  ctx.moveTo(0, H);
  ctx.lineTo(0, y);
  for (let x = 0; x <= W; x += 256) ctx.quadraticCurveTo(x + 128, y - amp - rand() * amp, x + 256, y + (rand() - 0.5) * 30);
  ctx.lineTo(W, H);
  ctx.closePath();
  fillInk(ctx, color, 5);
}

function village(ctx: Ctx, y: number, color: string, windows: string, seed: number): void {
  const rand = seeded(seed);
  let x = 40;
  while (x < W) {
    const w = 110 + rand() * 120;
    const h = 120 + rand() * 180;
    ctx.beginPath();
    ctx.rect(x, y - h, w, h + 200);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(x - 14, y - h);
    ctx.lineTo(x + w / 2, y - h - 70);
    ctx.lineTo(x + w + 14, y - h);
    ctx.closePath();
    ctx.fill();
    if (rand() > 0.6) ctx.fillRect(x + w * 0.6, y - h - 90, 22, 60);
    for (let k = 0; k < 3; k++) {
      ctx.fillStyle = rand() > 0.35 ? windows : "rgba(0,0,0,0.2)";
      ctx.fillRect(x + 18 + k * (w / 3.4), y - h + 40 + (k % 2) * 50, 22, 30);
    }
    ctx.fillStyle = color;
    x += w + 30 + rand() * 60;
  }
}

function cypress(ctx: Ctx, x: number, y: number, h: number, color: string): void {
  ctx.beginPath();
  ctx.moveTo(x, y - h);
  ctx.quadraticCurveTo(x + h * 0.16, y - h * 0.4, x + h * 0.08, y);
  ctx.lineTo(x - h * 0.08, y);
  ctx.quadraticCurveTo(x - h * 0.16, y - h * 0.4, x, y - h);
  fillInk(ctx, color, 5);
}

function shelf(ctx: Ctx, x: number, y: number, w: number): void {
  ctx.fillStyle = "#6b4226";
  ctx.fillRect(x, y, w, 22);
  ctx.strokeRect(x, y, w, 22);
}

function clock(ctx: Ctx, x: number, y: number, r: number, seed: number): void {
  const rand = seeded(seed);
  ctx.beginPath();
  ctx.moveTo(x - r, y + r * 0.2);
  ctx.lineTo(x, y - r * 1.6);
  ctx.lineTo(x + r, y + r * 0.2);
  ctx.lineTo(x + r, y + r * 1.8);
  ctx.lineTo(x - r, y + r * 1.8);
  ctx.closePath();
  fillInk(ctx, "#8b5a2b", 5);
  blob(ctx, x, y + r * 0.4, r * 0.7, r * 0.7);
  fillInk(ctx, "#fff6d8", 4);
  ctx.beginPath();
  ctx.moveTo(x, y + r * 0.4);
  ctx.lineTo(x + Math.cos(rand() * 6) * r * 0.5, y + r * 0.4 + Math.sin(rand() * 6) * r * 0.5);
  ctx.moveTo(x, y + r * 0.4);
  ctx.lineTo(x, y);
  ink(ctx, 4);
}

function window(ctx: Ctx, x: number, y: number, w: number, h: number, view: string[]): void {
  ctx.fillStyle = linear(ctx, y, y + h, view);
  ctx.fillRect(x, y, w, h);
  ctx.lineWidth = 18;
  ctx.strokeStyle = "#5a3519";
  ctx.strokeRect(x, y, w, h);
  ctx.beginPath();
  ctx.moveTo(x + w / 2, y);
  ctx.lineTo(x + w / 2, y + h);
  ctx.moveTo(x, y + h / 2);
  ctx.lineTo(x + w, y + h / 2);
  ink(ctx, 12, "#5a3519");
}

function brickWall(ctx: Ctx, base: string, seed: number): void {
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, W, H);
  const rand = seeded(seed);
  for (let y = 0; y < H; y += 60) {
    for (let x = (y / 60) % 2 ? -60 : 0; x < W; x += 120) {
      ctx.fillStyle = `rgba(${120 + rand() * 60},${60 + rand() * 30},${30 + rand() * 20},0.35)`;
      ctx.fillRect(x + 3, y + 3, 114, 54);
    }
  }
}

function floorBand(ctx: Ctx, color: string): void {
  ctx.fillStyle = color;
  ctx.fillRect(0, H * 0.82, W, H * 0.18);
  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.fillRect(0, H * 0.82, W, 10);
}

function rays(ctx: Ctx, x: number, y: number, color: string, count: number): void {
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(a) * 2400, y + Math.sin(a) * 2400);
    ctx.lineTo(x + Math.cos(a + Math.PI / count) * 2400, y + Math.sin(a + Math.PI / count) * 2400);
    ctx.closePath();
    ctx.fillStyle = i % 2 ? color : "rgba(255,255,255,0)";
    ctx.fill();
  }
}

function bunting(ctx: Ctx, y: number, colors: string[]): void {
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.quadraticCurveTo(W / 2, y + 120, W, y);
  ink(ctx, 4);
  for (let i = 0; i < 26; i++) {
    const t = (i + 0.5) / 26;
    const x = t * W;
    const yy = y + 4 * t * (1 - t) * 120;
    ctx.beginPath();
    ctx.moveTo(x - 30, yy);
    ctx.lineTo(x + 30, yy);
    ctx.lineTo(x, yy + 70);
    ctx.closePath();
    fillInk(ctx, colors[i % colors.length] ?? "#e8505b", 3);
  }
}

const DROPS: Record<DropId, (ctx: Ctx) => void> = {
  night: (ctx) => {
    sky(ctx, ["#0b0f3a", "#2a1f63", "#5a2f6b"]);
    stars(ctx, 260, 3);
    moon(ctx, 1620, 260, 120);
    hills(ctx, 860, "#24183f", 4, 90);
    village(ctx, 940, "#1a1230", "#ffd77a", 5);
    hills(ctx, 1040, "#140d26", 6, 40);
  },
  workshop: (ctx) => {
    brickWall(ctx, "#8a5a3a", 7);
    window(ctx, 820, 160, 420, 360, ["#8fd3ff", "#d8f1ff"]);
    for (const y of [220, 520]) shelf(ctx, 120, y, 520);
    for (const y of [220, 520]) shelf(ctx, 1420, y, 520);
    [[200, 120], [330, 140], [470, 115], [1500, 130], [1660, 110], [1820, 135]].forEach(([x, y], i) => clock(ctx, x ?? 0, y ?? 0, 42, i + 1));
    [[220, 420], [380, 440], [540, 410], [1520, 420], [1700, 440], [1860, 410]].forEach(([x, y], i) => clock(ctx, x ?? 0, y ?? 0, 38, i + 9));
    ctx.fillStyle = "#5a3519";
    for (let i = 0; i < 6; i++) ctx.fillRect(760 + i * 90, 600, 14, 120);
    floorBand(ctx, "#6b4226");
  },
  sunroom: (ctx) => {
    sky(ctx, ["#ffe7a8", "#ffc46b"]);
    rays(ctx, W / 2, 380, "rgba(255,255,255,0.22)", 26);
    blob(ctx, W / 2, 380, 300, 300);
    ctx.fillStyle = "#bfe8ff";
    ctx.fill();
    ink(ctx, 26, "#7a4d22");
    ctx.beginPath();
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      ctx.moveTo(W / 2, 380);
      ctx.lineTo(W / 2 + Math.cos(a) * 300, 380 + Math.sin(a) * 300);
    }
    ink(ctx, 12, "#7a4d22");
    hills(ctx, 980, "#d98c3f", 8, 30);
    floorBand(ctx, "#b5703a");
  },
  kitchen: (ctx) => {
    brickWall(ctx, "#5c3a2a", 9);
    ctx.beginPath();
    ctx.moveTo(700, H * 0.82);
    ctx.lineTo(700, 420);
    ctx.lineTo(1348, 420);
    ctx.lineTo(1348, H * 0.82);
    fillInk(ctx, "#3a2418", 8);
    ctx.beginPath();
    ctx.moveTo(620, 420);
    ctx.lineTo(820, 120);
    ctx.lineTo(1228, 120);
    ctx.lineTo(1428, 420);
    ctx.closePath();
    fillInk(ctx, "#8a6a52", 8);
    const g = ctx.createRadialGradient(1024, 900, 20, 1024, 900, 340);
    g.addColorStop(0, "#fff3a0");
    g.addColorStop(0.4, "#ff8a2a");
    g.addColorStop(1, "rgba(120,20,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(740, 480, 568, 520);
    blob(ctx, 1024, 760, 140, 110);
    fillInk(ctx, "#2b2b2b", 8);
    ctx.fillStyle = "#c9c9d2";
    ctx.fillRect(140, 300, 300, 18);
    for (let i = 0; i < 4; i++) {
      blob(ctx, 180 + i * 75, 270, 30, 30);
      fillInk(ctx, ["#d6453a", "#ffd23f", "#4caf50", "#3a6fc4"][i] ?? "#d6453a", 4);
    }
    floorBand(ctx, "#4a2a18");
  },
  tent: (ctx) => {
    for (let i = 0; i < 16; i++) {
      ctx.beginPath();
      ctx.moveTo(W / 2, -200);
      ctx.lineTo((i / 16) * W * 1.4 - W * 0.2, H);
      ctx.lineTo(((i + 1) / 16) * W * 1.4 - W * 0.2, H);
      ctx.closePath();
      ctx.fillStyle = i % 2 ? "#ffefd6" : "#d8343b";
      ctx.fill();
    }
    bunting(ctx, 150, ["#ffd23f", "#3fa8e8", "#4caf50", "#ff6fa3"]);
    bunting(ctx, 330, ["#3fa8e8", "#ffd23f", "#ff6fa3", "#4caf50"]);
    floorBand(ctx, "#6b2a1a");
  },
  road: (ctx) => {
    sky(ctx, ["#ff8a5a", "#ffc87a", "#ffe8b0"]);
    blob(ctx, 1500, 700, 170, 170);
    ctx.fillStyle = "#fff0a0";
    ctx.fill();
    hills(ctx, 780, "#c9844a", 11, 80);
    hills(ctx, 900, "#a8662f", 12, 50);
    [[260, 910, 420], [400, 930, 340], [1760, 910, 460], [1880, 930, 360]].forEach(([x, y, h]) => cypress(ctx, x ?? 0, y ?? 0, h ?? 0, "#2f4a24"));
    ctx.beginPath();
    ctx.moveTo(880, 860);
    ctx.lineTo(1160, 860);
    ctx.lineTo(1700, H);
    ctx.lineTo(340, H);
    ctx.closePath();
    fillInk(ctx, "#e6c08a", 5);
    floorBand(ctx, "#b57a3e");
  },
  field: (ctx) => {
    sky(ctx, ["#120a35", "#36206b", "#6b3a8a"]);
    stars(ctx, 220, 13);
    moon(ctx, 420, 260, 150);
    hills(ctx, 820, "#2a1f4f", 14, 70);
    for (let i = 0; i < 18; i++) {
      ctx.beginPath();
      ctx.moveTo(i * 120, 900);
      ctx.lineTo(i * 120 + 60, 860);
      ctx.lineTo(i * 120 + 120, 900);
      ink(ctx, 5, "#4a3a7a");
    }
    floorBand(ctx, "#2a1f4f");
  },
  grove: (ctx) => {
    sky(ctx, ["#1e3a6b", "#3a6fa8", "#8fd3e8"]);
    stars(ctx, 120, 17);
    for (let i = 0; i < 9; i++) {
      const x = 80 + i * 240;
      ctx.fillStyle = "#1e3a2a";
      ctx.fillRect(x - 30, 300, 60, 800);
      blob(ctx, x, 280, 190, 160);
      fillInk(ctx, i % 2 ? "#2a5a3a" : "#24503a", 5);
    }
    ctx.beginPath();
    ctx.rect(860, 520, 330, 330);
    fillInk(ctx, "#bfe8ff", 6);
    ctx.beginPath();
    ctx.moveTo(820, 520);
    ctx.lineTo(1025, 360);
    ctx.lineTo(1230, 520);
    ctx.closePath();
    fillInk(ctx, "#3fa8d9", 6);
    window(ctx, 940, 600, 160, 130, ["#fff3a0", "#ffd23f"]);
    floorBand(ctx, "#2a4a2a");
  },
  carnival: (ctx) => {
    sky(ctx, ["#ff8ad8", "#ffd0a0", "#fff3c4"]);
    ctx.beginPath();
    ctx.arc(1560, 560, 380, 0, Math.PI * 2);
    ink(ctx, 16, "#a83a8a");
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(1560, 560);
      ctx.lineTo(1560 + Math.cos(a) * 380, 560 + Math.sin(a) * 380);
      ink(ctx, 6, "#a83a8a");
      blob(ctx, 1560 + Math.cos(a) * 380, 560 + Math.sin(a) * 380 + 30, 40, 34);
      fillInk(ctx, ["#ffd23f", "#3fa8e8", "#4caf50", "#e8505b"][i % 4] ?? "#ffd23f", 4);
    }
    for (let i = 0; i < 5; i++) {
      const x = 120 + i * 200;
      ctx.beginPath();
      ctx.moveTo(x, 900);
      ctx.lineTo(x + 80, 640);
      ctx.lineTo(x + 160, 900);
      ctx.closePath();
      fillInk(ctx, i % 2 ? "#ffefd6" : "#e8505b", 5);
    }
    bunting(ctx, 90, ["#e8505b", "#ffd23f", "#3fa8e8", "#4caf50"]);
    floorBand(ctx, "#d98ab5");
  },
  storm: (ctx) => {
    sky(ctx, ["#1a2238", "#33405e", "#4a6080"]);
    for (let i = 0; i < 7; i++) {
      blob(ctx, 150 + i * 300, 160 + (i % 2) * 60, 260, 110);
      ctx.fillStyle = "rgba(20,25,40,0.7)";
      ctx.fill();
    }
    ctx.beginPath();
    ctx.moveTo(1400, 140);
    ctx.lineTo(1330, 360);
    ctx.lineTo(1400, 360);
    ctx.lineTo(1320, 600);
    ink(ctx, 14, "#fff7a0");
    for (let row = 0; row < 7; row++) {
      ctx.beginPath();
      const y = 620 + row * 90;
      ctx.moveTo(0, y);
      for (let x = 0; x <= W; x += 128) ctx.quadraticCurveTo(x + 64, y - 50, x + 128, y);
      ctx.lineTo(W, H);
      ctx.lineTo(0, H);
      ctx.closePath();
      fillInk(ctx, ["#1e4a7a", "#235a8a", "#2a6a9a", "#2f7aa8"][row % 4] ?? "#1e4a7a", 5);
    }
  },
  belly: (ctx) => {
    const g = ctx.createRadialGradient(W / 2, H * 0.6, 60, W / 2, H * 0.6, 1200);
    g.addColorStop(0, "#7a2a4a");
    g.addColorStop(1, "#1a0614");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
    for (let i = 0; i < 9; i++) {
      const x = 120 + i * 225;
      ctx.beginPath();
      ctx.moveTo(x, -20);
      ctx.quadraticCurveTo(x + (i - 4) * 60, H * 0.5, x + (i - 4) * 20, H);
      ink(ctx, 34, "rgba(255,190,200,0.35)");
    }
    ctx.fillStyle = "rgba(60,120,160,0.5)";
    ctx.fillRect(0, H * 0.84, W, H * 0.16);
  },
  sunrise: (ctx) => {
    sky(ctx, ["#7ec8f2", "#ffd0e0", "#ffe8a8"]);
    rays(ctx, W / 2, 940, "rgba(255,255,255,0.25)", 30);
    const colors = ["#ff5d5d", "#ffa23f", "#ffe24f", "#5ccf6a", "#4aa8ff", "#9b6bff"];
    colors.forEach((c, i) => {
      ctx.beginPath();
      ctx.arc(W / 2, 1100, 900 - i * 42, Math.PI, 0);
      ctx.lineWidth = 42;
      ctx.strokeStyle = c;
      ctx.stroke();
    });
    blob(ctx, W / 2, 940, 160, 160);
    ctx.fillStyle = "#fff3a0";
    ctx.fill();
    hills(ctx, 900, "#7ac46a", 21, 60);
    village(ctx, 1000, "#f2d6a8", "#ffffff", 22);
    floorBand(ctx, "#5aa84a");
  },
};

export function paintDrop(id: DropId): HTMLCanvasElement {
  return paint(W, H, (ctx) => {
    DROPS[id](ctx);
    canvasGrain(ctx, id.length * 13);
  });
}
