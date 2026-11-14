import { heart, spade } from "./characters";
import { blob, type Ctx, fillInk, ink, linear, paint, radial, seeded } from "./canvas";

export const SW = 1600;
export const SH = 1024;
const LINE = "#3b2416";

function grain(ctx: Ctx, seed: number): void {
  const rand = seeded(seed);
  ctx.save();
  ctx.globalCompositeOperation = "source-atop";
  for (let i = 0; i < 2600; i++) {
    ctx.fillStyle = rand() > 0.5 ? "rgba(255,255,255,0.06)" : "rgba(60,30,10,0.06)";
    ctx.fillRect(rand() * SW, rand() * SH, 2 + rand() * 3, 2 + rand() * 3);
  }
  ctx.restore();
}

function cut(ctx: Ctx, draw: (c: Ctx) => void): void {
  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = "#000";
  draw(ctx);
  ctx.restore();
}

function sheet(seed: number, draw: (ctx: Ctx) => void): HTMLCanvasElement {
  return paint(SW, SH, (ctx) => {
    draw(ctx);
    grain(ctx, seed);
  });
}

export function leaf(ctx: Ctx, x: number, y: number, r: number, a: number, color: string): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(a);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.quadraticCurveTo(r * 0.6, -r * 0.45, r, 0);
  ctx.quadraticCurveTo(r * 0.6, r * 0.45, 0, 0);
  fillInk(ctx, color, 2);
  ctx.restore();
}

export function flower(ctx: Ctx, x: number, y: number, r: number, petal: string): void {
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    blob(ctx, x + Math.cos(a) * r * 0.7, y + Math.sin(a) * r * 0.7, r * 0.5, r * 0.5);
    fillInk(ctx, petal, 2);
  }
  blob(ctx, x, y, r * 0.4, r * 0.4);
  fillInk(ctx, "#ffd43b", 2);
}

function cloud(ctx: Ctx, x: number, y: number, s: number): void {
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  for (const [dx, dy, r] of [
    [0, 0, 60],
    [60, -20, 70],
    [130, 0, 56],
    [70, 20, 60],
  ] as const) {
    blob(ctx, x + dx * s, y + dy * s, r * s, r * s * 0.8);
    ctx.fill();
  }
}

function hills(ctx: Ctx, y: number, color: string, amp: number, freq: number, phase: number): void {
  ctx.beginPath();
  ctx.moveTo(0, SH);
  for (let x = 0; x <= SW; x += 20) ctx.lineTo(x, y + Math.sin(x * freq + phase) * amp + Math.sin(x * freq * 2.3) * amp * 0.3);
  ctx.lineTo(SW, SH);
  ctx.closePath();
  fillInk(ctx, color, 4);
}

function grassEdge(ctx: Ctx, y: number, color: string, seed: number): void {
  const rand = seeded(seed);
  ctx.beginPath();
  ctx.moveTo(0, SH);
  ctx.lineTo(0, y);
  for (let x = 0; x <= SW; x += 22) {
    ctx.lineTo(x + 11, y - 20 - rand() * 34);
    ctx.lineTo(x + 22, y - rand() * 8);
  }
  ctx.lineTo(SW, SH);
  ctx.closePath();
  fillInk(ctx, color, 4);
}

function tree(ctx: Ctx, x: number, base: number, h: number, crown: number, leafColor: string, seed: number): void {
  ctx.beginPath();
  ctx.moveTo(x - 46, base);
  ctx.quadraticCurveTo(x - 20, base - h * 0.5, x - 26, base - h);
  ctx.lineTo(x + 26, base - h);
  ctx.quadraticCurveTo(x + 20, base - h * 0.5, x + 46, base);
  ctx.closePath();
  fillInk(ctx, linear(ctx, base - h, base, ["#7a4b2a", "#5a341c"]), 4);
  const rand = seeded(seed);
  for (let i = 0; i < 26; i++) {
    const a = rand() * Math.PI * 2;
    const r = Math.sqrt(rand()) * crown;
    const cx = x + Math.cos(a) * r;
    const cy = base - h - Math.sin(a) * r * 0.6;
    blob(ctx, cx, cy, crown * 0.42, crown * 0.36);
    fillInk(ctx, radial(ctx, cx, cy, crown * 0.42, "#9ad06a", leafColor), 3);
  }
}

export interface FrameOptions {
  paper: string;
  trim: string;
  shape: "arch" | "oval" | "keyhole" | "rect";
  motif: "vines" | "roses" | "wood" | "forest" | "tent" | "grass" | "cards";
  title?: string;
  seed: number;
}

function opening(ctx: Ctx, shape: FrameOptions["shape"]): void {
  ctx.beginPath();
  if (shape === "oval") ctx.ellipse(SW / 2, SH / 2 + 30, 640, 400, 0, 0, Math.PI * 2);
  else if (shape === "keyhole") {
    ctx.arc(SW / 2, 380, 300, Math.PI * 0.82, Math.PI * 2.18);
    ctx.lineTo(SW / 2 + 520, SH - 60);
    ctx.lineTo(SW / 2 - 520, SH - 60);
    ctx.closePath();
  } else if (shape === "arch") {
    ctx.moveTo(130, SH - 40);
    ctx.lineTo(130, 420);
    ctx.bezierCurveTo(130, 100, SW - 130, 100, SW - 130, 420);
    ctx.lineTo(SW - 130, SH - 40);
    ctx.closePath();
  } else ctx.roundRect(110, 120, SW - 220, SH - 170, 40);
  ctx.fill();
}

export function frame(options: FrameOptions): HTMLCanvasElement {
  const rand = seeded(options.seed);
  return sheet(options.seed, (ctx) => {
    ctx.fillStyle = options.paper;
    ctx.fillRect(0, 0, SW, SH);
    if (options.motif === "wood") {
      ctx.strokeStyle = "rgba(40,20,8,0.35)";
      for (let x = 0; x < SW; x += 64) {
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, SH);
        ctx.stroke();
      }
    }
    if (options.motif === "tent") {
      for (let x = 0; x < SW; x += 120) {
        ctx.fillStyle = "#fffaf0";
        ctx.fillRect(x, 0, 60, SH);
      }
    }
    if (options.motif === "cards") {
      for (let i = 0; i < 40; i++) {
        const x = rand() * SW;
        const y = rand() * SH;
        if (i % 2 === 0) heart(ctx, x, y, 18, "rgba(200,30,40,0.5)");
        else spade(ctx, x, y, 16, "rgba(40,30,30,0.45)");
      }
    }
    cut(ctx, (c) => opening(c, options.shape));
    ctx.save();
    ctx.globalCompositeOperation = "source-atop";
    ctx.lineWidth = 26;
    ctx.strokeStyle = options.trim;
    ctx.beginPath();
    if (options.shape === "oval") ctx.ellipse(SW / 2, SH / 2 + 30, 652, 412, 0, 0, Math.PI * 2);
    else if (options.shape === "rect") ctx.roundRect(98, 108, SW - 196, SH - 146, 46);
    ctx.stroke();
    ctx.restore();
    if (options.motif === "vines" || options.motif === "roses" || options.motif === "grass" || options.motif === "forest") {
      ctx.save();
      ctx.globalCompositeOperation = "source-atop";
      for (let i = 0; i < 160; i++) {
        const side = rand();
        const x = side < 0.5 ? rand() * 200 + (side < 0.25 ? 0 : SW - 200) : rand() * SW;
        const y = side < 0.5 ? rand() * SH : rand() < 0.5 ? rand() * 150 : SH - rand() * 120;
        if (options.motif === "forest") leaf(ctx, x, y, 40, rand() * 6, rand() > 0.5 ? "#2f5a3a" : "#3e7046");
        else if (options.motif === "grass") leaf(ctx, x, y, 70, -Math.PI / 2 + (rand() - 0.5), rand() > 0.5 ? "#6aa84f" : "#4c8a3a");
        else leaf(ctx, x, y, 36, rand() * 6, rand() > 0.5 ? "#5a9a4a" : "#3f7a3a");
        if (i % 6 === 0) {
          if (options.motif === "roses") {
            blob(ctx, x, y, 22, 20);
            fillInk(ctx, rand() > 0.5 ? "#e03131" : "#fffaf0", 3);
          } else if (options.motif !== "forest") flower(ctx, x, y, 16, rand() > 0.5 ? "#f783ac" : "#b197fc");
        }
      }
      ctx.restore();
    }
    if (options.title) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(SW / 2 - 470, 26);
      ctx.lineTo(SW / 2 + 470, 26);
      ctx.lineTo(SW / 2 + 440, 66);
      ctx.lineTo(SW / 2 + 470, 106);
      ctx.lineTo(SW / 2 - 470, 106);
      ctx.lineTo(SW / 2 - 440, 66);
      ctx.closePath();
      fillInk(ctx, "#fffaf0", 5);
      ctx.fillStyle = "#8a2b2b";
      ctx.font = "700 54px 'Fredericka the Great'";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(options.title, SW / 2, 68, 860);
      ctx.restore();
    }
  });
}

export function sky(top: string, bottom: string, seed: number, clouds = true, stars = false): HTMLCanvasElement {
  const rand = seeded(seed);
  return sheet(seed, (ctx) => {
    ctx.fillStyle = linear(ctx, 0, SH, [top, bottom]);
    ctx.fillRect(0, 0, SW, SH);
    if (stars) {
      ctx.fillStyle = "#fff6d0";
      for (let i = 0; i < 120; i++) {
        blob(ctx, rand() * SW, rand() * SH * 0.7, 1.5 + rand() * 2.5, 1.5 + rand() * 2.5);
        ctx.fill();
      }
      blob(ctx, 1240, 210, 70, 70);
      ctx.fillStyle = "#fff3c4";
      ctx.fill();
    }
    if (clouds) for (let i = 0; i < 5; i++) cloud(ctx, rand() * SW, 120 + rand() * 260, 0.8 + rand() * 0.8);
    hills(ctx, 760, "#9cc97a", 30, 0.004, seed);
    hills(ctx, 840, "#7db35e", 24, 0.006, seed + 2);
  });
}

export function riverLayer(): HTMLCanvasElement {
  return sheet(31, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(0, 700);
    for (let x = 0; x <= SW; x += 40) ctx.lineTo(x, 700 + Math.sin(x * 0.006) * 18);
    ctx.lineTo(SW, 860);
    ctx.lineTo(0, 860);
    ctx.closePath();
    fillInk(ctx, linear(ctx, 690, 860, ["#8fd0f0", "#4a9ad0"]), 4);
    ctx.strokeStyle = "rgba(255,255,255,0.7)";
    ctx.lineWidth = 4;
    for (let i = 0; i < 18; i++) {
      const x = (i * 97) % SW;
      const y = 730 + (i % 4) * 28;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.quadraticCurveTo(x + 30, y - 8, x + 60, y);
      ctx.stroke();
    }
    grassEdge(ctx, 880, "#6aa84f", 3);
    tree(ctx, 1380, 720, 160, 120, "#4c8a3a", 7);
    tree(ctx, 240, 720, 120, 90, "#4c8a3a", 9);
  });
}

export function bankLayer(withTree: boolean): HTMLCanvasElement {
  return sheet(41, (ctx) => {
    grassEdge(ctx, 900, "#5c9a44", 11);
    const rand = seeded(13);
    for (let i = 0; i < 26; i++) flower(ctx, rand() * SW, 920 + rand() * 80, 12 + rand() * 6, rand() > 0.5 ? "#ffffff" : "#ffd8e8");
    if (withTree) {
      tree(ctx, 180, 1000, 640, 260, "#3f7a3a", 21);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(SW, 0);
      ctx.lineTo(SW, 60);
      for (let x = SW; x > 0; x -= 80) ctx.quadraticCurveTo(x - 40, 150 + (x % 160 === 0 ? 30 : 0), x - 80, 60);
      ctx.closePath();
      fillInk(ctx, "#3f7a3a", 4);
      for (let i = 0; i < 70; i++) leaf(ctx, rand() * SW, 30 + rand() * 90, 40, rand() * 6, rand() > 0.5 ? "#6aa84f" : "#4c8a3a");
    }
  });
}

export function hedgeLayer(hole: boolean): HTMLCanvasElement {
  return sheet(51, (ctx) => {
    const rand = seeded(17);
    ctx.beginPath();
    ctx.moveTo(0, SH);
    ctx.lineTo(0, 520);
    for (let x = 0; x <= SW; x += 60) ctx.quadraticCurveTo(x + 30, 470 + rand() * 30, x + 60, 520);
    ctx.lineTo(SW, SH);
    ctx.closePath();
    fillInk(ctx, linear(ctx, 470, SH, ["#5a9a4a", "#2f5a3a"]), 4);
    for (let i = 0; i < 90; i++) leaf(ctx, rand() * SW, 520 + rand() * 360, 34, rand() * 6, rand() > 0.5 ? "#6aa84f" : "#3f7a3a");
    if (hole) {
      blob(ctx, 1220, 860, 150, 110);
      fillInk(ctx, radial(ctx, 1220, 880, 150, "#1a0f08", "#3a2414"), 6);
      ctx.strokeStyle = "#6b4226";
      ctx.lineWidth = 16;
      for (const dx of [-140, -60, 40, 130]) {
        ctx.beginPath();
        ctx.moveTo(1220 + dx, 760);
        ctx.quadraticCurveTo(1220 + dx * 1.3, 820, 1220 + dx * 1.1, 900);
        ctx.stroke();
      }
    }
    cut(ctx, (c) => {
      c.fillRect(0, 980, SW, 44);
    });
  });
}

export function earthRing(seed: number): HTMLCanvasElement {
  const rand = seeded(seed);
  return sheet(seed, (ctx) => {
    ctx.fillStyle = radial(ctx, SW / 2, SH / 2, 900, "#8a5a36", "#4a2c18");
    ctx.fillRect(0, 0, SW, SH);
    for (let i = 0; i < 40; i++) {
      blob(ctx, rand() * SW, rand() * SH, 10 + rand() * 30, 8 + rand() * 20);
      ctx.fillStyle = rand() > 0.5 ? "rgba(40,20,8,0.35)" : "rgba(180,130,80,0.3)";
      ctx.fill();
    }
    const items = ["shelf", "map", "clock", "lamp", "frame"];
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2 + rand();
      const x = SW / 2 + Math.cos(a) * 610;
      const y = SH / 2 + Math.sin(a) * 400;
      const kind = items[(i + seed) % items.length];
      ctx.save();
      ctx.translate(x, y);
      if (kind === "shelf") {
        ctx.fillStyle = "#c08a50";
        ctx.fillRect(-90, 20, 180, 14);
        for (let j = 0; j < 3; j++) {
          ctx.beginPath();
          ctx.roundRect(-80 + j * 56, -40, 44, 60, 8);
          fillInk(ctx, j === 1 ? "#f08c00" : "#e8f4ff", 3);
        }
      } else if (kind === "map") {
        ctx.beginPath();
        ctx.rect(-80, -60, 160, 110);
        fillInk(ctx, "#f3e2b0", 3);
        ctx.strokeStyle = "#7a5a2a";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(-60, -20);
        ctx.quadraticCurveTo(0, -60, 50, 10);
        ctx.stroke();
      } else if (kind === "clock") {
        blob(ctx, 0, 0, 56, 56);
        fillInk(ctx, "#fffaf0", 5);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, -40);
        ctx.moveTo(0, 0);
        ctx.lineTo(26, 10);
        ink(ctx, 4, LINE);
      } else if (kind === "lamp") {
        ctx.beginPath();
        ctx.moveTo(-30, 30);
        ctx.lineTo(30, 30);
        ctx.lineTo(16, -30);
        ctx.lineTo(-16, -30);
        ctx.closePath();
        fillInk(ctx, "#ffd43b", 3);
      } else {
        ctx.beginPath();
        ctx.ellipse(0, 0, 52, 66, 0, 0, Math.PI * 2);
        fillInk(ctx, "#d4a24c", 6);
        blob(ctx, 0, 0, 38, 50);
        fillInk(ctx, "#b5c9e8", 2);
      }
      ctx.restore();
    }
    cut(ctx, (c) => {
      c.beginPath();
      const r = 470 + (seed % 3) * 20;
      for (let i = 0; i <= 40; i++) {
        const a = (i / 40) * Math.PI * 2;
        const rr = r + Math.sin(a * 7 + seed) * 26;
        c.lineTo(SW / 2 + Math.cos(a) * rr * 1.18, SH / 2 + Math.sin(a) * rr * 0.86);
      }
      c.fill();
    });
  });
}

export function deepGlow(): HTMLCanvasElement {
  return sheet(61, (ctx) => {
    ctx.fillStyle = radial(ctx, SW / 2, SH / 2, 800, "#fff1c9", "#2a160c");
    ctx.fillRect(0, 0, SW, SH);
  });
}

export function hallDoors(): HTMLCanvasElement {
  return sheet(71, (ctx) => {
    ctx.fillStyle = linear(ctx, 0, SH, ["#5a2e3e", "#3a1a28"]);
    ctx.fillRect(0, 0, SW, SH);
    ctx.fillStyle = "rgba(255,220,180,0.06)";
    for (let x = 0; x < SW; x += 80) ctx.fillRect(x, 0, 40, SH);
    for (const x of [200, 520, 1080, 1400]) {
      ctx.beginPath();
      ctx.moveTo(x - 100, 900);
      ctx.lineTo(x - 100, 360);
      ctx.quadraticCurveTo(x, 240, x + 100, 360);
      ctx.lineTo(x + 100, 900);
      ctx.closePath();
      fillInk(ctx, linear(ctx, 260, 900, ["#a0663a", "#6b3e22"]), 6);
      blob(ctx, x + 70, 640, 12, 12);
      fillInk(ctx, "#ffd43b", 3);
    }
    ctx.beginPath();
    ctx.moveTo(740, 900);
    ctx.lineTo(740, 780);
    ctx.quadraticCurveTo(800, 730, 860, 780);
    ctx.lineTo(860, 900);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 800, 840, 90, "#fff6b0", "#7bc96f"), 5);
    for (let i = 0; i < 6; i++) flower(ctx, 760 + i * 18, 880 - (i % 2) * 20, 8, "#f783ac");
    ctx.fillStyle = "#2a1420";
    ctx.fillRect(0, 900, SW, SH - 900);
    for (let x = 0; x < SW; x += 100) {
      ctx.fillStyle = (x / 100) % 2 === 0 ? "#efe2c8" : "#2a1420";
      ctx.fillRect(x, 900, 100, 62);
      ctx.fillStyle = (x / 100) % 2 === 0 ? "#2a1420" : "#efe2c8";
      ctx.fillRect(x, 962, 100, 62);
    }
  });
}

export function curtains(color: string): HTMLCanvasElement {
  return sheet(81, (ctx) => {
    for (const side of [0, 1]) {
      ctx.beginPath();
      const x0 = side === 0 ? 0 : SW;
      const x1 = side === 0 ? 300 : SW - 300;
      ctx.moveTo(x0, 0);
      ctx.lineTo(x1, 0);
      ctx.quadraticCurveTo(x1 + (side === 0 ? -60 : 60), 500, x1 + (side === 0 ? -150 : 150), SH);
      ctx.lineTo(x0, SH);
      ctx.closePath();
      fillInk(ctx, color, 5);
      ctx.strokeStyle = "rgba(0,0,0,0.18)";
      ctx.lineWidth = 8;
      for (let i = 1; i < 5; i++) {
        const x = side === 0 ? i * 52 : SW - i * 52;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.quadraticCurveTo(x + (side === 0 ? -20 : 20), 500, x + (side === 0 ? -60 : 60), SH);
        ctx.stroke();
      }
    }
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(SW, 0);
    ctx.lineTo(SW, 90);
    for (let x = SW; x > 0; x -= 100) ctx.quadraticCurveTo(x - 50, 170, x - 100, 90);
    ctx.closePath();
    fillInk(ctx, color, 5);
  });
}

export function tallFlowers(seed: number, dusk: boolean): HTMLCanvasElement {
  const rand = seeded(seed);
  return sheet(seed, (ctx) => {
    for (let i = 0; i < 14; i++) {
      const x = rand() * SW;
      const top = 200 + rand() * 360;
      ctx.beginPath();
      ctx.moveTo(x, SH);
      ctx.quadraticCurveTo(x + (rand() - 0.5) * 80, (SH + top) / 2, x, top);
      ink(ctx, 12, dusk ? "#2f5a3a" : "#4c8a3a");
      leaf(ctx, x, (SH + top) / 2, 90, -0.6 - rand() * 0.6, dusk ? "#3e7046" : "#6aa84f");
      leaf(ctx, x, (SH + top) / 2 + 80, 90, Math.PI + 0.6 + rand() * 0.6, dusk ? "#3e7046" : "#6aa84f");
      flower(ctx, x, top, 40 + rand() * 30, ["#f783ac", "#b197fc", "#ffa94d", "#74c0fc"][i % 4] ?? "#f783ac");
    }
    grassEdge(ctx, 960, dusk ? "#2f5a3a" : "#5c9a44", seed);
  });
}

export function forestLayer(seed: number, dark: boolean): HTMLCanvasElement {
  const rand = seeded(seed);
  return sheet(seed, (ctx) => {
    for (let i = 0; i < 6; i++) {
      const x = 80 + rand() * (SW - 160);
      ctx.beginPath();
      ctx.moveTo(x - 50, SH);
      ctx.bezierCurveTo(x - 10, 700, x - 90, 400, x - 20, 0);
      ctx.lineTo(x + 30, 0);
      ctx.bezierCurveTo(x - 30, 400, x + 50, 700, x + 50, SH);
      ctx.closePath();
      fillInk(ctx, dark ? "#2a1d3a" : "#4a3020", 4);
      for (let j = 0; j < 3; j++) {
        const y = 200 + j * 180 + rand() * 60;
        const dir = rand() > 0.5 ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.bezierCurveTo(x + dir * 80, y - 40, x + dir * 140, y + 20, x + dir * 220, y - 50);
        ink(ctx, 16, dark ? "#2a1d3a" : "#4a3020");
      }
    }
    grassEdge(ctx, 980, dark ? "#1f3a2a" : "#3f7a3a", seed);
  });
}

export function catBranch(): HTMLCanvasElement {
  return sheet(91, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(1500, SH);
    ctx.bezierCurveTo(1440, 700, 1520, 500, 1460, 0);
    ctx.lineTo(1600, 0);
    ctx.lineTo(1600, SH);
    ctx.closePath();
    fillInk(ctx, "#3a2a1a", 5);
    ctx.beginPath();
    ctx.moveTo(1480, 430);
    ctx.bezierCurveTo(1200, 380, 900, 470, 560, 410);
    ctx.lineTo(560, 450);
    ctx.bezierCurveTo(900, 510, 1200, 430, 1480, 490);
    ctx.closePath();
    fillInk(ctx, "#4a3420", 5);
    const rand = seeded(5);
    for (let i = 0; i < 40; i++) leaf(ctx, 560 + rand() * 900, 380 + rand() * 120, 36, rand() * 6, rand() > 0.5 ? "#2f5a3a" : "#4c8a3a");
  });
}

export function signpost(): HTMLCanvasElement {
  return paint(360, 560, (ctx) => {
    ctx.fillStyle = "#6b4226";
    ctx.fillRect(160, 80, 36, 480);
    for (const [y, dir, text] of [
      [130, 1, "THIS WAY"],
      [250, -1, "THAT WAY"],
    ] as const) {
      ctx.beginPath();
      const x0 = dir === 1 ? 60 : 300;
      ctx.moveTo(x0, y);
      ctx.lineTo(x0 + dir * 220, y);
      ctx.lineTo(x0 + dir * 270, y + 40);
      ctx.lineTo(x0 + dir * 220, y + 80);
      ctx.lineTo(x0, y + 80);
      ctx.closePath();
      fillInk(ctx, "#e8c88a", 4);
      ctx.fillStyle = LINE;
      ctx.font = "700 34px 'IM Fell English SC'";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, x0 + dir * 120, y + 42);
    }
  });
}

export function hareHouse(): HTMLCanvasElement {
  return sheet(101, (ctx) => {
    tree(ctx, 220, 900, 260, 170, "#4c8a3a", 3);
    ctx.beginPath();
    ctx.moveTo(900, 900);
    ctx.lineTo(900, 520);
    ctx.lineTo(1150, 330);
    ctx.lineTo(1400, 520);
    ctx.lineTo(1400, 900);
    ctx.closePath();
    fillInk(ctx, "#f3d9a4", 5);
    ctx.beginPath();
    ctx.moveTo(870, 540);
    ctx.lineTo(1150, 300);
    ctx.lineTo(1430, 540);
    ctx.quadraticCurveTo(1150, 470, 870, 540);
    fillInk(ctx, "#a8744a", 5);
    for (const [x, t] of [
      [1070, -0.2],
      [1220, 0.2],
    ] as const) {
      ctx.save();
      ctx.translate(x, 360);
      ctx.rotate(t);
      blob(ctx, 0, -90, 34, 100);
      fillInk(ctx, "#b5835a", 5);
      blob(ctx, 0, -86, 16, 76);
      ctx.fillStyle = "#f2c1a8";
      ctx.fill();
      ctx.restore();
    }
    ctx.beginPath();
    ctx.roundRect(1100, 700, 100, 200, 40);
    fillInk(ctx, "#8a5a2a", 4);
    for (const x of [970, 1280]) {
      ctx.beginPath();
      ctx.roundRect(x, 600, 80, 80, 10);
      fillInk(ctx, "#bfe3ff", 4);
    }
    grassEdge(ctx, 930, "#5c9a44", 23);
  });
}

export function teaTable(): HTMLCanvasElement {
  return paint(1500, 320, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(20, 60);
    ctx.lineTo(1480, 60);
    ctx.lineTo(1460, 250);
    for (let x = 1460; x > 40; x -= 70) ctx.quadraticCurveTo(x - 35, 290, x - 70, 250);
    ctx.lineTo(40, 250);
    ctx.closePath();
    fillInk(ctx, "#fffaf0", 5);
    ctx.fillStyle = "rgba(214,51,108,0.25)";
    for (let x = 40; x < 1460; x += 90) ctx.fillRect(x, 60, 45, 190);
    ctx.fillStyle = "#6b4226";
    ctx.fillRect(120, 250, 26, 70);
    ctx.fillRect(1350, 250, 26, 70);
  });
}

export function roseBush(): HTMLCanvasElement {
  return paint(520, 420, (ctx) => {
    const rand = seeded(7);
    for (let i = 0; i < 26; i++) {
      blob(ctx, 60 + rand() * 400, 120 + rand() * 260, 70, 60);
      fillInk(ctx, rand() > 0.5 ? "#3f7a3a" : "#2f6a32", 3);
    }
    for (let i = 0; i < 40; i++) leaf(ctx, 40 + rand() * 440, 100 + rand() * 300, 34, rand() * 6, "#5a9a4a");
  });
}

export function rose(): HTMLCanvasElement {
  return paint(90, 90, (ctx) => {
    blob(ctx, 45, 45, 38, 36);
    fillInk(ctx, "#ffffff", 4);
    ctx.beginPath();
    ctx.arc(45, 45, 22, 0, Math.PI * 1.6);
    ink(ctx, 4, "rgba(60,30,30,0.5)");
    ctx.beginPath();
    ctx.arc(45, 45, 10, Math.PI, Math.PI * 2.6);
    ink(ctx, 4, "rgba(60,30,30,0.5)");
  });
}

export function gardenBack(): HTMLCanvasElement {
  return sheet(111, (ctx) => {
    ctx.fillStyle = linear(ctx, 0, SH, ["#9fd3ff", "#e8f6ff"]);
    ctx.fillRect(0, 0, SW, SH);
    cloud(ctx, 200, 150, 1);
    cloud(ctx, 1100, 110, 0.8);
    ctx.fillStyle = "#e9d8f5";
    ctx.beginPath();
    ctx.rect(560, 330, 480, 420);
    fillInk(ctx, "#f1e3d0", 5);
    for (const x of [560, 800, 1040]) {
      ctx.beginPath();
      ctx.rect(x - 60, 220, 120, 530);
      fillInk(ctx, "#f1e3d0", 5);
      ctx.beginPath();
      ctx.moveTo(x - 80, 230);
      ctx.lineTo(x, 110);
      ctx.lineTo(x + 80, 230);
      ctx.closePath();
      fillInk(ctx, "#e03131", 5);
      heart(ctx, x, 320, 22, "#e03131");
    }
    for (let x = 60; x < SW; x += 180) {
      if (x > 500 && x < 1100) continue;
      ctx.fillStyle = "#6b4226";
      ctx.fillRect(x - 8, 660, 16, 140);
      blob(ctx, x, 600, 70, 90);
      fillInk(ctx, "#4c8a3a", 4);
    }
    hills(ctx, 800, "#7db35e", 10, 0.004, 1);
  });
}

export function hoops(): HTMLCanvasElement {
  return sheet(121, (ctx) => {
    for (const x of [380, 800, 1220]) {
      ctx.save();
      ctx.translate(x, 760);
      ctx.beginPath();
      ctx.roundRect(-70, -110, 140, 190, 12);
      fillInk(ctx, "#fffdf6", 4);
      heart(ctx, 0, -16, 26, "#e03131");
      ctx.beginPath();
      ctx.moveTo(-60, 80);
      ctx.lineTo(-60, 200);
      ctx.moveTo(60, 80);
      ctx.lineTo(60, 200);
      ink(ctx, 10, LINE);
      ctx.restore();
    }
    grassEdge(ctx, 990, "#4c8a3a", 31);
  });
}

export function lawn(): HTMLCanvasElement {
  return sheet(131, (ctx) => {
    ctx.fillStyle = linear(ctx, 820, SH, ["#7cc35e", "#4c8a3a"]);
    ctx.beginPath();
    ctx.moveTo(0, 860);
    ctx.quadraticCurveTo(SW / 2, 800, SW, 860);
    ctx.lineTo(SW, SH);
    ctx.lineTo(0, SH);
    ctx.closePath();
    fillInk(ctx, linear(ctx, 820, SH, ["#7cc35e", "#4c8a3a"]), 4);
  });
}

export function wallpaper(): HTMLCanvasElement {
  return sheet(141, (ctx) => {
    ctx.fillStyle = "#3d2a5c";
    ctx.fillRect(0, 0, SW, SH);
    for (let y = 40; y < SH; y += 120) {
      for (let x = (y / 120) % 2 === 0 ? 40 : 100; x < SW; x += 120) {
        ctx.save();
        ctx.translate(x, y);
        for (let i = 0; i < 4; i++) {
          ctx.rotate(Math.PI / 2);
          leaf(ctx, 0, 0, 40, 0, "rgba(246,201,69,0.35)");
        }
        ctx.restore();
      }
    }
    for (let x = 0; x < SW; x += 100) {
      ctx.fillStyle = (x / 100) % 2 === 0 ? "#efe2c8" : "#2a1420";
      ctx.fillRect(x, 900, 100, 62);
      ctx.fillStyle = (x / 100) % 2 === 0 ? "#2a1420" : "#efe2c8";
      ctx.fillRect(x, 962, 100, 62);
    }
  });
}
