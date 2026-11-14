import { type Ctx, blob, fillInk, ink, linear, paint, radial, withPaperBorder } from "./canvas";

export type Mood = "smile" | "joy" | "scared" | "sad" | "angry" | "sleep";
export type JackProp = "beans" | "axe" | "hen" | "harp";
export type JackPose = "stand" | "wave" | "climb";

export interface JackLook {
  mood: Mood;
  pose?: JackPose;
  prop?: JackProp;
}

const SKIN = "#ffd9b8";
const SKIN_DARK = "#f2b48c";
const INK = "#3b2416";

function eyes(ctx: Ctx, xs: number[], y: number, mood: Mood, size = 1): void {
  for (const x of xs) {
    if (mood === "joy" || mood === "sleep") {
      ctx.beginPath();
      if (mood === "joy") ctx.arc(x, y + 6 * size, 18 * size, Math.PI * 1.1, Math.PI * 1.9);
      else ctx.arc(x, y - 6 * size, 18 * size, Math.PI * 0.1, Math.PI * 0.9);
      ink(ctx, 8 * size);
      continue;
    }
    const big = mood === "scared" ? 1.25 : 1;
    blob(ctx, x, y, 18 * big * size, 23 * big * size);
    fillInk(ctx, "#ffffff", 4 * size);
    blob(ctx, x + 3 * size, y + 4 * size, 11 * size, 13 * size);
    ctx.fillStyle = mood === "angry" ? "#7a1f1f" : "#3a2a6b";
    ctx.fill();
    blob(ctx, x + 7 * size, y - 2 * size, 4 * size, 5 * size);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }
  if (mood === "smile") return;
  for (const [i, x] of xs.entries()) {
    const side = i === 0 ? -1 : 1;
    const tilt = mood === "angry" ? -1 : 1;
    ctx.beginPath();
    ctx.moveTo(x - side * 16 * size, y - (36 + tilt * 6) * size);
    ctx.lineTo(x + side * 16 * size, y - (36 - tilt * 6) * size);
    ink(ctx, 8 * size);
  }
}

function mouth(ctx: Ctx, mood: Mood, x: number, y: number, size = 1): void {
  if (mood === "scared" || mood === "angry") {
    blob(ctx, x, y + 6 * size, (mood === "angry" ? 34 : 15) * size, (mood === "angry" ? 22 : 19) * size);
    fillInk(ctx, "#8a2f3a", 4 * size);
    if (mood === "angry") {
      ctx.fillStyle = "#fffbe0";
      ctx.fillRect(x - 24 * size, y - 12 * size, 48 * size, 10 * size);
    }
    return;
  }
  if (mood === "joy") {
    ctx.beginPath();
    ctx.moveTo(x - 36 * size, y - 4 * size);
    ctx.quadraticCurveTo(x, y + 48 * size, x + 36 * size, y - 4 * size);
    ctx.closePath();
    fillInk(ctx, "#8a2f3a", 4 * size);
    return;
  }
  ctx.beginPath();
  if (mood === "sad") {
    ctx.moveTo(x - 24 * size, y + 14 * size);
    ctx.quadraticCurveTo(x, y - 6 * size, x + 24 * size, y + 14 * size);
  } else if (mood === "sleep") {
    blob(ctx, x, y + 6 * size, 10 * size, 12 * size);
  } else {
    ctx.moveTo(x - 28 * size, y);
    ctx.quadraticCurveTo(x, y + 26 * size, x + 28 * size, y);
  }
  ink(ctx, 6 * size);
}

function limb(ctx: Ctx, color: string, x: number, y: number, angle: number, length = 120, width = 26): [number, number] {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  blob(ctx, 0, length / 2, width, length / 2 + 4);
  fillInk(ctx, color, 5);
  ctx.restore();
  return [x - Math.sin(angle) * length, y + Math.cos(angle) * length];
}

function hand(ctx: Ctx, x: number, y: number, r = 20, color = SKIN): void {
  blob(ctx, x, y, r, r);
  fillInk(ctx, color, 5);
}

function beans(ctx: Ctx, x: number, y: number): void {
  [[-22, -6], [0, -14], [22, -4], [-10, 10], [12, 10]].forEach(([dx, dy], i) => {
    blob(ctx, x + (dx ?? 0), y + (dy ?? 0) - 26, 11, 15);
    fillInk(ctx, radial(ctx, x + (dx ?? 0), y + (dy ?? 0) - 26, 16, "#d8ff9a", i % 2 ? "#6fbf3a" : "#b35ad8"), 3);
  });
}

function axe(ctx: Ctx, x: number, y: number): void {
  ctx.beginPath();
  ctx.moveTo(x - 10, y + 70);
  ctx.lineTo(x + 30, y - 170);
  ink(ctx, 18, INK);
  ctx.beginPath();
  ctx.moveTo(x - 10, y + 70);
  ctx.lineTo(x + 30, y - 170);
  ink(ctx, 10, "#a8743a");
  ctx.beginPath();
  ctx.moveTo(x + 22, y - 170);
  ctx.quadraticCurveTo(x + 110, y - 200, x + 120, y - 120);
  ctx.quadraticCurveTo(x + 70, y - 140, x + 34, y - 120);
  ctx.closePath();
  fillInk(ctx, linear(ctx, y - 200, y - 120, ["#eef2f8", "#9aa3b2"]), 5);
}

export function drawHen(ctx: Ctx, x: number, y: number, s: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(s, s);
  ctx.beginPath();
  ctx.moveTo(-90, 0);
  ctx.quadraticCurveTo(-140, -90, -100, -110);
  ctx.quadraticCurveTo(-80, -60, -60, -50);
  ctx.closePath();
  fillInk(ctx, "#e0a72e", 5);
  blob(ctx, 0, 0, 100, 70);
  fillInk(ctx, radial(ctx, 0, 0, 110, "#fff3b0", "#f5b82e"), 6);
  ctx.beginPath();
  ctx.moveTo(-40, -10);
  ctx.quadraticCurveTo(10, 40, 50, -5);
  ink(ctx, 5, "#c9962e");
  blob(ctx, 70, -70, 46, 46);
  fillInk(ctx, radial(ctx, 70, -70, 50, "#fff3b0", "#f5b82e"), 6);
  for (const dx of [-20, 0, 20]) {
    blob(ctx, 66 + dx, -120 - Math.abs(dx) * -0.4, 13, 16);
    fillInk(ctx, "#e8413c", 4);
  }
  ctx.beginPath();
  ctx.moveTo(110, -78);
  ctx.lineTo(148, -66);
  ctx.lineTo(110, -56);
  ctx.closePath();
  fillInk(ctx, "#ff9a3d", 4);
  blob(ctx, 104, -48, 9, 14);
  fillInk(ctx, "#e8413c", 3);
  blob(ctx, 82, -80, 8, 9);
  ctx.fillStyle = INK;
  ctx.fill();
  ctx.restore();
}

export function drawHarp(ctx: Ctx, x: number, y: number, s: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(s, s);
  ctx.beginPath();
  ctx.moveTo(-60, 90);
  ctx.quadraticCurveTo(-90, -40, -40, -120);
  ctx.quadraticCurveTo(40, -170, 70, -100);
  ctx.quadraticCurveTo(10, -70, 60, 90);
  ink(ctx, 26, INK);
  ctx.beginPath();
  ctx.moveTo(-60, 90);
  ctx.quadraticCurveTo(-90, -40, -40, -120);
  ctx.quadraticCurveTo(40, -170, 70, -100);
  ctx.quadraticCurveTo(10, -70, 60, 90);
  ink(ctx, 16, "#f5b82e");
  ctx.beginPath();
  ctx.moveTo(-70, 90);
  ctx.lineTo(70, 90);
  ink(ctx, 22, INK);
  ctx.beginPath();
  ctx.moveTo(-70, 90);
  ctx.lineTo(70, 90);
  ink(ctx, 12, "#c9962e");
  for (let i = 0; i < 6; i++) {
    const sx = -48 + i * 18;
    ctx.beginPath();
    ctx.moveTo(sx, 84);
    ctx.lineTo(sx + 6, -100 + Math.abs(i - 3) * 14);
    ink(ctx, 3, "#fff3cf");
  }
  blob(ctx, -40, -128, 26, 26);
  fillInk(ctx, radial(ctx, -40, -128, 28, "#fff0e0", SKIN), 4);
  ctx.beginPath();
  ctx.moveTo(-50, -126);
  ctx.quadraticCurveTo(-40, -118, -30, -126);
  ink(ctx, 3);
  ctx.beginPath();
  ctx.arc(-40, -138, 30, Math.PI, 0);
  ink(ctx, 8, "#e0a72e");
  ctx.restore();
}

function jackProp(ctx: Ctx, prop: JackProp, x: number, y: number): void {
  if (prop === "beans") beans(ctx, x, y);
  else if (prop === "axe") axe(ctx, x, y);
  else if (prop === "hen") drawHen(ctx, x + 20, y - 40, 0.55);
  else drawHarp(ctx, x + 10, y - 30, 0.55);
}

function jackLegs(ctx: Ctx, pose: JackPose): void {
  const spots: [number, number][] = pose === "climb" ? [[170, 0.35], [270, -0.1]] : [[180, 0], [260, 0]];
  for (const [x, a] of spots) {
    const [fx, fy] = limb(ctx, "#7a5a3a", x, 500, a, 130, 24);
    blob(ctx, fx + 10, fy + 8, 36, 20);
    fillInk(ctx, "#5a3519", 5);
  }
}

function jackBody(ctx: Ctx): void {
  ctx.beginPath();
  ctx.moveTo(140, 330);
  ctx.lineTo(300, 330);
  ctx.lineTo(320, 540);
  ctx.lineTo(120, 540);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 220, 400, 180, "#8fd16a", "#3f8f2f"));
  ctx.beginPath();
  ctx.roundRect(190, 440, 50, 50, 6);
  fillInk(ctx, "#d9a05a", 4);
  ctx.setLineDash([6, 6]);
  ink(ctx, 3, "#7a4a2a");
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.roundRect(126, 470, 188, 22, 6);
  fillInk(ctx, "#6e4220", 5);
  ctx.beginPath();
  ctx.moveTo(180, 330);
  ctx.lineTo(220, 380);
  ctx.lineTo(260, 330);
  ink(ctx, 6, "#2f6b2a");
}

function jackHead(ctx: Ctx, mood: Mood): void {
  for (const side of [-1, 1]) {
    blob(ctx, 220 + side * 112, 235, 20, 26);
    fillInk(ctx, SKIN, 5);
  }
  blob(ctx, 220, 225, 112, 118);
  fillInk(ctx, radial(ctx, 220, 225, 125, "#fff0e0", SKIN));
  ctx.fillStyle = "rgba(255,110,110,0.35)";
  blob(ctx, 150, 268, 24, 15);
  ctx.fill();
  blob(ctx, 290, 268, 24, 15);
  ctx.fill();
  ctx.fillStyle = "rgba(160,90,40,0.5)";
  for (const [x, y] of [[140, 255], [152, 266], [288, 255], [300, 266]] as const) {
    blob(ctx, x, y, 3, 3);
    ctx.fill();
  }
  eyes(ctx, [180, 260], 222, mood);
  blob(ctx, 220, 256, 12, 9);
  ctx.fillStyle = SKIN_DARK;
  ctx.fill();
  mouth(ctx, mood, 220, 288);
  ctx.beginPath();
  ctx.moveTo(105, 215);
  ctx.quadraticCurveTo(95, 90, 220, 88);
  ctx.quadraticCurveTo(350, 90, 335, 215);
  ctx.lineTo(315, 160);
  ctx.lineTo(290, 190);
  ctx.lineTo(265, 150);
  ctx.lineTo(235, 180);
  ctx.lineTo(205, 145);
  ctx.lineTo(175, 182);
  ctx.lineTo(150, 150);
  ctx.lineTo(128, 200);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 200, 110, 160, "#f0b060", "#b0622a"));
  ctx.beginPath();
  ctx.moveTo(110, 120);
  ctx.quadraticCurveTo(220, 40, 330, 120);
  ctx.lineTo(300, 70);
  ctx.quadraticCurveTo(220, 20, 140, 70);
  ctx.closePath();
  fillInk(ctx, "#3f8f2f", 5);
  ctx.beginPath();
  ctx.moveTo(300, 70);
  ctx.quadraticCurveTo(350, 40, 370, 60);
  ink(ctx, 6, "#e8413c");
  if (mood === "scared") {
    blob(ctx, 335, 170, 10, 17);
    fillInk(ctx, "#9fd8ff", 3);
  }
}

export function paintJack(look: JackLook): HTMLCanvasElement {
  const pose = look.pose ?? "stand";
  const art = paint(480, 690, (ctx) => {
    jackLegs(ctx, pose);
    jackBody(ctx);
    const left = pose === "climb" ? 2.8 : 0.45;
    const right = pose === "climb" ? -2.6 : pose === "wave" ? -2.6 : look.prop === "axe" ? -0.9 : look.prop ? -1.3 : -0.45;
    const [lx, ly] = limb(ctx, "#3f8f2f", 145, 350, left);
    hand(ctx, lx, ly);
    const [rx, ry] = limb(ctx, "#3f8f2f", 295, 350, right);
    if (look.prop === "axe") axe(ctx, rx, ry);
    hand(ctx, rx, ry);
    if (look.prop && look.prop !== "axe") jackProp(ctx, look.prop, rx, ry);
    jackHead(ctx, look.mood);
  });
  return withPaperBorder(art, 12);
}

export function paintMother(mood: Mood): HTMLCanvasElement {
  const art = paint(480, 780, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(170, 330);
    ctx.lineTo(310, 330);
    ctx.quadraticCurveTo(360, 520, 410, 760);
    ctx.quadraticCurveTo(240, 790, 70, 760);
    ctx.quadraticCurveTo(120, 520, 170, 330);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 240, 520, 300, "#8fb6e8", "#3f6aa8"));
    ctx.beginPath();
    ctx.moveTo(180, 430);
    ctx.lineTo(300, 430);
    ctx.lineTo(330, 740);
    ctx.quadraticCurveTo(240, 760, 150, 740);
    ctx.closePath();
    fillInk(ctx, "#fffaf0", 5);
    for (const [x, y] of [[230, 560], [270, 650], [205, 690]] as const) {
      ctx.beginPath();
      ctx.roundRect(x - 18, y - 16, 36, 32, 4);
      fillInk(ctx, "#e8a05a", 3);
    }
    const arms = mood === "angry" ? [2.5, -2.5] : [0.5, -0.5];
    arms.forEach((a, i) => {
      const [hx, hy] = limb(ctx, "#3f6aa8", i === 0 ? 175 : 305, 350, a ?? 0, 150, 28);
      hand(ctx, hx, hy, 22);
    });
    blob(ctx, 240, 230, 100, 108);
    fillInk(ctx, radial(ctx, 240, 230, 115, "#fff0e0", SKIN));
    eyes(ctx, [205, 275], 222, mood === "angry" ? "angry" : mood);
    blob(ctx, 240, 256, 12, 9);
    ctx.fillStyle = SKIN_DARK;
    ctx.fill();
    mouth(ctx, mood, 240, 292);
    blob(ctx, 240, 82, 52, 46);
    fillInk(ctx, radial(ctx, 240, 82, 52, "#c08a5a", "#6e3f1e"));
    ctx.beginPath();
    ctx.moveTo(140, 230);
    ctx.quadraticCurveTo(140, 110, 240, 110);
    ctx.quadraticCurveTo(340, 110, 340, 230);
    ctx.quadraticCurveTo(310, 160, 240, 160);
    ctx.quadraticCurveTo(170, 160, 140, 230);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 240, 130, 120, "#c08a5a", "#6e3f1e"));
    ctx.beginPath();
    ctx.moveTo(150, 140);
    ctx.quadraticCurveTo(240, 90, 330, 140);
    ink(ctx, 12, "#e8413c");
  });
  return withPaperBorder(art, 12);
}

export function paintOldMan(): HTMLCanvasElement {
  const art = paint(460, 780, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(380, 760);
    ctx.lineTo(370, 360);
    ink(ctx, 18, INK);
    ctx.beginPath();
    ctx.moveTo(380, 760);
    ctx.lineTo(370, 360);
    ink(ctx, 10, "#8a5a2a");
    ctx.beginPath();
    ctx.moveTo(160, 300);
    ctx.lineTo(300, 300);
    ctx.quadraticCurveTo(350, 520, 380, 760);
    ctx.quadraticCurveTo(230, 780, 80, 760);
    ctx.quadraticCurveTo(110, 520, 160, 300);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 230, 500, 300, "#c58adf", "#6a3a8a"));
    for (const [x, y] of [[170, 520], [270, 610], [200, 690], [300, 450]] as const) {
      ctx.save();
      ctx.translate(x, y);
      ctx.beginPath();
      for (let i = 0; i < 10; i++) {
        const r = i % 2 ? 9 : 22;
        const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
        ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
      }
      ctx.closePath();
      fillInk(ctx, "#ffd23f", 3);
      ctx.restore();
    }
    const [hx, hy] = limb(ctx, "#6a3a8a", 300, 330, -1.4, 130, 28);
    hand(ctx, hx, hy, 22);
    beans(ctx, hx, hy);
    const [lx, ly] = limb(ctx, "#6a3a8a", 160, 330, 0.3, 130, 28);
    hand(ctx, lx, ly, 22);
    blob(ctx, 230, 220, 92, 100);
    fillInk(ctx, radial(ctx, 230, 220, 105, "#fff0e0", SKIN));
    eyes(ctx, [195, 265], 205, "smile");
    blob(ctx, 230, 245, 20, 16);
    fillInk(ctx, "#f2a07a", 4);
    ctx.beginPath();
    ctx.moveTo(140, 240);
    ctx.quadraticCurveTo(150, 400, 230, 420);
    ctx.quadraticCurveTo(310, 400, 320, 240);
    ctx.quadraticCurveTo(290, 290, 230, 280);
    ctx.quadraticCurveTo(170, 290, 140, 240);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 230, 340, 120, "#ffffff", "#d8d4e0"));
    ctx.beginPath();
    ctx.moveTo(190, 300);
    ctx.quadraticCurveTo(230, 320, 270, 300);
    ink(ctx, 6);
    ctx.beginPath();
    ctx.moveTo(120, 170);
    ctx.quadraticCurveTo(200, 120, 230, 10);
    ctx.quadraticCurveTo(270, 100, 340, 170);
    ctx.quadraticCurveTo(230, 140, 120, 170);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 230, 80, 140, "#9a6ac0", "#4e2f6a"));
  });
  return withPaperBorder(art, 12);
}

export function paintGiantWife(): HTMLCanvasElement {
  const art = paint(560, 900, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(180, 360);
    ctx.lineTo(380, 360);
    ctx.quadraticCurveTo(470, 600, 520, 880);
    ctx.quadraticCurveTo(280, 910, 40, 880);
    ctx.quadraticCurveTo(90, 600, 180, 360);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 280, 600, 360, "#7fd0b0", "#2f8f6a"), 7);
    ctx.beginPath();
    ctx.moveTo(200, 470);
    ctx.lineTo(360, 470);
    ctx.lineTo(400, 860);
    ctx.quadraticCurveTo(280, 880, 160, 860);
    ctx.closePath();
    fillInk(ctx, "#fffaf0", 6);
    [[1.6, 180], [-0.3, 380]].forEach(([a, x]) => {
      const [hx, hy] = limb(ctx, "#2f8f6a", x ?? 0, 390, a ?? 0, 170, 36);
      blob(ctx, hx, hy, 30, 30);
      fillInk(ctx, SKIN, 6);
    });
    blob(ctx, 280, 250, 130, 136);
    fillInk(ctx, radial(ctx, 280, 250, 145, "#fff0e0", SKIN), 7);
    eyes(ctx, [235, 325], 240, "scared", 1.2);
    mouth(ctx, "smile", 280, 330, 1.2);
    ctx.fillStyle = "rgba(255,110,110,0.35)";
    blob(ctx, 195, 300, 30, 18);
    ctx.fill();
    blob(ctx, 365, 300, 30, 18);
    ctx.fill();
    for (const side of [-1, 1]) {
      blob(ctx, 280 + side * 140, 300, 50, 70);
      fillInk(ctx, radial(ctx, 280 + side * 140, 300, 70, "#ffe58a", "#e0a72e"), 6);
    }
    ctx.beginPath();
    ctx.moveTo(150, 230);
    ctx.quadraticCurveTo(160, 100, 280, 100);
    ctx.quadraticCurveTo(400, 100, 410, 230);
    ctx.quadraticCurveTo(360, 160, 280, 165);
    ctx.quadraticCurveTo(200, 160, 150, 230);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 280, 130, 140, "#ffe58a", "#e0a72e"), 6);
    blob(ctx, 280, 110, 120, 40);
    fillInk(ctx, "#fffaf0", 6);
  });
  return withPaperBorder(art, 14);
}
