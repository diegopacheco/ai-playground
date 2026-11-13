import { type Ctx, blob, fillInk, ink, linear, paint, radial, withPaperBorder } from "./canvas";

export type ChildName = "joao" | "maria";
export type ChildMood = "smile" | "joy" | "scared" | "sad";
export type ChildProp = "bread" | "broom" | "bone" | "basket";
export type WitchMood = "sweet" | "wicked" | "squint" | "falling";
export type FatherMood = "sad" | "joy";

export interface ChildLook {
  name: ChildName;
  mood: ChildMood;
  wave?: boolean;
  prop?: ChildProp;
}

const SKIN = "#ffd9b8";
const SKIN_DARK = "#f2b48c";
const INK = "#3b2416";

function eyes(ctx: Ctx, xs: number[], y: number, mood: ChildMood | "joy"): void {
  for (const x of xs) {
    if (mood === "joy") {
      ctx.beginPath();
      ctx.arc(x, y + 6, 18, Math.PI * 1.1, Math.PI * 1.9);
      ink(ctx, 8);
      continue;
    }
    const big = mood === "scared" ? 1.25 : 1;
    blob(ctx, x, y, 18 * big, 23 * big);
    fillInk(ctx, "#ffffff", 4);
    blob(ctx, x + 3, y + 4, 11, 13);
    ctx.fillStyle = "#3a2a6b";
    ctx.fill();
    blob(ctx, x + 7, y - 2, 4, 5);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }
  if (mood === "sad" || mood === "scared") {
    for (const [i, x] of xs.entries()) {
      const side = i === 0 ? -1 : 1;
      ctx.beginPath();
      ctx.moveTo(x - side * 14, y - 38);
      ctx.lineTo(x + side * 16, y - 30);
      ink(ctx, 6);
    }
  }
}

function childMouth(ctx: Ctx, mood: ChildMood, x: number, y: number): void {
  if (mood === "scared") {
    blob(ctx, x, y + 6, 15, 19);
    fillInk(ctx, "#8a2f3a", 4);
    return;
  }
  if (mood === "joy") {
    ctx.beginPath();
    ctx.moveTo(x - 36, y - 4);
    ctx.quadraticCurveTo(x, y + 48, x + 36, y - 4);
    ctx.closePath();
    fillInk(ctx, "#8a2f3a", 4);
    blob(ctx, x, y + 18, 14, 7);
    ctx.fillStyle = "#ff8a9a";
    ctx.fill();
    return;
  }
  ctx.beginPath();
  if (mood === "sad") {
    ctx.moveTo(x - 24, y + 14);
    ctx.quadraticCurveTo(x, y - 6, x + 24, y + 14);
  } else {
    ctx.moveTo(x - 28, y);
    ctx.quadraticCurveTo(x, y + 26, x + 28, y);
  }
  ink(ctx, 6);
}

function childLegs(ctx: Ctx, look: ChildLook): void {
  for (const x of [180, 260]) {
    ctx.beginPath();
    ctx.roundRect(x - 22, 520, 44, 120, 18);
    fillInk(ctx, look.name === "maria" ? "#fffaf0" : SKIN, 5);
    if (look.name === "joao") {
      ctx.beginPath();
      ctx.roundRect(x - 23, 590, 46, 40, 10);
      fillInk(ctx, "#ffffff", 4);
      ctx.strokeStyle = "#c94c4c";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(x - 20, 602);
      ctx.lineTo(x + 20, 602);
      ctx.stroke();
    }
    blob(ctx, x + 10, 648, 38, 20);
    fillInk(ctx, "#5a3519", 5);
  }
}

function joaoBody(ctx: Ctx): void {
  ctx.beginPath();
  ctx.roundRect(128, 330, 184, 180, 46);
  fillInk(ctx, radial(ctx, 220, 400, 160, "#7fb6ff", "#2f6fd0"));
  ctx.beginPath();
  ctx.roundRect(132, 460, 176, 90, 24);
  fillInk(ctx, linear(ctx, 460, 550, ["#9a6232", "#6e4220"]));
  for (const x of [170, 270]) {
    ctx.beginPath();
    ctx.moveTo(x, 336);
    ctx.lineTo(x + (x < 220 ? 10 : -10), 465);
    ink(ctx, 14, "#6e4220");
    blob(ctx, x + (x < 220 ? 9 : -9), 450, 8, 8);
    fillInk(ctx, "#ffd23f", 3);
  }
  ctx.beginPath();
  ctx.moveTo(220, 470);
  ctx.lineTo(220, 545);
  ink(ctx, 4, "#4a2a12");
}

function mariaBody(ctx: Ctx): void {
  ctx.beginPath();
  ctx.moveTo(165, 330);
  ctx.lineTo(275, 330);
  ctx.quadraticCurveTo(300, 420, 350, 560);
  ctx.quadraticCurveTo(220, 590, 90, 560);
  ctx.quadraticCurveTo(140, 420, 165, 330);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 220, 420, 220, "#ff7a86", "#c9283c"));
  ctx.beginPath();
  ctx.moveTo(180, 400);
  ctx.lineTo(260, 400);
  ctx.lineTo(290, 540);
  ctx.quadraticCurveTo(220, 556, 150, 540);
  ctx.closePath();
  fillInk(ctx, "#fffaf0", 5);
  ctx.fillStyle = "#ff9fb0";
  for (let y = 420; y < 540; y += 30) for (let x = 175; x < 270; x += 30) {
    blob(ctx, x + ((y / 30) % 2) * 15, y, 5, 5);
    ctx.fill();
  }
  ctx.beginPath();
  ctx.moveTo(170, 336);
  ctx.quadraticCurveTo(220, 380, 270, 336);
  ctx.lineTo(262, 330);
  ctx.lineTo(178, 330);
  ctx.closePath();
  fillInk(ctx, "#ffffff", 4);
}

function hand(ctx: Ctx, x: number, y: number): void {
  blob(ctx, x, y, 20, 20);
  fillInk(ctx, SKIN, 5);
}

function sleeve(ctx: Ctx, color: string, x: number, y: number, angle: number): [number, number] {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  blob(ctx, 0, 55, 26, 62);
  fillInk(ctx, color, 5);
  ctx.restore();
  return [x - Math.sin(angle) * 120, y + Math.cos(angle) * 120];
}

function propAt(ctx: Ctx, prop: ChildProp, x: number, y: number): void {
  if (prop === "bread") {
    blob(ctx, x + 10, y - 8, 52, 30);
    fillInk(ctx, radial(ctx, x + 10, y - 8, 55, "#f6c77a", "#b97a2e"), 5);
    for (const d of [-20, 0, 20]) {
      ctx.beginPath();
      ctx.moveTo(x + 10 + d - 8, y - 26);
      ctx.lineTo(x + 10 + d + 8, y + 8);
      ink(ctx, 4, "#8a5a20");
    }
  } else if (prop === "bone") {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 70, y - 40);
    ink(ctx, 26, INK);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 70, y - 40);
    ink(ctx, 16, "#fff6e0");
    for (const [bx, by] of [[x + 72, y - 50], [x + 82, y - 34]] as const) {
      blob(ctx, bx, by, 14, 14);
      fillInk(ctx, "#fff6e0", 5);
    }
  } else if (prop === "basket") {
    ctx.beginPath();
    ctx.arc(x, y + 10, 40, Math.PI, 0);
    ink(ctx, 7, "#8a5a20");
    ctx.beginPath();
    ctx.moveTo(x - 50, y + 12);
    ctx.lineTo(x + 50, y + 12);
    ctx.lineTo(x + 38, y + 62);
    ctx.lineTo(x - 38, y + 62);
    ctx.closePath();
    fillInk(ctx, linear(ctx, y + 12, y + 62, ["#d9a05a", "#9a6232"]), 5);
    blob(ctx, x - 16, y + 8, 14, 14);
    fillInk(ctx, "#e8413c", 4);
    blob(ctx, x + 14, y + 6, 14, 14);
    fillInk(ctx, "#8fd14f", 4);
  }
}

function broom(ctx: Ctx, x: number, y: number): void {
  ctx.beginPath();
  ctx.moveTo(x - 40, y - 150);
  ctx.lineTo(x + 30, y + 150);
  ink(ctx, 16, INK);
  ctx.beginPath();
  ctx.moveTo(x - 40, y - 150);
  ctx.lineTo(x + 30, y + 150);
  ink(ctx, 9, "#a8743a");
  ctx.beginPath();
  ctx.moveTo(x + 8, y + 140);
  ctx.lineTo(x + 70, y + 250);
  ctx.lineTo(x - 30, y + 260);
  ctx.closePath();
  fillInk(ctx, linear(ctx, y + 140, y + 260, ["#f3d27a", "#c9962e"]), 5);
}

function childArms(ctx: Ctx, look: ChildLook): void {
  const color = look.name === "joao" ? "#2f6fd0" : "#c9283c";
  const [lx, ly] = sleeve(ctx, color, 140, 350, 0.45);
  hand(ctx, lx, ly);
  const angle = look.wave ? 2.6 : look.prop === "bone" ? 1.9 : look.prop ? 1.2 : -0.45;
  const [rx, ry] = sleeve(ctx, color, 300, 350, angle);
  if (look.prop === "broom") broom(ctx, rx, ry);
  hand(ctx, rx, ry);
  if (look.prop && look.prop !== "broom") propAt(ctx, look.prop, rx + 8, ry);
}

function childHairBack(ctx: Ctx, name: ChildName): void {
  if (name !== "maria") return;
  for (const side of [-1, 1]) {
    for (let i = 0; i < 5; i++) {
      blob(ctx, 220 + side * (118 + i * 4), 250 + i * 34, 22 - i * 2, 22);
      fillInk(ctx, radial(ctx, 220 + side * 118, 250 + i * 34, 30, "#ffe58a", "#e0a72e"), 5);
    }
    blob(ctx, 220 + side * 138, 420, 18, 12);
    fillInk(ctx, "#2f9fd0", 4);
  }
}

function childHairFront(ctx: Ctx, name: ChildName): void {
  if (name === "maria") {
    ctx.beginPath();
    ctx.moveTo(105, 230);
    ctx.quadraticCurveTo(110, 100, 220, 95);
    ctx.quadraticCurveTo(330, 100, 335, 230);
    ctx.quadraticCurveTo(300, 150, 230, 150);
    ctx.quadraticCurveTo(180, 175, 105, 230);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 200, 120, 160, "#fff0a8", "#e0a72e"));
    ctx.beginPath();
    ctx.moveTo(275, 105);
    ctx.lineTo(320, 70);
    ctx.lineTo(330, 125);
    ctx.closePath();
    fillInk(ctx, "#ff5d8f", 5);
    ctx.beginPath();
    ctx.moveTo(275, 105);
    ctx.lineTo(240, 62);
    ctx.lineTo(258, 120);
    ctx.closePath();
    fillInk(ctx, "#ff5d8f", 5);
    return;
  }
  ctx.beginPath();
  ctx.moveTo(108, 220);
  ctx.quadraticCurveTo(100, 95, 220, 92);
  ctx.quadraticCurveTo(345, 95, 332, 220);
  ctx.lineTo(310, 175);
  ctx.lineTo(290, 200);
  ctx.lineTo(270, 160);
  ctx.lineTo(240, 190);
  ctx.lineTo(215, 155);
  ctx.lineTo(185, 192);
  ctx.lineTo(160, 160);
  ctx.lineTo(135, 205);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 200, 110, 160, "#b7764a", "#6e3f1e"));
  ctx.beginPath();
  ctx.moveTo(215, 95);
  ctx.quadraticCurveTo(235, 50, 265, 60);
  ink(ctx, 10, "#6e3f1e");
}

function childHead(ctx: Ctx, look: ChildLook): void {
  childHairBack(ctx, look.name);
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
  if (look.name === "joao") {
    ctx.fillStyle = "rgba(160,90,40,0.5)";
    for (const [x, y] of [[140, 255], [152, 266], [288, 255], [300, 266]] as const) {
      blob(ctx, x, y, 3, 3);
      ctx.fill();
    }
  }
  eyes(ctx, [180, 260], 222, look.mood);
  blob(ctx, 220, 256, 12, 9);
  ctx.fillStyle = SKIN_DARK;
  ctx.fill();
  childMouth(ctx, look.mood, 220, 288);
  childHairFront(ctx, look.name);
  if (look.mood === "scared") {
    blob(ctx, 335, 170, 10, 17);
    fillInk(ctx, "#9fd8ff", 3);
  }
}

export function paintChild(look: ChildLook): HTMLCanvasElement {
  const art = paint(440, 690, (ctx) => {
    childLegs(ctx, look);
    if (look.name === "joao") joaoBody(ctx);
    else mariaBody(ctx);
    childArms(ctx, look);
    childHead(ctx, look);
  });
  return withPaperBorder(art, 12);
}

const WITCH_SKIN = "#bfe0a0";
const WITCH_SKIN_DARK = "#86b56a";

function witchCane(ctx: Ctx): void {
  ctx.beginPath();
  ctx.moveTo(110, 760);
  ctx.lineTo(120, 430);
  ctx.quadraticCurveTo(122, 380, 165, 390);
  ink(ctx, 18, INK);
  ctx.beginPath();
  ctx.moveTo(110, 760);
  ctx.lineTo(120, 430);
  ctx.quadraticCurveTo(122, 380, 165, 390);
  ink(ctx, 10, "#7a4a2a");
}

function witchDress(ctx: Ctx): void {
  for (const x of [230, 340]) {
    ctx.beginPath();
    ctx.moveTo(x - 40, 745);
    ctx.quadraticCurveTo(x + 40, 730, x + 75, 755);
    ctx.lineTo(x + 50, 775);
    ctx.lineTo(x - 40, 775);
    ctx.closePath();
    fillInk(ctx, "#1d1622", 5);
  }
  ctx.beginPath();
  ctx.moveTo(230, 330);
  ctx.lineTo(350, 330);
  ctx.quadraticCurveTo(400, 520, 440, 750);
  ctx.quadraticCurveTo(285, 770, 140, 750);
  ctx.quadraticCurveTo(180, 520, 230, 330);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 290, 520, 300, "#9a6fd0", "#4e2f7a"));
  for (const [x, y, c] of [[200, 640, "#e0a72e"], [350, 560, "#2f9f7a"]] as const) {
    ctx.beginPath();
    ctx.roundRect(x - 26, y - 22, 52, 44, 6);
    fillInk(ctx, c, 4);
    ctx.setLineDash([6, 6]);
    ink(ctx, 3, "#fffaf0");
    ctx.setLineDash([]);
  }
  ctx.beginPath();
  ctx.moveTo(200, 340);
  ctx.quadraticCurveTo(290, 470, 380, 340);
  ctx.lineTo(395, 420);
  ctx.quadraticCurveTo(290, 520, 185, 420);
  ctx.closePath();
  fillInk(ctx, "#2f6b4a", 5);
}

function witchArm(ctx: Ctx, x: number, y: number, angle: number): [number, number] {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(-28, 0);
  ctx.lineTo(28, 0);
  ctx.lineTo(40, 130);
  ctx.lineTo(-40, 130);
  ctx.closePath();
  fillInk(ctx, "#6a45a0", 5);
  ctx.restore();
  const hx = x - Math.sin(angle) * 140;
  const hy = y + Math.cos(angle) * 140;
  blob(ctx, hx, hy, 22, 20);
  fillInk(ctx, WITCH_SKIN, 5);
  for (let i = -1; i <= 1; i++) {
    ctx.beginPath();
    ctx.moveTo(hx + i * 10, hy + 12);
    ctx.lineTo(hx + i * 14 - Math.sin(angle) * 26, hy + 12 + Math.cos(angle) * 26);
    ink(ctx, 10, INK);
    ctx.beginPath();
    ctx.moveTo(hx + i * 10, hy + 12);
    ctx.lineTo(hx + i * 14 - Math.sin(angle) * 26, hy + 12 + Math.cos(angle) * 26);
    ink(ctx, 5, WITCH_SKIN);
  }
  return [hx, hy];
}

function witchArms(ctx: Ctx, mood: WitchMood): void {
  if (mood === "falling") {
    witchArm(ctx, 220, 360, 2.7);
    witchArm(ctx, 360, 360, -2.7);
    return;
  }
  witchArm(ctx, 220, 360, 0.55);
  const reach = mood === "squint" ? -1.55 : mood === "sweet" ? -2.4 : -0.6;
  const [hx, hy] = witchArm(ctx, 360, 360, reach);
  if (mood === "sweet") {
    ctx.beginPath();
    ctx.moveTo(hx, hy);
    ctx.lineTo(hx + 8, hy + 110);
    ink(ctx, 8, "#fffaf0");
    blob(ctx, hx, hy - 30, 42, 42);
    fillInk(ctx, "#ff5d8f", 5);
    ctx.beginPath();
    for (let a = 0; a < Math.PI * 5; a += 0.2) ctx.lineTo(hx + Math.cos(a) * a * 2.5, hy - 30 + Math.sin(a) * a * 2.5);
    ink(ctx, 6, "#ffffff");
  }
}

function witchHair(ctx: Ctx): void {
  ctx.strokeStyle = "#8e8a98";
  for (let i = 0; i < 16; i++) {
    const side = i < 8 ? -1 : 1;
    const k = i % 8;
    ctx.beginPath();
    ctx.moveTo(290 + side * 70, 170);
    ctx.quadraticCurveTo(290 + side * (130 + k * 6), 230 + k * 10, 290 + side * (110 + k * 9), 300 + k * 12);
    ink(ctx, 12, "#5b5766");
    ctx.beginPath();
    ctx.moveTo(290 + side * 70, 170);
    ctx.quadraticCurveTo(290 + side * (130 + k * 6), 230 + k * 10, 290 + side * (110 + k * 9), 300 + k * 12);
    ink(ctx, 6, "#d8d4e0");
  }
}

function witchFace(ctx: Ctx, mood: WitchMood): void {
  blob(ctx, 290, 240, 98, 108);
  fillInk(ctx, radial(ctx, 290, 240, 115, "#e2f5c8", WITCH_SKIN));
  ctx.fillStyle = "rgba(220,120,140,0.35)";
  blob(ctx, 225, 280, 22, 14);
  ctx.fill();
  blob(ctx, 350, 280, 22, 14);
  ctx.fill();
  for (const x of [250, 330]) {
    if (mood === "falling") {
      for (const d of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(x - 16, 220 - d * 16);
        ctx.lineTo(x + 16, 220 + d * 16);
        ink(ctx, 8);
      }
      continue;
    }
    const lid = mood === "squint" ? 7 : mood === "wicked" ? 12 : 18;
    blob(ctx, x, 222, 16, lid);
    fillInk(ctx, "#fffbe0", 4);
    blob(ctx, x + 3, 224, 7, Math.min(lid, 9));
    ctx.fillStyle = mood === "wicked" ? "#c9283c" : "#1d1622";
    ctx.fill();
    ctx.beginPath();
    const tilt = mood === "wicked" ? 1 : mood === "sweet" ? -0.6 : 0.3;
    ctx.moveTo(x - 26, 192 + (x < 290 ? -tilt * 6 : tilt * 10));
    ctx.lineTo(x + 26, 192 + (x < 290 ? tilt * 10 : -tilt * 6));
    ink(ctx, 9, "#5b5766");
  }
  if (mood !== "falling") {
    for (const x of [250, 330]) {
      blob(ctx, x, 222, 34, 32);
      ink(ctx, 7, "#c9962e");
    }
    ctx.beginPath();
    ctx.moveTo(284, 220);
    ctx.lineTo(296, 220);
    ink(ctx, 6, "#c9962e");
  }
  ctx.beginPath();
  ctx.moveTo(282, 232);
  ctx.quadraticCurveTo(350, 250, 382, 312);
  ctx.quadraticCurveTo(340, 300, 300, 292);
  ctx.quadraticCurveTo(278, 280, 282, 232);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 320, 270, 70, "#d6efbc", WITCH_SKIN_DARK), 5);
  blob(ctx, 335, 272, 9, 9);
  fillInk(ctx, "#7a9a4a", 3);
  if (mood === "falling") {
    blob(ctx, 285, 325, 26, 30);
    fillInk(ctx, "#5e1f2c", 5);
    return;
  }
  ctx.beginPath();
  if (mood === "wicked") {
    ctx.moveTo(235, 312);
    ctx.quadraticCurveTo(290, 362, 345, 310);
    ctx.quadraticCurveTo(290, 340, 235, 312);
    ctx.closePath();
    fillInk(ctx, "#5e1f2c", 5);
    ctx.beginPath();
    ctx.moveTo(268, 324);
    ctx.lineTo(276, 344);
    ctx.lineTo(284, 326);
    ctx.closePath();
    fillInk(ctx, "#fffbe0", 3);
  } else if (mood === "squint") {
    ctx.moveTo(245, 322);
    ctx.quadraticCurveTo(290, 310, 330, 324);
    ink(ctx, 7);
  } else {
    ctx.moveTo(240, 310);
    ctx.quadraticCurveTo(290, 345, 340, 310);
    ink(ctx, 7);
  }
}

function witchHat(ctx: Ctx, mood: WitchMood): void {
  ctx.save();
  if (mood === "falling") {
    ctx.translate(290, 150);
    ctx.rotate(-0.5);
    ctx.translate(-290, -150);
  }
  blob(ctx, 290, 150, 175, 34);
  fillInk(ctx, "#2a2233");
  ctx.beginPath();
  ctx.moveTo(195, 150);
  ctx.quadraticCurveTo(250, 80, 260, 20);
  ctx.quadraticCurveTo(300, -10, 330, 10);
  ctx.quadraticCurveTo(310, 40, 380, 150);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 270, 60, 120, "#4a3a5c", "#1d1622"));
  ctx.beginPath();
  ctx.moveTo(205, 128);
  ctx.quadraticCurveTo(290, 108, 372, 128);
  ctx.lineTo(376, 150);
  ctx.quadraticCurveTo(290, 132, 200, 150);
  ctx.closePath();
  fillInk(ctx, "#9a6fd0", 4);
  ctx.beginPath();
  ctx.roundRect(272, 118, 36, 32, 4);
  ink(ctx, 6, "#ffd23f");
  ctx.restore();
}

export function paintWitch(mood: WitchMood): HTMLCanvasElement {
  const art = paint(560, 790, (ctx) => {
    if (mood !== "falling") witchCane(ctx);
    witchHair(ctx);
    witchDress(ctx);
    witchArms(ctx, mood);
    witchFace(ctx, mood);
    witchHat(ctx, mood);
  });
  return withPaperBorder(art, 12);
}

function plaid(ctx: Ctx, x: number, y: number, w: number, h: number): void {
  ctx.save();
  ctx.clip();
  ctx.fillStyle = "#c9283c";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "rgba(40,10,20,0.35)";
  for (let i = x; i < x + w; i += 44) ctx.fillRect(i, y, 16, h);
  for (let j = y; j < y + h; j += 44) ctx.fillRect(x, j, w, 16);
  ctx.restore();
}

export function paintFather(mood: FatherMood): HTMLCanvasElement {
  const art = paint(500, 800, (ctx) => {
    for (const x of [200, 300]) {
      ctx.beginPath();
      ctx.roundRect(x - 34, 520, 68, 210, 16);
      fillInk(ctx, "#4a5a7a");
      blob(ctx, x + 12, 735, 52, 26);
      fillInk(ctx, "#4a2a12");
    }
    ctx.beginPath();
    ctx.roundRect(140, 330, 220, 230, 50);
    plaid(ctx, 140, 330, 220, 230);
    ctx.beginPath();
    ctx.roundRect(140, 330, 220, 230, 50);
    ink(ctx, 6);
    ctx.beginPath();
    ctx.roundRect(140, 520, 220, 26, 6);
    fillInk(ctx, "#5a3519", 5);
    ctx.beginPath();
    ctx.roundRect(232, 518, 36, 30, 4);
    ink(ctx, 5, "#ffd23f");
    const arms: [number, number, number][] = mood === "joy" ? [[150, 360, 2.3], [350, 360, -2.3]] : [[150, 360, 0.25], [350, 360, -0.25]];
    for (const [x, y, a] of arms) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(a);
      ctx.beginPath();
      ctx.roundRect(-30, 0, 60, 150, 26);
      plaid(ctx, -30, 0, 60, 150);
      ctx.beginPath();
      ctx.roundRect(-30, 0, 60, 150, 26);
      ink(ctx, 5);
      blob(ctx, 0, 160, 24, 24);
      fillInk(ctx, SKIN, 5);
      ctx.restore();
    }
    blob(ctx, 250, 230, 100, 108);
    fillInk(ctx, radial(ctx, 250, 230, 115, "#fff0e0", SKIN));
    ctx.beginPath();
    ctx.moveTo(150, 230);
    ctx.quadraticCurveTo(160, 380, 250, 390);
    ctx.quadraticCurveTo(340, 380, 350, 230);
    ctx.quadraticCurveTo(320, 290, 250, 290);
    ctx.quadraticCurveTo(180, 290, 150, 230);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 250, 330, 110, "#a8643a", "#5a3519"));
    eyes(ctx, [212, 288], 215, mood === "joy" ? "joy" : "sad");
    blob(ctx, 250, 255, 18, 14);
    fillInk(ctx, "#f2a07a", 4);
    childMouth(ctx, mood === "joy" ? "joy" : "sad", 250, 305);
    ctx.beginPath();
    ctx.moveTo(150, 170);
    ctx.quadraticCurveTo(250, 30, 350, 170);
    ctx.closePath();
    fillInk(ctx, "#2f8f46");
    ctx.beginPath();
    ctx.roundRect(140, 150, 220, 36, 14);
    fillInk(ctx, "#1f6b44", 5);
    blob(ctx, 250, 70, 24, 24);
    fillInk(ctx, "#fffaf0", 5);
  });
  return withPaperBorder(art, 12);
}
