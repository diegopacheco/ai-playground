import { type Ctx, blob, fillInk, ink, paint, radial, withPaperBorder } from "./canvas";

export type PigMood = "smile" | "scared" | "joy";
export type PigHat = "straw" | "cap" | "bow" | "kerchief";
export type WolfMood = "sly" | "puff" | "tired" | "ouch";

export interface PigLook {
  outfit: string;
  hat: PigHat;
  mood: PigMood;
  wave?: boolean;
  apron?: boolean;
}

const PINK_LIGHT = "#ffd3dc";
const PINK = "#f59fb3";
const PINK_DARK = "#e07a93";

function pigTail(ctx: Ctx): void {
  ctx.beginPath();
  ctx.moveTo(360, 500);
  ctx.bezierCurveTo(430, 470, 440, 420, 405, 420);
  ctx.bezierCurveTo(375, 420, 385, 460, 415, 455);
  ink(ctx, 12, "#3b2416");
  ctx.beginPath();
  ctx.moveTo(360, 500);
  ctx.bezierCurveTo(430, 470, 440, 420, 405, 420);
  ctx.bezierCurveTo(375, 420, 385, 460, 415, 455);
  ink(ctx, 7, PINK_DARK);
}

function pigLegs(ctx: Ctx): void {
  for (const x of [205, 315]) {
    ctx.beginPath();
    ctx.roundRect(x - 34, 560, 68, 100, 26);
    fillInk(ctx, PINK_DARK);
    ctx.beginPath();
    ctx.roundRect(x - 34, 630, 68, 30, 12);
    fillInk(ctx, "#6b3a2a");
  }
}

function pigBody(ctx: Ctx, look: PigLook): void {
  blob(ctx, 260, 490, 125, 118);
  fillInk(ctx, radial(ctx, 260, 490, 140, PINK_LIGHT, PINK));
  ctx.save();
  blob(ctx, 260, 490, 122, 115);
  ctx.clip();
  ctx.fillStyle = look.apron ? "#fbf6ea" : look.outfit;
  ctx.fillRect(100, 478, 320, 160);
  if (look.apron) {
    ctx.fillStyle = look.outfit;
    for (let y = 488; y < 640; y += 26) for (let x = 120; x < 420; x += 26) {
      blob(ctx, x + ((y / 26) % 2) * 13, y, 5, 5);
      ctx.fill();
    }
  }
  ctx.restore();
  ctx.beginPath();
  ctx.moveTo(140, 478);
  ctx.lineTo(380, 478);
  ink(ctx, 5);
  ctx.fillStyle = look.apron ? "#fbf6ea" : look.outfit;
  for (const x of [205, 315]) {
    ctx.beginPath();
    ctx.roundRect(x - 13, 395, 26, 90, 6);
    fillInk(ctx, look.apron ? "#fbf6ea" : look.outfit, 4);
    blob(ctx, x, 470, 9, 9);
    fillInk(ctx, "#ffd23f", 3);
  }
  ctx.beginPath();
  ctx.roundRect(225, 515, 70, 50, 10);
  fillInk(ctx, look.apron ? "#fbf6ea" : shade(look.outfit), 4);
}

function shade(color: string): string {
  const n = Number.parseInt(color.slice(1), 16);
  const r = Math.max(((n >> 16) & 255) - 35, 0);
  const g = Math.max(((n >> 8) & 255) - 35, 0);
  const b = Math.max((n & 255) - 35, 0);
  return `rgb(${r},${g},${b})`;
}

function pigArms(ctx: Ctx, look: PigLook): void {
  ctx.save();
  ctx.translate(140, 460);
  ctx.rotate(0.5);
  blob(ctx, 0, 30, 30, 58);
  fillInk(ctx, PINK);
  ctx.restore();
  ctx.save();
  if (look.wave) {
    ctx.translate(385, 410);
    ctx.rotate(-2.4);
  } else {
    ctx.translate(380, 460);
    ctx.rotate(-0.5);
  }
  blob(ctx, 0, 30, 30, 58);
  fillInk(ctx, PINK);
  ctx.restore();
}

function pigHead(ctx: Ctx, look: PigLook): void {
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(260 + side * 70, 160);
    ctx.lineTo(260 + side * 150, 95);
    ctx.lineTo(260 + side * 140, 215);
    ctx.closePath();
    fillInk(ctx, PINK);
    ctx.beginPath();
    ctx.moveTo(260 + side * 95, 165);
    ctx.lineTo(260 + side * 138, 125);
    ctx.lineTo(260 + side * 132, 195);
    ctx.closePath();
    ctx.fillStyle = PINK_DARK;
    ctx.fill();
  }
  blob(ctx, 260, 275, 158, 140);
  fillInk(ctx, radial(ctx, 260, 275, 170, "#ffe3e9", PINK));
  ctx.fillStyle = "rgba(255,90,120,0.35)";
  blob(ctx, 158, 325, 30, 20);
  ctx.fill();
  blob(ctx, 362, 325, 30, 20);
  ctx.fill();
  pigEyes(ctx, look.mood);
  blob(ctx, 260, 318, 60, 42);
  fillInk(ctx, radial(ctx, 260, 318, 62, "#ffc2cf", PINK_DARK), 5);
  for (const x of [238, 282]) {
    blob(ctx, x, 318, 9, 15);
    ctx.fillStyle = "#7a2f43";
    ctx.fill();
  }
  pigMouth(ctx, look.mood);
}

function pigEyes(ctx: Ctx, mood: PigMood): void {
  for (const x of [200, 320]) {
    if (mood === "joy") {
      ctx.beginPath();
      ctx.arc(x, 252, 22, Math.PI * 1.1, Math.PI * 1.9);
      ink(ctx, 9);
      continue;
    }
    const big = mood === "scared" ? 1.25 : 1;
    blob(ctx, x, 245, 25 * big, 30 * big);
    fillInk(ctx, "#ffffff", 5);
    blob(ctx, x + 4, 250, 14, 16);
    ctx.fillStyle = "#24140c";
    ctx.fill();
    blob(ctx, x + 9, 242, 5, 6);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }
  if (mood === "scared") {
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(260 + side * 30, 190);
      ctx.lineTo(260 + side * 90, 175);
      ink(ctx, 8);
    }
    blob(ctx, 395, 215, 12, 20);
    fillInk(ctx, "#9fd8ff", 3);
  }
}

function pigMouth(ctx: Ctx, mood: PigMood): void {
  if (mood === "scared") {
    blob(ctx, 260, 382, 20, 24);
    fillInk(ctx, "#7a2f43", 5);
    return;
  }
  if (mood === "joy") {
    ctx.beginPath();
    ctx.moveTo(215, 368);
    ctx.quadraticCurveTo(260, 430, 305, 368);
    ctx.closePath();
    fillInk(ctx, "#7a2f43", 5);
    blob(ctx, 260, 395, 18, 9);
    ctx.fillStyle = "#ff7d93";
    ctx.fill();
    return;
  }
  ctx.beginPath();
  ctx.moveTo(220, 372);
  ctx.quadraticCurveTo(260, 405, 300, 372);
  ink(ctx, 7);
}

function pigHat(ctx: Ctx, look: PigLook): void {
  if (look.hat === "straw") {
    blob(ctx, 260, 150, 170, 34);
    fillInk(ctx, radial(ctx, 260, 150, 170, "#ffe9a3", "#e5b84a"));
    ctx.beginPath();
    ctx.ellipse(260, 140, 90, 70, 0, Math.PI, Math.PI * 2);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 260, 110, 90, "#fff0b8", "#e9be55"));
    ctx.fillStyle = "#c0392b";
    ctx.fillRect(172, 122, 176, 16);
    ctx.strokeStyle = "rgba(150,100,20,0.5)";
    ctx.lineWidth = 3;
    for (let x = 110; x < 420; x += 14) {
      ctx.beginPath();
      ctx.moveTo(x, 138);
      ctx.lineTo(x + 8, 162);
      ctx.stroke();
    }
  } else if (look.hat === "cap") {
    ctx.beginPath();
    ctx.ellipse(260, 160, 120, 85, 0, Math.PI, Math.PI * 2);
    ctx.closePath();
    fillInk(ctx, radial(ctx, 250, 120, 120, "#7fd18b", "#2f8f46"));
    ctx.beginPath();
    ctx.ellipse(335, 160, 95, 24, -0.1, 0, Math.PI * 2);
    fillInk(ctx, "#2f8f46");
    blob(ctx, 260, 76, 14, 14);
    fillInk(ctx, "#ffd23f", 4);
  } else if (look.hat === "kerchief") {
    ctx.beginPath();
    ctx.moveTo(110, 200);
    ctx.quadraticCurveTo(260, 40, 410, 200);
    ctx.quadraticCurveTo(260, 150, 110, 200);
    fillInk(ctx, "#d6455d");
    ctx.fillStyle = "#ffffff";
    for (let i = 0; i < 9; i++) {
      blob(ctx, 160 + i * 25, 150 - Math.sin((i / 8) * Math.PI) * 45, 6, 6);
      ctx.fill();
    }
  }
}

function pigBow(ctx: Ctx, color: string): void {
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(260, 410);
    ctx.lineTo(260 + side * 55, 385);
    ctx.lineTo(260 + side * 55, 440);
    ctx.closePath();
    fillInk(ctx, color, 5);
  }
  blob(ctx, 260, 412, 16, 16);
  fillInk(ctx, color, 5);
}

export function paintPig(look: PigLook): HTMLCanvasElement {
  const art = paint(520, 680, (ctx) => {
    pigTail(ctx);
    pigLegs(ctx);
    pigBody(ctx, look);
    pigArms(ctx, look);
    pigHead(ctx, look);
    pigHat(ctx, look);
    if (look.hat === "bow") pigBow(ctx, "#2b6cd8");
  });
  return withPaperBorder(art, 12);
}

const FUR = "#8e8a98";
const FUR_LIGHT = "#c9c5d3";
const FUR_DARK = "#5b5766";

function wolfTail(ctx: Ctx): void {
  ctx.beginPath();
  ctx.moveTo(190, 560);
  ctx.bezierCurveTo(60, 560, 30, 420, 70, 360);
  ctx.bezierCurveTo(110, 420, 150, 470, 215, 500);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 110, 470, 140, FUR_LIGHT, FUR_DARK));
}

function wolfLegs(ctx: Ctx): void {
  for (const x of [225, 335]) {
    ctx.beginPath();
    ctx.roundRect(x - 40, 560, 80, 150, 18);
    fillInk(ctx, "#6b4c8a");
    ctx.beginPath();
    ctx.roundRect(x - 28, 600, 34, 30, 6);
    fillInk(ctx, "#8e6cb0", 3);
    ctx.beginPath();
    ctx.ellipse(x + 18, 722, 60, 26, 0, 0, Math.PI * 2);
    fillInk(ctx, FUR_DARK);
  }
}

function wolfBody(ctx: Ctx, mood: WolfMood): void {
  blob(ctx, 280, 470, 120, 150);
  fillInk(ctx, radial(ctx, 280, 470, 160, FUR_LIGHT, FUR));
  blob(ctx, 290, 490, 70, 105);
  ctx.fillStyle = "#ece6f1";
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(165, 560);
  ctx.quadraticCurveTo(280, 600, 395, 560);
  ctx.lineTo(390, 600);
  ctx.quadraticCurveTo(280, 640, 170, 600);
  ctx.closePath();
  fillInk(ctx, "#6b4c8a");
  for (const side of [-1, 1]) {
    ctx.save();
    const raised = mood === "ouch";
    ctx.translate(280 + side * 120, 410);
    ctx.rotate(side * (raised ? 2.5 : 0.35));
    blob(ctx, 0, 70, 32, 85);
    fillInk(ctx, radial(ctx, 0, 60, 90, FUR_LIGHT, FUR));
    for (let i = -1; i <= 1; i++) {
      ctx.beginPath();
      ctx.moveTo(i * 14, 150);
      ctx.lineTo(i * 16 + 4, 172);
      ink(ctx, 6, "#f5f0e0");
    }
    ctx.restore();
  }
}

function wolfHead(ctx: Ctx, mood: WolfMood): void {
  for (const x of [200, 340]) {
    ctx.beginPath();
    ctx.moveTo(x - 50, 175);
    ctx.lineTo(x + (x < 280 ? -25 : 25), 40);
    ctx.lineTo(x + 50, 160);
    ctx.closePath();
    fillInk(ctx, FUR);
    ctx.beginPath();
    ctx.moveTo(x - 25, 160);
    ctx.lineTo(x + (x < 280 ? -18 : 18), 80);
    ctx.lineTo(x + 25, 155);
    ctx.closePath();
    ctx.fillStyle = "#e9a0b4";
    ctx.fill();
  }
  const face = mood === "ouch" ? "#e2766a" : FUR;
  blob(ctx, 270, 225, 135, 110);
  fillInk(ctx, radial(ctx, 270, 225, 150, mood === "ouch" ? "#ffb3a0" : FUR_LIGHT, face));
  ctx.beginPath();
  ctx.moveTo(195, 290);
  ctx.lineTo(220, 255);
  ctx.lineTo(245, 300);
  ctx.lineTo(270, 260);
  ctx.lineTo(295, 300);
  ctx.lineTo(320, 255);
  ctx.lineTo(345, 290);
  ctx.quadraticCurveTo(270, 340, 195, 290);
  ctx.fillStyle = "#ece6f1";
  ctx.fill();
  if (mood === "puff") {
    for (const side of [-1, 1]) {
      blob(ctx, 270 + side * 95, 285, 70, 62);
      fillInk(ctx, radial(ctx, 270 + side * 95, 285, 75, "#ffd0d8", "#c9a3b8"));
    }
  }
  wolfSnout(ctx, mood);
  wolfEyes(ctx, mood);
}

function wolfSnout(ctx: Ctx, mood: WolfMood): void {
  ctx.beginPath();
  ctx.moveTo(220, 250);
  ctx.quadraticCurveTo(300, 225, 420, 268);
  ctx.quadraticCurveTo(445, 300, 410, 325);
  ctx.quadraticCurveTo(320, 350, 230, 320);
  ctx.closePath();
  fillInk(ctx, radial(ctx, 330, 285, 120, "#f1edf6", FUR_LIGHT));
  blob(ctx, 422, 280, 26, 20);
  fillInk(ctx, "#1d1622", 4);
  blob(ctx, 414, 272, 8, 5);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  if (mood === "puff") {
    blob(ctx, 405, 318, 22, 22);
    fillInk(ctx, "#7a2f43", 5);
    return;
  }
  if (mood === "tired") {
    ctx.beginPath();
    ctx.moveTo(300, 320);
    ctx.quadraticCurveTo(360, 345, 405, 320);
    ink(ctx, 6);
    ctx.beginPath();
    ctx.roundRect(355, 325, 34, 70, 16);
    fillInk(ctx, "#ff7d93", 4);
    return;
  }
  ctx.beginPath();
  ctx.moveTo(270, 318);
  ctx.quadraticCurveTo(350, 350, 410, 318);
  ctx.quadraticCurveTo(350, 375, 275, 330);
  ctx.closePath();
  fillInk(ctx, "#5e1f2c", 5);
  ctx.fillStyle = "#ffffff";
  for (let i = 0; i < 5; i++) {
    const x = 290 + i * 24;
    ctx.beginPath();
    ctx.moveTo(x, 324 + i * 2);
    ctx.lineTo(x + 10, 342 + i);
    ctx.lineTo(x + 20, 326 + i * 2);
    ctx.closePath();
    ctx.fill();
  }
}

function wolfEyes(ctx: Ctx, mood: WolfMood): void {
  for (const x of [225, 315]) {
    if (mood === "ouch") {
      for (const d of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(x - 18, 195 - d * 18);
        ctx.lineTo(x + 18, 195 + d * 18);
        ink(ctx, 9);
      }
      continue;
    }
    blob(ctx, x, 200, 28, mood === "tired" ? 14 : 24);
    fillInk(ctx, "#ffdd33", 5);
    blob(ctx, x + 6, 202, 6, mood === "tired" ? 8 : 16);
    ctx.fillStyle = "#1d1622";
    ctx.fill();
    ctx.beginPath();
    const tilt = mood === "sly" || mood === "puff" ? 1 : -0.4;
    ctx.moveTo(x - 36, 168 - (x < 270 ? -tilt * 6 : tilt * 12));
    ctx.lineTo(x + 34, 168 + (x < 270 ? tilt * 12 : -tilt * 6));
    ink(ctx, 11);
  }
  if (mood === "tired" || mood === "ouch") {
    for (const [x, y] of [[150, 150], [400, 170], [420, 120]] as const) {
      ctx.beginPath();
      ctx.moveTo(x, y - 22);
      ctx.quadraticCurveTo(x + 16, y + 6, x, y + 12);
      ctx.quadraticCurveTo(x - 16, y + 6, x, y - 22);
      fillInk(ctx, "#9fd8ff", 3);
    }
  }
}

export function paintWolf(mood: WolfMood): HTMLCanvasElement {
  const art = paint(560, 760, (ctx) => {
    wolfTail(ctx);
    wolfLegs(ctx);
    wolfBody(ctx, mood);
    wolfHead(ctx, mood);
  });
  return withPaperBorder(art, 12);
}
