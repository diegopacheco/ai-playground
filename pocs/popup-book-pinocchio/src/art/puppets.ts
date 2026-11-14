import { type Ctx, blob, fillInk, ink, linear, paint, radial, withPaperBorder } from "./canvas";

export type Mood = "smile" | "joy" | "sad" | "scared" | "sly" | "sleep";
export type Pose = "stand" | "dance" | "ears" | "swim" | "carry";

export interface Figure {
  canvas: HTMLCanvasElement;
  at: Record<string, [number, number]>;
}

const BORDER = 12;
const WOOD = "#e2a86a";
const WOOD_DARK = "#b9773d";
const SKIN = "#ffd9b8";
const INK = "#3b2416";

function framed(w: number, h: number, points: Record<string, [number, number]>, painter: (ctx: Ctx) => void): Figure {
  const canvas = withPaperBorder(paint(w, h, (ctx) => painter(ctx)), BORDER);
  const pad = BORDER + 6;
  const at: Record<string, [number, number]> = {};
  for (const [key, [x, y]] of Object.entries(points)) at[key] = [(x + pad) / canvas.width, (y + pad) / canvas.height];
  return { canvas, at };
}

function eye(ctx: Ctx, x: number, y: number, mood: Mood, size = 1): void {
  if (mood === "joy" || mood === "sleep") {
    ctx.beginPath();
    if (mood === "joy") ctx.arc(x, y + 5 * size, 13 * size, Math.PI * 1.1, Math.PI * 1.9);
    else ctx.arc(x, y - 5 * size, 13 * size, Math.PI * 0.1, Math.PI * 0.9);
    ink(ctx, 6 * size);
    return;
  }
  const big = mood === "scared" ? 1.25 : 1;
  blob(ctx, x, y, 13 * big * size, mood === "sly" ? 8 * size : 17 * big * size);
  fillInk(ctx, "#ffffff", 3 * size);
  blob(ctx, x + 3 * size, y + 2 * size, 7 * size, mood === "sly" ? 6 * size : 9 * size);
  ctx.fillStyle = "#2b2a5c";
  ctx.fill();
  blob(ctx, x + 5 * size, y - 2 * size, 3 * size, 3 * size);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  if (mood === "sad" || mood === "scared") {
    ctx.beginPath();
    ctx.moveTo(x - 14 * size, y - 22 * size);
    ctx.lineTo(x + 12 * size, y - 30 * size);
    ink(ctx, 5 * size);
  }
}

function mouth(ctx: Ctx, x: number, y: number, mood: Mood, size = 1): void {
  ctx.beginPath();
  if (mood === "scared") {
    blob(ctx, x, y, 11 * size, 14 * size);
    fillInk(ctx, "#8a2f3a", 3 * size);
    return;
  }
  if (mood === "sad") ctx.arc(x, y + 14 * size, 15 * size, Math.PI * 1.2, Math.PI * 1.8);
  else if (mood === "joy") {
    ctx.moveTo(x - 22 * size, y - 4 * size);
    ctx.quadraticCurveTo(x, y + 30 * size, x + 22 * size, y - 4 * size);
    ctx.closePath();
    fillInk(ctx, "#8a2f3a", 4 * size);
    return;
  } else ctx.arc(x, y - 8 * size, 18 * size, Math.PI * 0.2, Math.PI * 0.8);
  ink(ctx, 5 * size);
}

function limb(ctx: Ctx, x0: number, y0: number, x1: number, y1: number, width: number, color: string, joint: boolean): void {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.lineWidth = width + 8;
  ctx.strokeStyle = INK;
  ctx.stroke();
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.stroke();
  if (joint) {
    blob(ctx, (x0 + x1) / 2, (y0 + y1) / 2, width * 0.55, width * 0.55);
    fillInk(ctx, WOOD_DARK, 3);
  }
}

export interface PinocchioLook {
  mood: Mood;
  pose?: Pose;
  real?: boolean;
}

export function paintPinocchio(look: PinocchioLook): Figure {
  const pose = look.pose ?? "stand";
  const real = look.real ?? false;
  const skin = real ? SKIN : WOOD;
  const arms: Record<Pose, [number, number, number, number]> = {
    stand: [96, 470, 270, 470],
    dance: [70, 260, 300, 250],
    ears: [118, 210, 250, 205],
    swim: [290, 380, 310, 360],
    carry: [70, 300, 290, 300],
  };
  const [lx, ly, rx, ry] = arms[pose];
  const kick = pose === "dance" ? 1 : 0;
  return framed(360, 760, { nose: [232, 222], head: [180, 200], earL: [140, 140], earR: [214, 140], handL: [lx, ly], handR: [rx, ry] }, (ctx) => {
    limb(ctx, 150, 560, 140, 700, 30, skin, !real);
    limb(ctx, 210, 560, 220 + kick * 90, 700 - kick * 70, 30, skin, !real);
    blob(ctx, 130, 712, 36, 20);
    fillInk(ctx, "#3a6fc4", 5);
    blob(ctx, 236 + kick * 90, 712 - kick * 70, 36, 20);
    fillInk(ctx, "#3a6fc4", 5);
    ctx.beginPath();
    ctx.moveTo(118, 470);
    ctx.lineTo(242, 470);
    ctx.lineTo(250, 580);
    ctx.lineTo(185, 580);
    ctx.lineTo(180, 540);
    ctx.lineTo(175, 580);
    ctx.lineTo(110, 580);
    ctx.closePath();
    fillInk(ctx, "#d6453a", 6);
    limb(ctx, 130, 330, lx, ly, 26, skin, !real);
    limb(ctx, 230, 330, rx, ry, 26, skin, !real);
    ctx.beginPath();
    ctx.moveTo(122, 310);
    ctx.quadraticCurveTo(180, 290, 238, 310);
    ctx.lineTo(250, 480);
    ctx.lineTo(110, 480);
    ctx.closePath();
    fillInk(ctx, real ? "#3d8bd9" : "#4caf50", 6);
    for (const y of [350, 395, 440]) {
      blob(ctx, 180, y, 9, 9);
      fillInk(ctx, "#ffd23f", 3);
    }
    for (const [x, y] of [[lx, ly], [rx, ry]] as const) {
      blob(ctx, x, y, 20, 20);
      fillInk(ctx, real ? SKIN : "#ffffff", 5);
    }
    for (let i = 0; i < 7; i++) {
      blob(ctx, 128 + i * 17, 302, 16, 12);
      fillInk(ctx, "#ffffff", 3);
    }
    blob(ctx, 180, 200, 82, 90);
    fillInk(ctx, real ? radial(ctx, 180, 200, 90, "#ffe7cf", SKIN) : radial(ctx, 180, 200, 90, "#f6c590", WOOD), 6);
    if (real) {
      ctx.beginPath();
      ctx.moveTo(100, 190);
      ctx.quadraticCurveTo(110, 100, 200, 112);
      ctx.quadraticCurveTo(262, 122, 262, 180);
      ctx.quadraticCurveTo(200, 140, 100, 190);
      fillInk(ctx, "#6b3a1e", 5);
    } else {
      ctx.beginPath();
      ctx.moveTo(106, 150);
      ctx.lineTo(186, 0);
      ctx.lineTo(260, 150);
      ctx.quadraticCurveTo(180, 120, 106, 150);
      fillInk(ctx, linear(ctx, 0, 150, ["#ff6b5a", "#c42f2f"]), 6);
      blob(ctx, 186, 4, 14, 14);
      fillInk(ctx, "#ffd23f", 4);
    }
    eye(ctx, 158, 190, look.mood);
    eye(ctx, 208, 186, look.mood);
    blob(ctx, 150, 236, 16, 10);
    ctx.fillStyle = "rgba(255,90,90,0.45)";
    ctx.fill();
    mouth(ctx, 196, 262, look.mood);
  });
}

export type GeppettoProp = "none" | "mallet" | "candle";

export function paintGeppetto(mood: Mood, prop: GeppettoProp = "none"): Figure {
  return framed(420, 780, { head: [200, 190], hand: [340, 400] }, (ctx) => {
    limb(ctx, 160, 600, 150, 740, 40, "#5a4a3a", false);
    limb(ctx, 240, 600, 250, 740, 40, "#5a4a3a", false);
    for (const x of [140, 262]) {
      blob(ctx, x, 750, 46, 20);
      fillInk(ctx, "#3b2416", 5);
    }
    ctx.beginPath();
    ctx.moveTo(110, 320);
    ctx.quadraticCurveTo(200, 290, 290, 320);
    ctx.lineTo(310, 620);
    ctx.lineTo(90, 620);
    ctx.closePath();
    fillInk(ctx, "#7a5236", 6);
    ctx.beginPath();
    ctx.moveTo(140, 360);
    ctx.lineTo(260, 360);
    ctx.lineTo(276, 630);
    ctx.lineTo(124, 630);
    ctx.closePath();
    fillInk(ctx, "#efe3c8", 5);
    limb(ctx, 120, 340, 70, 470, 38, "#7a5236", false);
    limb(ctx, 280, 340, 340, 400, 38, "#7a5236", false);
    blob(ctx, 66, 480, 24, 24);
    fillInk(ctx, SKIN, 5);
    if (prop === "mallet") {
      limb(ctx, 340, 400, 360, 250, 12, "#8b5a2b", false);
      ctx.fillStyle = "#8b5a2b";
      ctx.fillRect(320, 210, 80, 50);
      ink(ctx, 5);
      ctx.strokeRect(320, 210, 80, 50);
    }
    if (prop === "candle") {
      ctx.fillStyle = "#fff6d8";
      ctx.fillRect(330, 300, 26, 90);
      ctx.strokeRect(330, 300, 26, 90);
      blob(ctx, 343, 286, 10, 18);
      fillInk(ctx, "#ffb52e", 3);
    }
    blob(ctx, 342, 400, 24, 24);
    fillInk(ctx, SKIN, 5);
    blob(ctx, 200, 190, 92, 100);
    fillInk(ctx, radial(ctx, 200, 190, 100, "#ffe7cf", SKIN), 6);
    ctx.beginPath();
    ctx.moveTo(104, 200);
    ctx.quadraticCurveTo(80, 70, 200, 80);
    ctx.quadraticCurveTo(320, 70, 296, 200);
    ctx.quadraticCurveTo(290, 130, 200, 130);
    ctx.quadraticCurveTo(110, 130, 104, 200);
    fillInk(ctx, "#f2c94c", 6);
    for (const x of [170, 232]) {
      blob(ctx, x, 186, 26, 24);
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.fill();
      ink(ctx, 5);
    }
    eye(ctx, 170, 188, mood, 0.8);
    eye(ctx, 232, 188, mood, 0.8);
    blob(ctx, 200, 225, 18, 16);
    fillInk(ctx, "#f2a58c", 4);
    ctx.beginPath();
    ctx.moveTo(130, 262);
    ctx.quadraticCurveTo(165, 230, 200, 248);
    ctx.quadraticCurveTo(235, 230, 270, 262);
    ctx.quadraticCurveTo(235, 268, 200, 256);
    ctx.quadraticCurveTo(165, 268, 130, 262);
    fillInk(ctx, "#ffffff", 4);
    mouth(ctx, 200, 292, mood, 0.7);
  });
}

export function paintCricket(): Figure {
  return framed(300, 360, { head: [190, 90] }, (ctx) => {
    for (const [x0, y0, x1, y1] of [[140, 250, 90, 340], [170, 260, 200, 345], [150, 220, 80, 270]] as const) limb(ctx, x0, y0, x1, y1, 10, "#4f9a2e", false);
    blob(ctx, 150, 220, 60, 80);
    fillInk(ctx, linear(ctx, 140, 300, ["#7ccf4a", "#3e8a24"]), 5);
    limb(ctx, 200, 170, 260, 150, 10, "#4f9a2e", false);
    limb(ctx, 262, 150, 262, 330, 6, "#6b3a1e", false);
    blob(ctx, 190, 110, 52, 46);
    fillInk(ctx, "#8ae05a", 5);
    for (const x of [172, 214]) {
      blob(ctx, x, 104, 16, 18);
      fillInk(ctx, "#ffffff", 3);
      blob(ctx, x + 3, 106, 7, 9);
      ctx.fillStyle = "#222";
      ctx.fill();
      blob(ctx, x, 104, 22, 22);
      ink(ctx, 4);
    }
    ctx.beginPath();
    ctx.moveTo(172, 50);
    ctx.quadraticCurveTo(150, 0, 110, 12);
    ctx.moveTo(206, 48);
    ctx.quadraticCurveTo(230, 0, 270, 10);
    ink(ctx, 5);
    mouth(ctx, 196, 140, "smile", 0.6);
  });
}

export function paintFox(): Figure {
  return framed(380, 760, { head: [190, 170] }, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(250, 560);
    ctx.quadraticCurveTo(380, 520, 350, 400);
    ctx.quadraticCurveTo(330, 500, 240, 520);
    fillInk(ctx, "#e8792b", 5);
    blob(ctx, 352, 404, 16, 22);
    fillInk(ctx, "#ffffff", 4);
    limb(ctx, 160, 560, 150, 720, 30, "#e8792b", false);
    limb(ctx, 220, 560, 230, 720, 30, "#e8792b", false);
    ctx.beginPath();
    ctx.moveTo(110, 300);
    ctx.lineTo(270, 300);
    ctx.lineTo(290, 590);
    ctx.lineTo(90, 590);
    ctx.closePath();
    fillInk(ctx, "#4a6b8a", 6);
    ctx.fillStyle = "#c9a35a";
    ctx.fillRect(130, 420, 40, 40);
    ctx.strokeRect(130, 420, 40, 40);
    limb(ctx, 260, 330, 300, 470, 24, "#4a6b8a", false);
    limb(ctx, 300, 470, 300, 740, 10, "#6b3a1e", false);
    ctx.beginPath();
    ctx.moveTo(110, 100);
    ctx.lineTo(130, 10);
    ctx.lineTo(170, 90);
    ctx.moveTo(210, 90);
    ctx.lineTo(250, 10);
    ctx.lineTo(270, 100);
    fillInk(ctx, "#e8792b", 5);
    ctx.beginPath();
    ctx.moveTo(90, 160);
    ctx.quadraticCurveTo(190, 50, 290, 160);
    ctx.quadraticCurveTo(260, 260, 190, 290);
    ctx.quadraticCurveTo(120, 260, 90, 160);
    fillInk(ctx, "#f08a3a", 6);
    ctx.beginPath();
    ctx.moveTo(130, 220);
    ctx.quadraticCurveTo(190, 300, 250, 220);
    ctx.quadraticCurveTo(190, 250, 130, 220);
    fillInk(ctx, "#fff3e0", 4);
    eye(ctx, 155, 165, "sly", 0.9);
    eye(ctx, 225, 165, "sly", 0.9);
    blob(ctx, 190, 230, 14, 10);
    fillInk(ctx, "#2b1a10", 3);
  });
}

export function paintCat(): Figure {
  return framed(340, 700, { head: [170, 170] }, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(80, 560);
    ctx.quadraticCurveTo(0, 520, 30, 400);
    ink(ctx, 26, "#6d6d78");
    limb(ctx, 140, 520, 130, 670, 28, "#8e8e9a", false);
    limb(ctx, 200, 520, 210, 670, 28, "#8e8e9a", false);
    blob(ctx, 170, 420, 90, 130);
    fillInk(ctx, "#8e8e9a", 6);
    blob(ctx, 170, 440, 50, 90);
    ctx.fillStyle = "#c9c9d2";
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(80, 120);
    ctx.lineTo(95, 30);
    ctx.lineTo(140, 90);
    ctx.moveTo(200, 90);
    ctx.lineTo(245, 30);
    ctx.lineTo(260, 120);
    fillInk(ctx, "#8e8e9a", 5);
    blob(ctx, 170, 170, 95, 85);
    fillInk(ctx, "#9e9eaa", 6);
    for (const x of [130, 210]) {
      blob(ctx, x, 160, 30, 26);
      fillInk(ctx, "#161616", 4);
    }
    ctx.beginPath();
    ctx.moveTo(160, 160);
    ctx.lineTo(180, 160);
    ink(ctx, 6);
    blob(ctx, 170, 210, 12, 9);
    fillInk(ctx, "#ff8aa0", 3);
    ctx.beginPath();
    for (const s of [-1, 1]) {
      ctx.moveTo(170 + s * 30, 215);
      ctx.lineTo(170 + s * 110, 200);
      ctx.moveTo(170 + s * 30, 225);
      ctx.lineTo(170 + s * 110, 235);
    }
    ink(ctx, 3);
    mouth(ctx, 170, 245, "smile", 0.6);
  });
}

export function paintFireEater(sneeze: boolean): Figure {
  return framed(520, 820, { head: [260, 200] }, (ctx) => {
    for (const x of [200, 320]) {
      limb(ctx, x, 640, x, 780, 52, "#2b2b3a", false);
      blob(ctx, x, 790, 56, 24);
      fillInk(ctx, "#1a1a1a", 5);
    }
    blob(ctx, 260, 470, 200, 220);
    fillInk(ctx, linear(ctx, 260, 690, ["#c42f2f", "#7a1414"]), 7);
    for (const s of [-1, 1]) {
      limb(ctx, 260 + s * 170, 380, 260 + s * 210, sneeze ? 260 : 540, 56, "#a32424", false);
      blob(ctx, 260 + s * 210, sneeze ? 250 : 550, 34, 34);
      fillInk(ctx, SKIN, 5);
    }
    blob(ctx, 260, 200, 120, 120);
    fillInk(ctx, radial(ctx, 260, 200, 120, "#ffd7c0", "#f0a888"), 7);
    ctx.beginPath();
    ctx.moveTo(150, 230);
    ctx.quadraticCurveTo(130, 520, 200, 640);
    ctx.quadraticCurveTo(260, 560, 320, 640);
    ctx.quadraticCurveTo(390, 520, 370, 230);
    ctx.quadraticCurveTo(260, 300, 150, 230);
    fillInk(ctx, linear(ctx, 230, 640, ["#2a2a2a", "#0e0e0e"]), 6);
    ctx.beginPath();
    ctx.moveTo(150, 120);
    ctx.quadraticCurveTo(260, 40, 370, 120);
    ctx.quadraticCurveTo(260, 90, 150, 120);
    fillInk(ctx, "#1a1a1a", 5);
    if (sneeze) {
      eye(ctx, 215, 180, "sleep");
      eye(ctx, 305, 180, "sleep");
      blob(ctx, 260, 300, 40, 34);
      fillInk(ctx, "#8a2f3a", 5);
    } else {
      eye(ctx, 215, 180, "smile");
      eye(ctx, 305, 180, "smile");
      for (const [x, d] of [[215, 1], [305, -1]] as const) {
        ctx.beginPath();
        ctx.moveTo(x - 30 * d, 140);
        ctx.lineTo(x + 25 * d, 155);
        ink(ctx, 12);
      }
    }
    blob(ctx, 260, 232, 30, 26);
    fillInk(ctx, "#e86a5a", 5);
  });
}

export function paintFairy(): Figure {
  return framed(380, 780, { head: [190, 170], wand: [330, 260] }, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(110, 120);
    ctx.quadraticCurveTo(40, 400, 80, 520);
    ctx.lineTo(300, 520);
    ctx.quadraticCurveTo(340, 400, 270, 120);
    fillInk(ctx, linear(ctx, 100, 520, ["#3fd0d9", "#1e8fb5"]), 6);
    ctx.beginPath();
    ctx.moveTo(130, 280);
    ctx.quadraticCurveTo(190, 260, 250, 280);
    ctx.lineTo(330, 760);
    ctx.lineTo(50, 760);
    ctx.closePath();
    fillInk(ctx, linear(ctx, 280, 760, ["#ffffff", "#bfe8ff", "#8fd0ff"]), 6);
    for (let i = 0; i < 14; i++) {
      blob(ctx, 70 + (i % 7) * 40, 560 + Math.floor(i / 7) * 110, 6, 6);
      ctx.fillStyle = "#ffe27a";
      ctx.fill();
    }
    limb(ctx, 250, 300, 330, 260, 22, SKIN, false);
    limb(ctx, 330, 260, 350, 120, 7, "#e9b949", false);
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
      const r = i % 2 === 0 ? 34 : 14;
      ctx.lineTo(352 + Math.cos(a) * r, 104 + Math.sin(a) * r);
    }
    ctx.closePath();
    fillInk(ctx, "#ffe27a", 4);
    blob(ctx, 190, 170, 72, 82);
    fillInk(ctx, radial(ctx, 190, 170, 80, "#fff0e0", SKIN), 6);
    ctx.beginPath();
    ctx.moveTo(116, 170);
    ctx.quadraticCurveTo(120, 80, 190, 82);
    ctx.quadraticCurveTo(260, 80, 264, 170);
    ctx.quadraticCurveTo(230, 120, 190, 130);
    ctx.quadraticCurveTo(150, 120, 116, 170);
    fillInk(ctx, "#3fd0d9", 5);
    eye(ctx, 165, 170, "smile", 0.8);
    eye(ctx, 215, 170, "smile", 0.8);
    mouth(ctx, 190, 215, "smile", 0.6);
  });
}

export function paintPuppet(kind: "harlequin" | "punch"): Figure {
  return framed(200, 420, { head: [100, 80] }, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(40, 150);
    ctx.lineTo(160, 150);
    ctx.lineTo(170, 400);
    ctx.lineTo(30, 400);
    ctx.closePath();
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    if (kind === "harlequin") {
      ctx.save();
      ctx.clip();
      const colors = ["#e8505b", "#3fa8e8", "#ffd23f", "#4caf50"];
      for (let y = 140; y < 420; y += 40) {
        for (let x = 20; x < 190; x += 40) {
          ctx.beginPath();
          ctx.moveTo(x, y - 20);
          ctx.lineTo(x + 20, y);
          ctx.lineTo(x, y + 20);
          ctx.lineTo(x - 20, y);
          ctx.closePath();
          ctx.fillStyle = colors[((x + y) / 40) % 4] ?? "#e8505b";
          ctx.fill();
        }
      }
      ctx.restore();
    }
    ctx.beginPath();
    ctx.moveTo(40, 150);
    ctx.lineTo(160, 150);
    ctx.lineTo(170, 400);
    ctx.lineTo(30, 400);
    ctx.closePath();
    ink(ctx, 5);
    limb(ctx, 45, 170, 10, 260, 16, kind === "punch" ? "#ffffff" : "#e8505b", false);
    limb(ctx, 155, 170, 190, 260, 16, kind === "punch" ? "#ffffff" : "#3fa8e8", false);
    blob(ctx, 100, 90, 50, 54);
    fillInk(ctx, kind === "punch" ? "#fff6e8" : SKIN, 5);
    if (kind === "punch") {
      ctx.beginPath();
      ctx.moveTo(56, 60);
      ctx.lineTo(100, -10);
      ctx.lineTo(144, 60);
      fillInk(ctx, "#ffffff", 5);
      blob(ctx, 100, 70, 52, 14);
      ctx.fillStyle = "#111";
      ctx.fill();
    } else {
      blob(ctx, 100, 70, 52, 30);
      fillInk(ctx, "#111", 4);
    }
    eye(ctx, 82, 92, "joy", 0.6);
    eye(ctx, 118, 92, "joy", 0.6);
    mouth(ctx, 100, 118, "joy", 0.5);
  });
}

export function paintWoodpecker(): HTMLCanvasElement {
  return paint(200, 160, (ctx) => {
    blob(ctx, 90, 95, 56, 40);
    fillInk(ctx, "#2b2b3a", 5);
    blob(ctx, 80, 92, 34, 22);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    blob(ctx, 140, 60, 30, 30);
    fillInk(ctx, "#e8343b", 5);
    ctx.beginPath();
    ctx.moveTo(165, 58);
    ctx.lineTo(198, 64);
    ctx.lineTo(165, 72);
    fillInk(ctx, "#ffb52e", 3);
    blob(ctx, 146, 56, 5, 5);
    ctx.fillStyle = "#111";
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(40, 100);
    ctx.lineTo(4, 120);
    ctx.lineTo(44, 116);
    fillInk(ctx, "#2b2b3a", 3);
  });
}

export function paintDonkeyEar(): HTMLCanvasElement {
  return paint(90, 260, (ctx, w, h) => {
    ctx.beginPath();
    ctx.moveTo(w / 2, 6);
    ctx.quadraticCurveTo(w - 4, h * 0.4, w * 0.75, h - 6);
    ctx.lineTo(w * 0.25, h - 6);
    ctx.quadraticCurveTo(4, h * 0.4, w / 2, 6);
    fillInk(ctx, "#8e8e9a", 6);
    blob(ctx, w / 2, h * 0.5, 16, 70);
    ctx.fillStyle = "#ffb3c1";
    ctx.fill();
  });
}
