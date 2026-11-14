import { blob, type Ctx, fillInk, ink, linear, paint, radial, withPaperBorder } from "./canvas";

const SKIN = "#ffe1cc";
const HAIR = "#f2c55c";
const DRESS = "#f6c945";
const TRIM = "#3b6fd8";
const APRON = "#fffaf0";
const LINE = "#3b2416";

function face(ctx: Ctx, x: number, y: number, r: number, mood: "smile" | "wow" | "sleep" = "smile"): void {
  blob(ctx, x, y, r, r * 1.05);
  fillInk(ctx, radial(ctx, x, y, r, "#fff1e6", SKIN), 4);
  ctx.fillStyle = LINE;
  if (mood === "sleep") {
    for (const dx of [-0.36, 0.36]) {
      ctx.beginPath();
      ctx.arc(x + dx * r, y - 0.05 * r, r * 0.13, 0, Math.PI);
      ink(ctx, 3, LINE);
    }
  } else {
    for (const dx of [-0.36, 0.36]) {
      blob(ctx, x + dx * r, y - 0.08 * r, r * 0.09, r * 0.13);
      ctx.fill();
    }
  }
  ctx.fillStyle = "rgba(240,120,120,0.45)";
  blob(ctx, x - 0.55 * r, y + 0.25 * r, r * 0.16, r * 0.1);
  ctx.fill();
  blob(ctx, x + 0.55 * r, y + 0.25 * r, r * 0.16, r * 0.1);
  ctx.fill();
  ctx.beginPath();
  if (mood === "wow") {
    blob(ctx, x, y + 0.42 * r, r * 0.12, r * 0.16);
    ctx.fillStyle = "#8a2b2b";
    ctx.fill();
  } else {
    ctx.arc(x, y + 0.28 * r, r * 0.26, 0.15 * Math.PI, 0.85 * Math.PI);
    ink(ctx, 3.5, LINE);
  }
}

function aliceHead(ctx: Ctx, x: number, y: number, flowing: boolean, mood: "smile" | "wow"): void {
  ctx.beginPath();
  if (flowing) {
    ctx.moveTo(x - 62, y + 10);
    ctx.bezierCurveTo(x - 90, y - 60, x - 60, y - 150, x - 20, y - 130);
    ctx.bezierCurveTo(x, y - 170, x + 40, y - 150, x + 30, y - 120);
    ctx.bezierCurveTo(x + 80, y - 140, x + 95, y - 50, x + 62, y + 10);
  } else {
    ctx.moveTo(x - 60, y - 20);
    ctx.bezierCurveTo(x - 78, y + 60, x - 80, y + 110, x - 50, y + 130);
    ctx.lineTo(x + 50, y + 130);
    ctx.bezierCurveTo(x + 80, y + 110, x + 78, y + 60, x + 60, y - 20);
    ctx.bezierCurveTo(x + 60, y - 80, x - 60, y - 80, x - 60, y - 20);
  }
  ctx.closePath();
  fillInk(ctx, linear(ctx, y - 80, y + 130, ["#ffe08a", HAIR, "#d9a03a"]), 4);
  face(ctx, x, y, 48, mood);
  ctx.beginPath();
  ctx.arc(x, y - 6, 52, 1.15 * Math.PI, 1.85 * Math.PI);
  ink(ctx, 9, TRIM);
  ctx.beginPath();
  ctx.moveTo(x - 50, y - 18);
  ctx.quadraticCurveTo(x - 20, y - 64, x + 30, y - 52);
  ctx.quadraticCurveTo(x + 10, y - 40, x - 50, y - 18);
  ctx.fillStyle = HAIR;
  ctx.fill();
}

function puffs(ctx: Ctx, x: number, y: number): void {
  for (const dx of [-38, 38]) {
    blob(ctx, x + dx, y, 22, 18);
    fillInk(ctx, radial(ctx, x + dx, y, 22, "#ffe58a", DRESS), 3);
  }
}

export type AlicePose = "stand" | "sit" | "fall";

export function alice(pose: AlicePose, mood: "smile" | "wow" = "smile"): HTMLCanvasElement {
  const w = pose === "sit" ? 420 : 320;
  const h = pose === "sit" ? 420 : 580;
  const art = paint(w, h, (ctx) => {
    const cx = w / 2;
    const hy = pose === "fall" ? 170 : 110;
    if (pose === "fall") {
      for (const side of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(cx + side * 40, 230);
        ctx.quadraticCurveTo(cx + side * 95, 170, cx + side * 120, 60);
        ink(ctx, 20, LINE);
        ctx.beginPath();
        ctx.moveTo(cx + side * 40, 230);
        ctx.quadraticCurveTo(cx + side * 95, 170, cx + side * 120, 60);
        ink(ctx, 14, SKIN);
        blob(ctx, cx + side * 122, 54, 13, 13);
        fillInk(ctx, SKIN, 3);
      }
    }
    if (pose !== "sit") {
      ctx.fillStyle = "#fff";
      for (const dx of [-18, 18]) {
        ctx.beginPath();
        const top = pose === "fall" ? 300 : 430;
        ctx.roundRect(cx + dx - 11, top, 22, pose === "fall" ? 120 : 92, 8);
        fillInk(ctx, "#ffffff", 3);
        blob(ctx, cx + dx + (pose === "fall" ? dx * 0.3 : 4), top + (pose === "fall" ? 122 : 98), 17, 10);
        fillInk(ctx, "#1d1a22", 3);
      }
    }
    if (pose !== "fall") {
      for (const side of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(cx + side * 40, 190);
        ctx.quadraticCurveTo(cx + side * 70, 250, cx + side * (pose === "sit" ? 40 : 66), pose === "sit" ? 300 : 305);
        ink(ctx, 20, LINE);
        ctx.beginPath();
        ctx.moveTo(cx + side * 40, 190);
        ctx.quadraticCurveTo(cx + side * 70, 250, cx + side * (pose === "sit" ? 40 : 66), pose === "sit" ? 300 : 305);
        ink(ctx, 14, SKIN);
      }
    }
    const top = pose === "fall" ? 250 : 170;
    const hem = pose === "sit" ? 380 : pose === "fall" ? 330 : 440;
    const spread = pose === "sit" ? 190 : pose === "fall" ? 150 : 112;
    ctx.beginPath();
    ctx.moveTo(cx - 34, top);
    ctx.lineTo(cx + 34, top);
    ctx.lineTo(cx + 38, top + 70);
    if (pose === "fall") {
      ctx.bezierCurveTo(cx + spread, top + 60, cx + spread + 10, hem - 120, cx + spread - 10, hem - 150);
      for (let i = 0; i < 6; i++) {
        const a = cx + spread - 10 - ((i + 1) * (2 * spread - 20)) / 6;
        ctx.quadraticCurveTo(a + (spread - 10) / 6, hem - 110, a, hem - 150);
      }
      ctx.bezierCurveTo(cx - spread - 10, hem - 120, cx - spread, top + 60, cx - 38, top + 70);
    } else {
      ctx.bezierCurveTo(cx + 70, top + 110, cx + spread, hem - 60, cx + spread, hem);
      for (let i = 0; i < 8; i++) {
        const a = cx + spread - ((i + 1) * spread * 2) / 8;
        ctx.quadraticCurveTo(a + spread / 8, hem + 14, a, hem);
      }
      ctx.bezierCurveTo(cx - spread, hem - 60, cx - 70, top + 110, cx - 38, top + 70);
    }
    ctx.closePath();
    fillInk(ctx, linear(ctx, top, hem, ["#ffe27a", DRESS, "#e0a92a"]), 4);
    if (pose !== "fall") {
      ctx.beginPath();
      ctx.moveTo(cx - spread + 6, hem - 10);
      for (let i = 0; i < 8; i++) {
        const a = cx - spread + ((i + 1) * spread * 2) / 8;
        ctx.quadraticCurveTo(a - spread / 8, hem + 6, a, hem - 10);
      }
      ink(ctx, 8, TRIM);
      ctx.beginPath();
      ctx.moveTo(cx - 30, top + 20);
      ctx.lineTo(cx + 30, top + 20);
      ctx.bezierCurveTo(cx + 50, top + 120, cx + spread * 0.55, hem - 50, cx + spread * 0.5, hem - 18);
      ctx.quadraticCurveTo(cx, hem, cx - spread * 0.5, hem - 18);
      ctx.bezierCurveTo(cx - spread * 0.55, hem - 50, cx - 50, top + 120, cx - 30, top + 20);
      fillInk(ctx, APRON, 3);
      ctx.beginPath();
      ctx.moveTo(cx - spread * 0.5, hem - 22);
      for (let i = 0; i < 10; i++) ctx.quadraticCurveTo(cx - spread * 0.5 + (i + 0.5) * spread * 0.1, hem - 6, cx - spread * 0.5 + (i + 1) * spread * 0.1, hem - 22);
      ink(ctx, 3, "#c9b9a0");
    } else {
      ctx.beginPath();
      ctx.moveTo(cx - spread + 4, hem - 150);
      ctx.quadraticCurveTo(cx, hem - 120, cx + spread - 14, hem - 150);
      ink(ctx, 8, TRIM);
    }
    blob(ctx, cx, top + 70, 40, 9);
    fillInk(ctx, TRIM, 3);
    puffs(ctx, cx, top + 14);
    aliceHead(ctx, cx, hy, pose === "fall", pose === "fall" ? "wow" : mood);
  });
  return withPaperBorder(art, 5);
}

export function sister(): HTMLCanvasElement {
  return withPaperBorder(
    paint(380, 430, (ctx) => {
      ctx.beginPath();
      ctx.moveTo(130, 170);
      ctx.lineTo(250, 170);
      ctx.bezierCurveTo(300, 250, 360, 330, 370, 400);
      ctx.lineTo(10, 400);
      ctx.bezierCurveTo(20, 330, 80, 250, 130, 170);
      fillInk(ctx, linear(ctx, 170, 400, ["#c9a7e8", "#9b72c9"]), 4);
      blob(ctx, 190, 98, 74, 70);
      fillInk(ctx, "#6b3a1e", 4);
      blob(ctx, 190, 34, 34, 26);
      fillInk(ctx, "#6b3a1e", 4);
      face(ctx, 190, 112, 46, "smile");
      ctx.beginPath();
      ctx.moveTo(110, 250);
      ctx.lineTo(190, 270);
      ctx.lineTo(270, 250);
      ctx.lineTo(270, 310);
      ctx.lineTo(190, 330);
      ctx.lineTo(110, 310);
      ctx.closePath();
      fillInk(ctx, "#8a2b2b", 4);
      ctx.fillStyle = "#fffaf0";
      ctx.fillRect(122, 262, 60, 44);
      ctx.fillRect(198, 262, 60, 44);
      for (const x of [100, 280]) {
        blob(ctx, x, 300, 16, 14);
        fillInk(ctx, SKIN, 3);
      }
    }),
    5,
  );
}

export function rabbit(frame: number): HTMLCanvasElement {
  return withPaperBorder(
    paint(300, 420, (ctx) => {
      for (const [x, tilt] of [
        [128, -0.25],
        [168, 0.12],
      ] as const) {
        ctx.save();
        ctx.translate(x, 110);
        ctx.rotate(tilt);
        blob(ctx, 0, -60, 20, 66);
        fillInk(ctx, "#ffffff", 4);
        blob(ctx, 0, -56, 9, 50);
        ctx.fillStyle = "#f7b6c8";
        ctx.fill();
        ctx.restore();
      }
      const legs = frame % 2 === 0 ? [-40, 34] : [-8, 8];
      for (const dx of legs) {
        ctx.beginPath();
        ctx.moveTo(150, 320);
        ctx.lineTo(150 + dx, 395);
        ink(ctx, 22, LINE);
        ctx.beginPath();
        ctx.moveTo(150, 320);
        ctx.lineTo(150 + dx, 395);
        ink(ctx, 16, "#ffffff");
        blob(ctx, 150 + dx + 14, 400, 24, 10);
        fillInk(ctx, "#ffffff", 3);
      }
      blob(ctx, 150, 270, 60, 72);
      fillInk(ctx, "#ffffff", 4);
      ctx.beginPath();
      ctx.moveTo(98, 220);
      ctx.lineTo(202, 220);
      ctx.lineTo(196, 320);
      ctx.lineTo(104, 320);
      ctx.closePath();
      fillInk(ctx, "#c0392b", 4);
      ctx.strokeStyle = "rgba(255,220,120,0.6)";
      ctx.lineWidth = 3;
      for (let y = 232; y < 320; y += 16) {
        ctx.beginPath();
        ctx.moveTo(104, y);
        ctx.lineTo(196, y);
        ctx.stroke();
      }
      for (const y of [240, 270, 300]) {
        blob(ctx, 150, y, 5, 5);
        fillInk(ctx, "#f5d76e", 2);
      }
      ctx.beginPath();
      ctx.moveTo(84, 210);
      ctx.lineTo(216, 210);
      ctx.lineTo(236, 330);
      ctx.lineTo(196, 330);
      ctx.lineTo(196, 230);
      ctx.lineTo(104, 230);
      ctx.lineTo(104, 330);
      ctx.lineTo(64, 330);
      ctx.closePath();
      fillInk(ctx, "#2e4a8a", 4);
      blob(ctx, 150, 150, 52, 56);
      fillInk(ctx, radial(ctx, 150, 150, 56, "#ffffff", "#ece6e0"), 4);
      blob(ctx, 172, 140, 8, 10);
      ctx.fillStyle = "#e0457b";
      ctx.fill();
      blob(ctx, 196, 168, 7, 5);
      ctx.fillStyle = "#f08aa8";
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(206, 172);
      ctx.lineTo(236, 166);
      ctx.moveTo(206, 176);
      ctx.lineTo(236, 182);
      ink(ctx, 1.5, "#8a7a70");
      ctx.beginPath();
      ctx.moveTo(214, 250);
      ctx.quadraticCurveTo(240, 230, 252, 200);
      ink(ctx, 16, "#ffffff");
      blob(ctx, 258, 192, 24, 24);
      fillInk(ctx, radial(ctx, 258, 192, 24, "#fff5b0", "#d9a520"), 4);
      blob(ctx, 258, 192, 15, 15);
      fillInk(ctx, "#fffdf2", 2);
      ctx.beginPath();
      ctx.moveTo(258, 192);
      ctx.lineTo(258, 182);
      ctx.moveTo(258, 192);
      ctx.lineTo(265, 196);
      ink(ctx, 2, LINE);
    }),
    5,
  );
}

export function caterpillar(): HTMLCanvasElement {
  return withPaperBorder(
    paint(300, 420, (ctx) => {
      const segs: [number, number, number][] = [
        [80, 380, 40],
        [130, 370, 42],
        [175, 345, 42],
        [190, 300, 40],
        [180, 255, 38],
        [168, 212, 36],
        [160, 172, 34],
      ];
      for (const [x, y, r] of segs) {
        blob(ctx, x, y, r, r * 0.95);
        fillInk(ctx, radial(ctx, x, y, r, "#8fc7ff", "#2d6fc9"), 4);
        blob(ctx, x + r * 0.35, y + 4, r * 0.35, r * 0.5);
        ctx.fillStyle = "rgba(200,235,255,0.7)";
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(x + r * 0.7, y + r * 0.5);
        ctx.lineTo(x + r * 0.95, y + r * 0.8);
        ink(ctx, 5, "#1f4f99");
      }
      blob(ctx, 160, 110, 54, 50);
      fillInk(ctx, radial(ctx, 160, 110, 54, "#a8d6ff", "#3a7bd5"), 4);
      for (const dx of [-18, 18]) {
        ctx.beginPath();
        ctx.arc(160 + dx, 102, 9, 0, Math.PI);
        ink(ctx, 4, LINE);
      }
      ctx.beginPath();
      ctx.moveTo(140, 132);
      ctx.quadraticCurveTo(160, 144, 180, 132);
      ink(ctx, 4, LINE);
      for (const dx of [-24, 24]) {
        ctx.beginPath();
        ctx.moveTo(160 + dx * 0.6, 64);
        ctx.quadraticCurveTo(160 + dx * 1.2, 36, 160 + dx * 1.5, 30);
        ink(ctx, 5, "#1f4f99");
        blob(ctx, 160 + dx * 1.5, 28, 7, 7);
        fillInk(ctx, "#f5d76e", 2);
      }
    }),
    5,
  );
}

export function cheshireBody(): HTMLCanvasElement {
  return withPaperBorder(
    paint(520, 300, (ctx) => {
      ctx.beginPath();
      ctx.moveTo(470, 210);
      ctx.bezierCurveTo(530, 240, 500, 120, 450, 150);
      ink(ctx, 30, LINE);
      ctx.beginPath();
      ctx.moveTo(470, 210);
      ctx.bezierCurveTo(530, 240, 500, 120, 450, 150);
      ink(ctx, 24, "#e8923a");
      blob(ctx, 330, 210, 150, 70);
      fillInk(ctx, linear(ctx, 140, 280, ["#f5ae5a", "#d9782a"]), 4);
      ctx.strokeStyle = "#8a4515";
      ctx.lineWidth = 9;
      for (let x = 230; x < 470; x += 34) {
        ctx.beginPath();
        ctx.moveTo(x, 150);
        ctx.quadraticCurveTo(x + 10, 200, x - 4, 240);
        ctx.stroke();
      }
      blob(ctx, 150, 150, 110, 100);
      fillInk(ctx, radial(ctx, 150, 150, 110, "#ffc477", "#e2873a"), 4);
      for (const dx of [-72, 72]) {
        ctx.beginPath();
        ctx.moveTo(150 + dx * 0.9, 80);
        ctx.lineTo(150 + dx * 1.15, 18);
        ctx.lineTo(150 + dx * 0.35, 60);
        ctx.closePath();
        fillInk(ctx, "#e2873a", 4);
      }
      for (const dx of [-42, 42]) {
        blob(ctx, 150 + dx, 120, 22, 26);
        fillInk(ctx, "#c8f06a", 3);
        blob(ctx, 150 + dx, 120, 5, 20);
        ctx.fillStyle = LINE;
        ctx.fill();
      }
      ctx.strokeStyle = "#8a4515";
      ctx.lineWidth = 7;
      for (const y of [64, 84]) {
        ctx.beginPath();
        ctx.moveTo(126, y);
        ctx.lineTo(150, y + 8);
        ctx.lineTo(174, y);
        ctx.stroke();
      }
      for (const x of [250, 420]) {
        blob(ctx, x, 270, 30, 18);
        fillInk(ctx, "#f5ae5a", 4);
      }
    }),
    5,
  );
}

export function grin(): HTMLCanvasElement {
  return withPaperBorder(
    paint(300, 120, (ctx) => {
      ctx.beginPath();
      ctx.moveTo(20, 30);
      ctx.quadraticCurveTo(150, 150, 280, 30);
      ctx.quadraticCurveTo(150, 90, 20, 30);
      fillInk(ctx, "#ffffff", 6);
      ctx.strokeStyle = "#7a6a60";
      ctx.lineWidth = 2;
      for (let x = 40; x < 270; x += 18) {
        ctx.beginPath();
        ctx.moveTo(x, 38 + Math.sin(((x - 20) / 260) * Math.PI) * 24);
        ctx.lineTo(x, 50 + Math.sin(((x - 20) / 260) * Math.PI) * 52);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(20, 30);
      ctx.quadraticCurveTo(150, 150, 280, 30);
      ink(ctx, 6, LINE);
    }),
    5,
  );
}

export function hatter(): HTMLCanvasElement {
  return withPaperBorder(
    paint(320, 520, (ctx) => {
      ctx.beginPath();
      ctx.moveTo(80, 230);
      ctx.lineTo(240, 230);
      ctx.lineTo(270, 500);
      ctx.lineTo(50, 500);
      ctx.closePath();
      fillInk(ctx, linear(ctx, 230, 500, ["#3f8a5a", "#245c3a"]), 4);
      ctx.beginPath();
      ctx.moveTo(130, 230);
      ctx.lineTo(190, 230);
      ctx.lineTo(178, 420);
      ctx.lineTo(142, 420);
      ctx.closePath();
      fillInk(ctx, "#fff3c4", 3);
      ctx.beginPath();
      ctx.moveTo(120, 236);
      ctx.lineTo(160, 254);
      ctx.lineTo(120, 272);
      ctx.closePath();
      ctx.moveTo(200, 236);
      ctx.lineTo(160, 254);
      ctx.lineTo(200, 272);
      ctx.closePath();
      fillInk(ctx, "#d6336c", 3);
      blob(ctx, 160, 254, 9, 9);
      fillInk(ctx, "#b0204f", 2);
      blob(ctx, 160, 170, 56, 60);
      fillInk(ctx, radial(ctx, 160, 170, 60, "#fff1e6", SKIN), 4);
      for (const dx of [-20, 20]) {
        blob(ctx, 160 + dx, 160, 9, 11);
        fillInk(ctx, "#ffffff", 2);
        blob(ctx, 160 + dx + 2, 162, 4, 5);
        ctx.fillStyle = LINE;
        ctx.fill();
      }
      blob(ctx, 172, 186, 16, 10);
      fillInk(ctx, "#ffc4a8", 3);
      ctx.beginPath();
      ctx.arc(160, 200, 22, 0.1 * Math.PI, 0.9 * Math.PI);
      ink(ctx, 4, LINE);
      for (const dx of [-62, 62]) {
        blob(ctx, 160 + dx, 150, 18, 28);
        fillInk(ctx, "#c0703a", 3);
      }
      ctx.beginPath();
      ctx.moveTo(60, 116);
      ctx.lineTo(260, 116);
      ctx.lineTo(250, 100);
      ctx.lineTo(70, 100);
      ctx.closePath();
      fillInk(ctx, "#5a3a7a", 4);
      ctx.beginPath();
      ctx.moveTo(96, 102);
      ctx.lineTo(84, 10);
      ctx.quadraticCurveTo(160, -6, 236, 10);
      ctx.lineTo(224, 102);
      ctx.closePath();
      fillInk(ctx, linear(ctx, 0, 100, ["#8a62b0", "#5a3a7a"]), 4);
      ctx.fillStyle = "#d6336c";
      ctx.fillRect(92, 74, 136, 18);
      ctx.save();
      ctx.translate(196, 50);
      ctx.rotate(0.15);
      ctx.fillStyle = "#fffaf0";
      ctx.fillRect(-22, -16, 44, 32);
      ctx.strokeStyle = LINE;
      ctx.lineWidth = 2;
      ctx.strokeRect(-22, -16, 44, 32);
      ctx.fillStyle = LINE;
      ctx.font = "700 18px 'IM Fell English'";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("10/6", 0, 1);
      ctx.restore();
    }),
    5,
  );
}

export function hare(): HTMLCanvasElement {
  return withPaperBorder(
    paint(300, 480, (ctx) => {
      for (const [x, tilt] of [
        [120, -0.35],
        [180, 0.3],
      ] as const) {
        ctx.save();
        ctx.translate(x, 140);
        ctx.rotate(tilt);
        blob(ctx, 0, -70, 22, 76);
        fillInk(ctx, "#b5835a", 4);
        blob(ctx, 0, -66, 10, 58);
        ctx.fillStyle = "#f2c1a8";
        ctx.fill();
        ctx.restore();
      }
      ctx.strokeStyle = "#e8c45a";
      ctx.lineWidth = 4;
      for (let i = 0; i < 9; i++) {
        ctx.beginPath();
        ctx.moveTo(110 + i * 9, 110);
        ctx.lineTo(96 + i * 12, 82 - (i % 3) * 6);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(70, 250);
      ctx.lineTo(230, 250);
      ctx.lineTo(250, 470);
      ctx.lineTo(50, 470);
      ctx.closePath();
      fillInk(ctx, linear(ctx, 250, 470, ["#e07a5f", "#b5523a"]), 4);
      blob(ctx, 150, 180, 62, 66);
      fillInk(ctx, radial(ctx, 150, 180, 66, "#d9a77a", "#a8744a"), 4);
      for (const dx of [-24, 24]) {
        blob(ctx, 150 + dx, 168, 14, 16);
        fillInk(ctx, "#ffffff", 2);
        blob(ctx, 150 + dx, 170, 5, 7);
        ctx.fillStyle = "#d6336c";
        ctx.fill();
      }
      blob(ctx, 150, 204, 10, 7);
      fillInk(ctx, "#f08aa8", 2);
      ctx.beginPath();
      ctx.moveTo(150, 212);
      ctx.lineTo(150, 226);
      ink(ctx, 3, LINE);
      blob(ctx, 150, 262, 22, 10);
      fillInk(ctx, "#f5d76e", 3);
    }),
    5,
  );
}

export function dormouse(): HTMLCanvasElement {
  return withPaperBorder(
    paint(160, 150, (ctx) => {
      for (const dx of [-34, 34]) {
        blob(ctx, 80 + dx, 46, 22, 22);
        fillInk(ctx, "#a8744a", 3);
        blob(ctx, 80 + dx, 46, 12, 12);
        ctx.fillStyle = "#f2c1a8";
        ctx.fill();
      }
      blob(ctx, 80, 90, 56, 52);
      fillInk(ctx, radial(ctx, 80, 90, 56, "#d9a77a", "#9a6a40"), 4);
      face(ctx, 80, 92, 30, "sleep");
    }),
    4,
  );
}

export function queen(): HTMLCanvasElement {
  return withPaperBorder(
    paint(360, 560, (ctx) => {
      ctx.beginPath();
      ctx.moveTo(130, 200);
      ctx.lineTo(230, 200);
      ctx.bezierCurveTo(300, 330, 350, 450, 350, 540);
      ctx.lineTo(10, 540);
      ctx.bezierCurveTo(10, 450, 60, 330, 130, 200);
      fillInk(ctx, linear(ctx, 200, 540, ["#e03131", "#a51d1d"]), 4);
      ctx.beginPath();
      ctx.moveTo(150, 200);
      ctx.lineTo(210, 200);
      ctx.lineTo(250, 540);
      ctx.lineTo(110, 540);
      ctx.closePath();
      fillInk(ctx, "#fffaf0", 3);
      for (let y = 250; y < 520; y += 56) heart(ctx, 180, y, 14, "#e03131");
      for (const [x, y] of [
        [70, 470],
        [290, 470],
        [95, 380],
        [265, 380],
      ] as const)
        heart(ctx, x, y, 12, "#ffd43b");
      ctx.beginPath();
      ctx.moveTo(90, 196);
      ctx.quadraticCurveTo(180, 240, 270, 196);
      ctx.lineTo(260, 170);
      ctx.quadraticCurveTo(180, 200, 100, 170);
      ctx.closePath();
      fillInk(ctx, "#fffaf0", 3);
      blob(ctx, 180, 128, 62, 62);
      fillInk(ctx, radial(ctx, 180, 128, 62, "#fff1e6", SKIN), 4);
      for (const dx of [-24, 24]) {
        ctx.beginPath();
        ctx.moveTo(180 + dx - 14, 108 + (dx < 0 ? -6 : 6) * -1);
        ctx.lineTo(180 + dx + 14, 108 + (dx < 0 ? 6 : -6) * -1);
        ink(ctx, 6, LINE);
        blob(ctx, 180 + dx, 124, 6, 7);
        ctx.fillStyle = LINE;
        ctx.fill();
      }
      blob(ctx, 180, 160, 16, 11);
      ctx.fillStyle = "#8a2b2b";
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(126, 72);
      ctx.lineTo(132, 22);
      ctx.lineTo(154, 52);
      ctx.lineTo(180, 10);
      ctx.lineTo(206, 52);
      ctx.lineTo(228, 22);
      ctx.lineTo(234, 72);
      ctx.closePath();
      fillInk(ctx, linear(ctx, 10, 72, ["#ffe066", "#d9a520"]), 4);
      for (const x of [132, 180, 228]) {
        blob(ctx, x, x === 180 ? 12 : 24, 7, 7);
        fillInk(ctx, "#e03131", 2);
      }
      ctx.beginPath();
      ctx.moveTo(250, 250);
      ctx.quadraticCurveTo(310, 220, 350, 150);
      ink(ctx, 22, LINE);
      ctx.beginPath();
      ctx.moveTo(250, 250);
      ctx.quadraticCurveTo(310, 220, 350, 150);
      ink(ctx, 16, "#e03131");
      blob(ctx, 350, 142, 12, 12);
      fillInk(ctx, SKIN, 3);
    }),
    5,
  );
}

export function heart(ctx: Ctx, x: number, y: number, r: number, color: string): void {
  ctx.beginPath();
  ctx.moveTo(x, y + r);
  ctx.bezierCurveTo(x - r * 1.6, y - r * 0.2, x - r * 0.8, y - r * 1.4, x, y - r * 0.5);
  ctx.bezierCurveTo(x + r * 0.8, y - r * 1.4, x + r * 1.6, y - r * 0.2, x, y + r);
  ctx.fillStyle = color;
  ctx.fill();
}

export function spade(ctx: Ctx, x: number, y: number, r: number, color: string): void {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.bezierCurveTo(x - r * 1.6, y + r * 0.2, x - r * 0.8, y + r * 1.3, x, y + r * 0.5);
  ctx.bezierCurveTo(x + r * 0.8, y + r * 1.3, x + r * 1.6, y + r * 0.2, x, y - r);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.fillRect(x - r * 0.15, y + r * 0.3, r * 0.3, r * 0.9);
}

export function cardMan(number: number, frame: number): HTMLCanvasElement {
  return withPaperBorder(
    paint(240, 440, (ctx) => {
      const legs = frame % 2 === 0 ? [-26, 26] : [-14, 14];
      for (const dx of legs) {
        ctx.beginPath();
        ctx.moveTo(120 + dx * 0.4, 360);
        ctx.lineTo(120 + dx, 420);
        ink(ctx, 10, LINE);
        blob(ctx, 120 + dx + 6, 424, 16, 8);
        fillInk(ctx, LINE, 2);
      }
      ctx.beginPath();
      ctx.moveTo(40, 200);
      ctx.lineTo(10, 280);
      ctx.moveTo(200, 200);
      ctx.lineTo(234, frame % 2 === 0 ? 150 : 170);
      ink(ctx, 10, LINE);
      ctx.beginPath();
      ctx.roundRect(40, 130, 160, 240, 14);
      fillInk(ctx, "#fffdf6", 5);
      ctx.fillStyle = LINE;
      ctx.font = "700 34px 'IM Fell English'";
      ctx.textAlign = "center";
      ctx.fillText(String(number), 64, 172);
      const pips = number === 2 ? [[120, 200], [120, 300]] : number === 5 ? [[84, 180], [156, 180], [120, 250], [84, 320], [156, 320]] : [[84, 175], [156, 175], [120, 215], [84, 255], [156, 255], [84, 330], [156, 330]];
      for (const [x, y] of pips) spade(ctx, x ?? 0, y ?? 0, 15, LINE);
      blob(ctx, 120, 82, 46, 48);
      fillInk(ctx, radial(ctx, 120, 82, 48, "#fff1e6", SKIN), 4);
      for (const dx of [-16, 16]) {
        blob(ctx, 120 + dx, 78, 5, 6);
        ctx.fillStyle = LINE;
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(120, 100, 10, 0.15 * Math.PI, 0.85 * Math.PI);
      ink(ctx, 3, LINE);
      ctx.beginPath();
      ctx.moveTo(234, frame % 2 === 0 ? 150 : 170);
      ctx.lineTo(238, frame % 2 === 0 ? 112 : 132);
      ink(ctx, 6, "#8a5a2a");
      blob(ctx, 238, frame % 2 === 0 ? 104 : 124, 7, 12);
      fillInk(ctx, "#e03131", 2);
    }),
    4,
  );
}

export function flamingo(): HTMLCanvasElement {
  return withPaperBorder(
    paint(260, 440, (ctx) => {
      for (const dx of [-12, 14]) {
        ctx.beginPath();
        ctx.moveTo(120 + dx, 290);
        ctx.lineTo(120 + dx * 1.6, 430);
        ink(ctx, 7, "#e86a8a");
      }
      blob(ctx, 120, 260, 80, 46);
      fillInk(ctx, radial(ctx, 120, 250, 80, "#ffc2d1", "#f06292"), 4);
      ctx.beginPath();
      ctx.moveTo(180, 240);
      ctx.bezierCurveTo(240, 180, 150, 120, 180, 70);
      ink(ctx, 22, LINE);
      ctx.beginPath();
      ctx.moveTo(180, 240);
      ctx.bezierCurveTo(240, 180, 150, 120, 180, 70);
      ink(ctx, 16, "#f48fb1");
      blob(ctx, 192, 60, 26, 22);
      fillInk(ctx, "#f48fb1", 4);
      ctx.beginPath();
      ctx.moveTo(210, 56);
      ctx.quadraticCurveTo(246, 64, 236, 100);
      ctx.lineTo(214, 72);
      ctx.closePath();
      fillInk(ctx, LINE, 2);
      blob(ctx, 196, 52, 5, 5);
      ctx.fillStyle = LINE;
      ctx.fill();
    }),
    4,
  );
}

export function hedgehog(): HTMLCanvasElement {
  return withPaperBorder(
    paint(160, 120, (ctx) => {
      ctx.beginPath();
      for (let i = 0; i <= 16; i++) {
        const a = Math.PI + (i / 16) * Math.PI;
        const r = i % 2 === 0 ? 70 : 52;
        ctx.lineTo(80 + Math.cos(a) * r, 100 + Math.sin(a) * r * 1.1);
      }
      ctx.closePath();
      fillInk(ctx, "#6b4a2e", 3);
      blob(ctx, 130, 92, 22, 18);
      fillInk(ctx, "#d9b38c", 3);
      blob(ctx, 146, 90, 4, 4);
      ctx.fillStyle = LINE;
      ctx.fill();
    }),
    4,
  );
}

export function butterfly(frame: number, color: string): HTMLCanvasElement {
  return paint(80, 60, (ctx) => {
    const open = frame % 2 === 0 ? 1 : 0.4;
    for (const side of [-1, 1]) {
      blob(ctx, 40 + side * 18 * open, 22, 18 * open, 16);
      fillInk(ctx, color, 2);
      blob(ctx, 40 + side * 12 * open, 42, 12 * open, 12);
      fillInk(ctx, color, 2);
    }
    blob(ctx, 40, 32, 4, 18);
    fillInk(ctx, LINE, 1);
  });
}

export function bubble(text: string, width = 520): HTMLCanvasElement {
  return withPaperBorder(
    paint(width, 150, (ctx) => {
      ctx.beginPath();
      ctx.roundRect(10, 10, width - 20, 100, 40);
      ctx.moveTo(width * 0.3, 106);
      ctx.lineTo(width * 0.24, 145);
      ctx.lineTo(width * 0.4, 106);
      fillInk(ctx, "#fffdf6", 5);
      ctx.fillStyle = LINE;
      ctx.font = "italic 44px 'IM Fell English'";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, width / 2, 62, width - 60);
    }),
    4,
  );
}
