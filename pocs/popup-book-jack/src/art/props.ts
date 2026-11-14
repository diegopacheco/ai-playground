import { blob, fillInk, ink, linear, paint, radial, seeded, withPaperBorder } from "./canvas";
import { drawHarp } from "./characters";

export function paintTree(seed: number, autumn = false): HTMLCanvasElement {
  const rand = seeded(seed);
  const art = paint(420, 620, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(180, 610);
    ctx.quadraticCurveTo(195, 450, 170, 300);
    ctx.lineTo(250, 300);
    ctx.quadraticCurveTo(225, 450, 245, 610);
    ctx.closePath();
    fillInk(ctx, linear(ctx, 300, 610, ["#9c6b43", "#6e4426"]));
    const greens = autumn ? ["#f2a541", "#e0712c", "#c94c2a"] : ["#8fd14f", "#4fae4a", "#2e8540"];
    const clumps: [number, number, number][] = [
      [210, 250, 150], [120, 300, 95], [300, 300, 100], [150, 180, 100], [270, 170, 105], [210, 110, 100],
    ];
    for (const [x, y, r] of clumps) {
      blob(ctx, x, y, r, r * 0.88);
      fillInk(ctx, radial(ctx, x, y, r, greens[0] ?? "#8fd14f", greens[2] ?? "#2e8540"), 7);
    }
    for (let i = 0; i < 9; i++) {
      blob(ctx, 90 + rand() * 240, 90 + rand() * 260, 14, 14);
      fillInk(ctx, autumn ? "#ffd23f" : "#e8413c", 4);
    }
  });
  return withPaperBorder(art, 14);
}

export function paintCloud(): HTMLCanvasElement {
  const art = paint(460, 220, (ctx) => {
    ctx.beginPath();
    for (const [x, y, r] of [[110, 140, 70], [200, 100, 90], [300, 115, 80], [370, 150, 55], [220, 160, 70]] as const) {
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, Math.PI * 2);
    }
    ctx.fillStyle = radial(ctx, 230, 120, 230, "#ffffff", "#dbe9f7");
    ctx.fill();
  });
  return withPaperBorder(art, 10);
}

export function paintSun(): HTMLCanvasElement {
  const art = paint(420, 420, (ctx) => {
    ctx.translate(210, 210);
    for (let i = 0; i < 14; i++) {
      ctx.rotate((Math.PI * 2) / 14);
      ctx.beginPath();
      ctx.moveTo(-26, -120);
      ctx.lineTo(0, -195);
      ctx.lineTo(26, -120);
      ctx.closePath();
      fillInk(ctx, "#ffb52e", 6);
    }
    blob(ctx, 0, 0, 125, 125);
    fillInk(ctx, radial(ctx, 0, 0, 130, "#fff4a8", "#ffc93c"), 7);
    for (const x of [-42, 42]) {
      ctx.beginPath();
      ctx.arc(x, -10, 16, Math.PI * 1.15, Math.PI * 1.85);
      ink(ctx, 8);
    }
    ctx.beginPath();
    ctx.arc(0, 20, 52, Math.PI * 0.2, Math.PI * 0.8);
    ink(ctx, 8);
    ctx.fillStyle = "rgba(255,120,90,0.45)";
    blob(ctx, -78, 30, 22, 14);
    ctx.fill();
    blob(ctx, 78, 30, 22, 14);
    ctx.fill();
  });
  return withPaperBorder(art, 12);
}

export function paintBush(seed: number): HTMLCanvasElement {
  const rand = seeded(seed);
  const art = paint(360, 200, (ctx) => {
    for (const [x, y, r] of [[90, 130, 70], [180, 105, 90], [270, 130, 72]] as const) {
      blob(ctx, x, y, r, r * 0.8);
      fillInk(ctx, radial(ctx, x, y, r, "#a5e06a", "#3f9a45"), 6);
    }
    for (let i = 0; i < 7; i++) {
      blob(ctx, 60 + rand() * 240, 70 + rand() * 90, 10, 10);
      fillInk(ctx, rand() > 0.5 ? "#ff6fa3" : "#ffffff", 3);
    }
  });
  return withPaperBorder(art, 10);
}

export function paintHills(sky: [string, string], mood: "day" | "dusk" | "night"): HTMLCanvasElement {
  const art = paint(1600, 760, (ctx, w, h) => {
    ctx.beginPath();
    ctx.roundRect(0, 0, w, h, 60);
    ctx.fillStyle = linear(ctx, 0, h, [sky[0], sky[1]]);
    ctx.fill();
    if (mood === "night") {
      const rand = seeded(7);
      ctx.fillStyle = "#fff7c2";
      for (let i = 0; i < 90; i++) {
        blob(ctx, rand() * w, rand() * h * 0.6, 2 + rand() * 3, 2 + rand() * 3);
        ctx.fill();
      }
    }
    const layers = mood === "day"
      ? [["#b9e3a0", "#8cc97a"], ["#86c56a", "#5aa356"], ["#5fae55", "#3f8a45"]]
      : mood === "dusk"
        ? [["#c8a7c9", "#9d7fa8"], ["#8f7aa5", "#6a5a86"], ["#5e5a85", "#433f66"]]
        : [["#3a4a7a", "#2a3560"], ["#2e3d68", "#1f2a4d"], ["#22305a", "#141c38"]];
    layers.forEach(([top, bottom], i) => {
      const base = 380 + i * 120;
      ctx.beginPath();
      ctx.moveTo(0, h);
      ctx.lineTo(0, base);
      for (let x = 0; x <= w; x += 40) {
        ctx.lineTo(x, base - Math.sin(x / (180 + i * 60) + i * 2) * (60 - i * 10) - Math.sin(x / 70 + i) * 8);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = linear(ctx, base - 80, h, [top ?? "#000", bottom ?? "#000"]);
      ctx.fill();
      ink(ctx, 6, "rgba(59,36,22,0.55)");
    });
  });
  return withPaperBorder(art, 14);
}

export function paintBubble(text: string[], tail: "left" | "right"): HTMLCanvasElement {
  const art = paint(620, 330, (ctx) => {
    ctx.beginPath();
    ctx.ellipse(310, 140, 290, 125, 0, 0, Math.PI * 2);
    ctx.moveTo(tail === "left" ? 170 : 450, 240);
    ctx.lineTo(tail === "left" ? 110 : 510, 320);
    ctx.lineTo(tail === "left" ? 240 : 380, 250);
    fillInk(ctx, "#ffffff", 7);
    ctx.fillStyle = "#3b2416";
    ctx.textAlign = "center";
    ctx.font = "700 46px Fredoka, sans-serif";
    text.forEach((line, i) => ctx.fillText(line, 310, 140 - ((text.length - 1) * 54) / 2 + i * 54 + 16));
  });
  return withPaperBorder(art, 10);
}

export function paintBanner(lines: string[], color: string): HTMLCanvasElement {
  const art = paint(1400, 520, (ctx) => {
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(700 + side * 520, 170);
      ctx.lineTo(700 + side * 680, 190);
      ctx.lineTo(700 + side * 620, 290);
      ctx.lineTo(700 + side * 680, 390);
      ctx.lineTo(700 + side * 520, 380);
      ctx.closePath();
      fillInk(ctx, "#a8323a", 8);
    }
    ctx.beginPath();
    ctx.moveTo(150, 90);
    ctx.quadraticCurveTo(700, 20, 1250, 90);
    ctx.lineTo(1250, 400);
    ctx.quadraticCurveTo(700, 330, 150, 400);
    ctx.closePath();
    fillInk(ctx, linear(ctx, 40, 400, [color, "#b8323f"]), 10);
    ctx.textAlign = "center";
    ctx.font = "120px Lobster, cursive";
    lines.forEach((line, i) => {
      const y = 215 + i * 125 - ((lines.length - 1) * 125) / 2 + 30;
      ctx.lineWidth = 16;
      ctx.strokeStyle = "#5e1f1f";
      ctx.strokeText(line, 700, y);
      ctx.fillStyle = "#fff4cf";
      ctx.fillText(line, 700, y);
    });
  });
  return withPaperBorder(art, 16);
}

export function paintButterfly(color: string): HTMLCanvasElement {
  const art = paint(200, 160, (ctx) => {
    for (const side of [-1, 1]) {
      blob(ctx, 100 + side * 48, 60, 46, 40);
      fillInk(ctx, radial(ctx, 100 + side * 48, 60, 50, "#ffffff", color), 5);
      blob(ctx, 100 + side * 36, 115, 30, 28);
      fillInk(ctx, color, 5);
    }
    blob(ctx, 100, 85, 10, 50);
    fillInk(ctx, "#3b2416", 3);
  });
  return withPaperBorder(art, 6);
}

export function paintHeart(): HTMLCanvasElement {
  return paint(128, 128, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(64, 112);
    ctx.bezierCurveTo(-10, 60, 20, 0, 64, 34);
    ctx.bezierCurveTo(108, 0, 138, 60, 64, 112);
    fillInk(ctx, "#ff4f7a", 6);
  });
}

export function paintPuff(): HTMLCanvasElement {
  return paint(128, 128, (ctx) => {
    const g = ctx.createRadialGradient(64, 64, 4, 64, 64, 62);
    g.addColorStop(0, "rgba(255,255,255,0.95)");
    g.addColorStop(0.6, "rgba(240,240,250,0.5)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 128, 128);
  });
}

export function paintMoon(): HTMLCanvasElement {
  const art = paint(300, 300, (ctx) => {
    ctx.beginPath();
    ctx.arc(150, 150, 130, Math.PI * 0.3, Math.PI * 1.7);
    ctx.bezierCurveTo(90, 60, 90, 240, 150 + Math.cos(Math.PI * 0.3) * 130, 150 + Math.sin(Math.PI * 0.3) * 130);
    fillInk(ctx, radial(ctx, 110, 150, 140, "#fffbe0", "#ffe27a"), 7);
    blob(ctx, 80, 120, 10, 10);
    ctx.fillStyle = "#3b2416";
    ctx.fill();
  });
  return withPaperBorder(art, 10);
}

export function paintCoin(): HTMLCanvasElement {
  return paint(96, 96, (ctx) => {
    blob(ctx, 48, 48, 40, 40);
    fillInk(ctx, radial(ctx, 48, 48, 44, "#fff3a8", "#e0a72e"), 6);
    blob(ctx, 48, 48, 24, 24);
    ink(ctx, 4, "#b07f22");
  });
}

export function paintCow(happy: boolean): HTMLCanvasElement {
  const art = paint(620, 440, (ctx) => {
    for (const x of [150, 220, 400, 470]) {
      ctx.beginPath();
      ctx.roundRect(x - 22, 250, 44, 150, 14);
      fillInk(ctx, "#fffaf0", 5);
      ctx.beginPath();
      ctx.roundRect(x - 24, 380, 48, 34, 8);
      fillInk(ctx, "#3b2416", 4);
    }
    ctx.beginPath();
    ctx.moveTo(90, 180);
    ctx.quadraticCurveTo(40, 230, 60, 300);
    ink(ctx, 8);
    blob(ctx, 300, 200, 230, 115);
    fillInk(ctx, "#fffaf0", 6);
    ctx.fillStyle = "#3b2416";
    for (const [x, y, rx, ry] of [[220, 160, 60, 40], [360, 240, 70, 45], [430, 150, 40, 30]] as const) {
      blob(ctx, x, y, rx, ry);
      ctx.fill();
    }
    blob(ctx, 300, 300, 50, 26);
    fillInk(ctx, "#ffb3c6", 5);
    blob(ctx, 520, 150, 80, 90);
    fillInk(ctx, "#fffaf0", 6);
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(520 + side * 40, 80);
      ctx.quadraticCurveTo(520 + side * 70, 40, 520 + side * 60, 20);
      ink(ctx, 14, "#d9c49a");
      blob(ctx, 520 + side * 85, 110, 30, 16);
      fillInk(ctx, "#fffaf0", 5);
    }
    blob(ctx, 525, 205, 62, 42);
    fillInk(ctx, "#ffb3c6", 5);
    for (const side of [-1, 1]) {
      blob(ctx, 525 + side * 22, 205, 8, 10);
      ctx.fillStyle = "#3b2416";
      ctx.fill();
      if (happy) {
        ctx.beginPath();
        ctx.arc(520 + side * 28, 140, 12, Math.PI * 1.1, Math.PI * 1.9);
        ink(ctx, 6);
      } else {
        blob(ctx, 520 + side * 28, 135, 10, 13);
        ctx.fillStyle = "#3b2416";
        ctx.fill();
      }
    }
    ctx.beginPath();
    ctx.moveTo(470, 240);
    ctx.lineTo(470, 280);
    ink(ctx, 6);
    blob(ctx, 470, 292, 16, 16);
    fillInk(ctx, "#ffd23f", 4);
  });
  return withPaperBorder(art, 12);
}

export function paintHarp(): HTMLCanvasElement {
  const art = paint(240, 320, (ctx) => drawHarp(ctx, 120, 190, 1));
  return withPaperBorder(art, 8);
}

export function paintNote(): HTMLCanvasElement {
  return paint(128, 128, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(80, 96);
    ctx.lineTo(80, 20);
    ctx.lineTo(110, 34);
    ink(ctx, 10, "#3b2416");
    blob(ctx, 62, 98, 22, 16);
    ctx.fillStyle = "#3b2416";
    ctx.fill();
  });
}

export function paintLeaf(): HTMLCanvasElement {
  return paint(128, 128, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(14, 64);
    ctx.quadraticCurveTo(64, 0, 116, 64);
    ctx.quadraticCurveTo(64, 128, 14, 64);
    fillInk(ctx, radial(ctx, 64, 64, 60, "#b6ff7a", "#3fa83a"), 5);
    ctx.beginPath();
    ctx.moveTo(18, 64);
    ctx.lineTo(112, 64);
    ink(ctx, 3, "#2f6b2a");
  });
}
