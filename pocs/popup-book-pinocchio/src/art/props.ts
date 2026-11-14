import { blob, fillInk, ink, linear, paint, radial, withPaperBorder } from "./canvas";

export function paintBanner(lines: string[], color: string): HTMLCanvasElement {
  return withPaperBorder(
    paint(1100, 150 + lines.length * 130, (ctx, w, h) => {
      ctx.beginPath();
      ctx.moveTo(60, 40);
      ctx.quadraticCurveTo(w / 2, 0, w - 60, 40);
      ctx.lineTo(w - 10, h / 2);
      ctx.lineTo(w - 60, h - 40);
      ctx.quadraticCurveTo(w / 2, h - 80, 60, h - 40);
      ctx.lineTo(10, h / 2);
      ctx.closePath();
      fillInk(ctx, linear(ctx, 0, h, [color, "#7a1422"]), 8);
      ctx.font = "120px Lobster";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      lines.forEach((line, i) => {
        const y = h / 2 + (i - (lines.length - 1) / 2) * 130;
        ctx.lineWidth = 14;
        ctx.strokeStyle = "#3b1a05";
        ctx.strokeText(line, w / 2, y);
        ctx.fillStyle = "#ffe9a8";
        ctx.fillText(line, w / 2, y);
      });
    }),
    12,
  );
}

export function paintBubble(text: string): HTMLCanvasElement {
  return withPaperBorder(
    paint(560, 300, (ctx, w) => {
      blob(ctx, w / 2, 130, 250, 110);
      ctx.moveTo(w / 2 - 40, 220);
      ctx.lineTo(w / 2 - 90, 296);
      ctx.lineTo(w / 2 + 20, 230);
      fillInk(ctx, "#ffffff", 7);
      ctx.font = "700 76px Fredoka";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#c42f2f";
      ctx.fillText(text, w / 2, 134);
    }),
    8,
  );
}

export function paintCoin(): HTMLCanvasElement {
  return paint(128, 128, (ctx, w, h) => {
    blob(ctx, w / 2, h / 2, 56, 56);
    fillInk(ctx, radial(ctx, w / 2, h / 2, 56, "#fff3a0", "#d99a1e"), 6);
    blob(ctx, w / 2, h / 2, 36, 36);
    ink(ctx, 4, "#a8741c");
  });
}

export function paintNote(): HTMLCanvasElement {
  return paint(128, 128, (ctx) => {
    blob(ctx, 44, 98, 24, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.fillRect(62, 20, 10, 80);
    ctx.fillRect(62, 20, 44, 14);
  });
}

export function paintStar(color: string): HTMLCanvasElement {
  return paint(128, 128, (ctx, w, h) => {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
      const r = i % 2 === 0 ? 58 : 24;
      ctx.lineTo(w / 2 + Math.cos(a) * r, h / 2 + Math.sin(a) * r);
    }
    ctx.closePath();
    fillInk(ctx, color, 5);
  });
}

export function paintHeart(): HTMLCanvasElement {
  return paint(128, 128, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(64, 112);
    ctx.bezierCurveTo(-10, 60, 20, 0, 64, 36);
    ctx.bezierCurveTo(108, 0, 138, 60, 64, 112);
    fillInk(ctx, "#ff4f7b", 5);
  });
}

export function paintRing(): HTMLCanvasElement {
  return paint(64, 64, (ctx) => {
    blob(ctx, 32, 32, 26, 26);
    ctx.fillStyle = "rgba(200,240,255,0.25)";
    ctx.fill();
    ink(ctx, 4, "rgba(230,250,255,0.9)");
    blob(ctx, 22, 22, 6, 6);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  });
}

export function paintChip(): HTMLCanvasElement {
  return paint(64, 64, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(8, 40);
    ctx.quadraticCurveTo(32, 0, 56, 24);
    ctx.quadraticCurveTo(36, 30, 8, 40);
    fillInk(ctx, "#f2c98a", 3);
  });
}

export function paintConfetti(): HTMLCanvasElement {
  return paint(32, 32, (ctx) => {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(4, 10, 24, 12);
  });
}

export function paintPuff(): HTMLCanvasElement {
  return paint(128, 128, (ctx, w, h) => {
    const g = ctx.createRadialGradient(w / 2, h / 2, 4, w / 2, h / 2, 60);
    g.addColorStop(0, "rgba(255,255,255,0.95)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  });
}

export function paintSack(): HTMLCanvasElement {
  return withPaperBorder(
    paint(220, 240, (ctx) => {
      blob(ctx, 110, 150, 90, 80);
      fillInk(ctx, "#c9a35a", 6);
      ctx.beginPath();
      ctx.moveTo(80, 80);
      ctx.lineTo(140, 80);
      ctx.lineTo(150, 30);
      ctx.lineTo(70, 30);
      ctx.closePath();
      fillInk(ctx, "#c9a35a", 6);
      ctx.font = "700 70px Fredoka";
      ctx.textAlign = "center";
      ctx.fillStyle = "#7a4d10";
      ctx.fillText("$", 110, 176);
    }),
    8,
  );
}

export function paintSign(text: string): HTMLCanvasElement {
  return withPaperBorder(
    paint(420, 520, (ctx) => {
      ctx.fillStyle = "#7a4d22";
      ctx.fillRect(190, 120, 40, 400);
      ctx.strokeRect(190, 120, 40, 400);
      ctx.beginPath();
      ctx.moveTo(20, 60);
      ctx.lineTo(340, 60);
      ctx.lineTo(400, 120);
      ctx.lineTo(340, 180);
      ctx.lineTo(20, 180);
      ctx.closePath();
      fillInk(ctx, "#e6c08a", 6);
      ctx.font = "700 52px Fredoka";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#5a3519";
      ctx.fillText(text, 200, 122);
    }),
    8,
  );
}

export function paintBush(color: string): HTMLCanvasElement {
  return withPaperBorder(
    paint(420, 220, (ctx) => {
      for (const [x, y, r] of [[90, 140, 80], [200, 100, 100], [320, 140, 80]] as const) {
        blob(ctx, x, y, r, r * 0.85);
        fillInk(ctx, color, 6);
      }
      for (let i = 0; i < 8; i++) {
        blob(ctx, 60 + i * 42, 120 + (i % 3) * 22, 10, 10);
        ctx.fillStyle = i % 2 ? "#ffd23f" : "#ff6fa3";
        ctx.fill();
      }
    }),
    8,
  );
}

export function paintCloud(color: string): HTMLCanvasElement {
  return withPaperBorder(
    paint(440, 220, (ctx) => {
      for (const [x, y, r] of [[110, 140, 70], [220, 100, 95], [330, 140, 70]] as const) {
        blob(ctx, x, y, r, r * 0.8);
        ctx.fillStyle = color;
        ctx.fill();
      }
      ctx.fillRect(110, 140, 220, 56);
    }),
    8,
  );
}

export function paintWave(color: string): HTMLCanvasElement {
  return withPaperBorder(
    paint(1600, 260, (ctx, w, h) => {
      ctx.beginPath();
      ctx.moveTo(0, h);
      ctx.lineTo(0, 120);
      for (let x = 0; x < w; x += 200) {
        ctx.quadraticCurveTo(x + 60, 0, x + 140, 70);
        ctx.quadraticCurveTo(x + 110, 120, x + 200, 120);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      fillInk(ctx, color, 7);
      for (let x = 60; x < w; x += 200) {
        blob(ctx, x, 60, 20, 12);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
      }
    }),
    8,
  );
}

export function paintBook(): HTMLCanvasElement {
  return withPaperBorder(
    paint(200, 160, (ctx) => {
      ctx.fillStyle = "#3a6fc4";
      ctx.fillRect(10, 10, 180, 140);
      ctx.strokeRect(10, 10, 180, 140);
      ctx.font = "700 60px Fredoka";
      ctx.textAlign = "center";
      ctx.fillStyle = "#ffffff";
      ctx.fillText("ABC", 100, 100);
    }),
    6,
  );
}
