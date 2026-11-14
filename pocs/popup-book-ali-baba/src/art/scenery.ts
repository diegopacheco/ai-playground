import {
  type Ctx,
  archHole,
  circle,
  crescent,
  diamond,
  dotsAlong,
  lettering,
  line,
  onionDome,
  oval,
  poly,
  punch,
  seeded,
  silhouette,
  star,
} from "./canvas";

export function palm(worldH: number, lean = 1, seed = 1): HTMLCanvasElement {
  const rand = seeded(seed);
  return silhouette(80, 120, worldH, (ctx) => {
    const top: [number, number] = [40 + lean * 10, 24];
    ctx.beginPath();
    ctx.moveTo(36, 120);
    ctx.quadraticCurveTo(38 + lean * 2, 70, top[0] - 2, top[1]);
    ctx.lineTo(top[0] + 2, top[1]);
    ctx.quadraticCurveTo(44 + lean * 3, 70, 44, 120);
    ctx.closePath();
    ctx.fill();
    for (let i = 0; i < 9; i++) {
      const a = -Math.PI + (i / 8) * Math.PI + (rand() - 0.5) * 0.25;
      const len = 26 + rand() * 12;
      const ex = top[0] + Math.cos(a) * len;
      const ey = top[1] + Math.sin(a) * len * 0.5 + len * 0.45;
      const mx = top[0] + Math.cos(a) * len * 0.55;
      const my = top[1] + Math.sin(a) * len * 0.55 - 4;
      ctx.beginPath();
      ctx.moveTo(top[0], top[1]);
      ctx.quadraticCurveTo(mx, my - 3, ex, ey);
      ctx.quadraticCurveTo(mx, my + 3.5, top[0], top[1] + 2);
      ctx.fill();
    }
    for (let i = 0; i < 3; i++) circle(ctx, top[0] - 2 + i * 2.5, top[1] + 4 + (i % 2) * 2, 2.4);
    punch(ctx, (c) => {
      for (let y = 34; y < 116; y += 6) {
        const t = (y - top[1]) / (120 - top[1]);
        const x = top[0] + (40 - top[0]) * t;
        line(c, 0.8, [x - 1.8, y, x, y + 1.5, x + 1.8, y]);
      }
    });
  }, seed);
}

export function bigTree(worldH: number): HTMLCanvasElement {
  const rand = seeded(42);
  return silhouette(120, 140, worldH, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(48, 140);
    ctx.quadraticCurveTo(54, 100, 50, 72);
    ctx.lineTo(70, 72);
    ctx.quadraticCurveTo(66, 100, 74, 140);
    ctx.closePath();
    ctx.fill();
    line(ctx, 6, [56, 84, 34, 62]);
    line(ctx, 5, [64, 80, 88, 58]);
    for (let i = 0; i < 46; i++) {
      const a = rand() * Math.PI * 2;
      const r = Math.sqrt(rand());
      circle(ctx, 60 + Math.cos(a) * r * 46, 50 + Math.sin(a) * r * 30, 9 + rand() * 7);
    }
    punch(ctx, (c) => {
      for (let i = 0; i < 70; i++) {
        const a = rand() * Math.PI * 2;
        const r = Math.sqrt(rand()) * 0.85;
        oval(c, 60 + Math.cos(a) * r * 44, 50 + Math.sin(a) * r * 28, 2.2, 1, rand() * 3);
      }
      oval(c, 82, 44, 9, 7);
    });
  }, 42);
}

export function city(worldH: number): HTMLCanvasElement {
  return silhouette(400, 120, worldH, (ctx) => {
    ctx.fillRect(0, 96, 400, 24);
    for (let x = 0; x < 400; x += 10) ctx.fillRect(x, 91, 6, 6);
    const domes: [number, number, number, number][] = [
      [60, 62, 34, 30],
      [150, 50, 48, 44],
      [250, 66, 30, 26],
      [330, 58, 40, 34],
    ];
    for (const [x, base, w, h] of domes) {
      ctx.fillRect(x - w * 0.55, base, w * 1.1, 96 - base);
      onionDome(ctx, x, base + 1, w, h);
      line(ctx, 1.2, [x, base - h, x, base - h - 6]);
      crescent(ctx, x, base - h - 8.5, 2.6, -1.6);
    }
    for (const [x, top] of [
      [105, 14],
      [200, 8],
      [292, 22],
      [375, 26],
    ] as const) {
      ctx.fillRect(x - 3.5, top + 10, 7, 96 - top);
      ctx.fillRect(x - 6, top + 24, 12, 3);
      onionDome(ctx, x, top + 11, 9, 9);
      line(ctx, 1, [x, top + 2, x, top - 3]);
    }
    for (const [x, w, h] of [
      [20, 30, 26],
      [118, 22, 20],
      [216, 34, 18],
      [268, 18, 30],
      [360, 26, 22],
    ] as const) {
      ctx.fillRect(x - w / 2, 96 - h, w, h);
    }
    punch(ctx, (c) => {
      for (const [x, base, w] of domes) {
        for (let i = -1; i <= 1; i++) archHole(c, x + i * w * 0.3, base + 8, w * 0.12, 9);
        star(c, x, base - 10, 2.6, 8, 0.5);
      }
      for (const x of [105, 200, 292, 375]) for (let y = 50; y < 90; y += 12) archHole(c, x, y, 2.6, 5);
      for (let x = 14; x < 400; x += 26) archHole(c, x, 103, 3, 7);
    });
  }, 51);
}

export function cliff(worldH: number): HTMLCanvasElement {
  const rand = seeded(8);
  return silhouette(200, 150, worldH, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(0, 150);
    ctx.lineTo(0, 70);
    let x = 0;
    while (x < 200) {
      x += 8 + rand() * 12;
      const y = 26 + Math.abs(Math.sin(x * 0.03)) * 30 + rand() * 18;
      ctx.lineTo(Math.min(x, 200), y);
    }
    ctx.lineTo(200, 150);
    ctx.closePath();
    ctx.fill();
    punch(ctx, (c) => {
      archHole(c, 100, 76, 40, 74);
      for (let i = 0; i < 9; i++) {
        let cx = 20 + rand() * 160;
        let cy = 50 + rand() * 80;
        if (Math.abs(cx - 100) < 30) cx += 60 * Math.sign(cx - 100 || 1);
        const pts: number[] = [cx, cy];
        for (let j = 0; j < 4; j++) {
          cx += (rand() - 0.5) * 14;
          cy += 4 + rand() * 6;
          pts.push(cx, cy);
        }
        line(c, 0.9, pts);
      }
    });
  }, 8);
}

export function ground(kind: "hills" | "dunes" | "street" | "floor", worldH = 0.5): HTMLCanvasElement {
  const rand = seeded(kind.length * 7);
  return silhouette(400, 24, worldH, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(0, 24);
    if (kind === "street" || kind === "floor") {
      ctx.lineTo(0, 10);
      ctx.lineTo(400, 10);
    } else {
      for (let x = 0; x <= 400; x += 4) {
        const y = kind === "dunes" ? 10 + Math.sin(x * 0.018) * 5 + Math.sin(x * 0.05) * 2 : 8 + Math.sin(x * 0.03) * 4 + rand() * 1.5;
        ctx.lineTo(x, y);
      }
    }
    ctx.lineTo(400, 24);
    ctx.closePath();
    ctx.fill();
    if (kind === "hills") for (let x = 6; x < 400; x += 14 + rand() * 20) poly(ctx, [x, 9 + Math.sin(x * 0.03) * 4, x + 2, 2 + rand() * 3, x + 4, 9 + Math.sin(x * 0.03) * 4]);
    punch(ctx, (c) => {
      if (kind === "street") for (let x = 8; x < 400; x += 16) diamond(c, x, 17, 3, 2);
      if (kind === "floor") for (let x = 10; x < 400; x += 20) star(c, x, 17, 3.2, 8, 0.5);
    });
  });
}

export function skyBanner(seed = 2, moon = true): HTMLCanvasElement {
  const rand = seeded(seed);
  return silhouette(400, 54, 1.25, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(400, 0);
    ctx.lineTo(400, 34);
    for (let x = 400; x >= 0; x -= 25) ctx.quadraticCurveTo(x - 12.5, 54 - (x % 50 === 0 ? 0 : 6), x - 25, 36);
    ctx.closePath();
    ctx.fill();
    punch(ctx, (c) => {
      if (moon) crescent(c, 300, 20, 11, -2.6);
      for (let i = 0; i < 46; i++) {
        const x = 6 + rand() * 388;
        const y = 4 + rand() * 30;
        if (moon && Math.hypot(x - 300, y - 20) < 16) continue;
        const r = 0.6 + rand() * 1.8;
        if (r > 1.6) star(c, x, y, r * 1.4, 4, 0.35);
        else circle(c, x, y, r * 0.6);
      }
    });
  }, seed);
}

export function stalactites(): HTMLCanvasElement {
  const rand = seeded(19);
  return silhouette(400, 60, 1.4, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(400, 0);
    ctx.lineTo(400, 20);
    let x = 400;
    while (x > 0) {
      const w = 8 + rand() * 18;
      ctx.lineTo(x - w / 2, 18 + rand() * 40);
      x -= w;
      ctx.lineTo(Math.max(x, 0), 14 + rand() * 8);
    }
    ctx.lineTo(0, 16);
    ctx.closePath();
    ctx.fill();
    punch(ctx, (c) => {
      for (let i = 0; i < 30; i++) diamond(c, 6 + rand() * 388, 3 + rand() * 10, 1.2, 2);
    });
  }, 19);
}

export function house(marked: boolean, variant: number): HTMLCanvasElement {
  return silhouette(80, 120, 2.2, (ctx) => {
    ctx.fillRect(4, 30, 72, 90);
    for (let x = 4; x < 76; x += 9) ctx.fillRect(x, 24, 6, 7);
    if (variant % 2 === 0) {
      ctx.fillRect(26, 22, 28, 9);
      onionDome(ctx, 40, 23, 26, 22);
      line(ctx, 1, [40, 1, 40, -2]);
    } else {
      ctx.fillRect(56, 6, 12, 24);
      onionDome(ctx, 62, 7, 12, 8);
    }
    punch(ctx, (c) => {
      archHole(c, 22, 40, 12, 18);
      archHole(c, 58, 40, 12, 18);
      line(c, 1, [26, 78, 26, 116, 54, 116, 54, 78]);
      c.beginPath();
      c.arc(40, 78, 14, Math.PI, 0);
      c.lineWidth = 1;
      c.stroke();
      dotsAlong(c, 30, 92, 50, 92, 5, 0.9);
      dotsAlong(c, 30, 104, 50, 104, 5, 0.9);
      circle(c, 48, 98, 1.5);
      if (variant % 2 === 0) star(c, 40, 12, 2.6, 8, 0.5);
      if (marked) {
        line(c, 2.4, [34, 70, 46, 82]);
        line(c, 2.4, [46, 70, 34, 82]);
      }
    });
    ctx.fillRect(21, 40, 2, 18);
    ctx.fillRect(57, 40, 2, 18);
    ctx.fillRect(16, 49, 12, 1.6);
    ctx.fillRect(52, 49, 12, 1.6);
  }, 60 + variant);
}

export function colonnade(): HTMLCanvasElement {
  return silhouette(400, 130, 3.4, (ctx) => {
    ctx.fillRect(0, 0, 400, 30);
    for (let x = 0; x <= 400; x += 50) ctx.fillRect(x - 5, 26, 10, 104);
    for (let x = 0; x < 400; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 30);
      ctx.lineTo(x + 50, 30);
      ctx.lineTo(x + 50, 62);
      ctx.bezierCurveTo(x + 50, 36, x + 25, 34, x + 25, 34);
      ctx.bezierCurveTo(x + 25, 34, x, 36, x, 62);
      ctx.closePath();
      ctx.fill();
    }
    punch(ctx, (c) => {
      for (let x = 12; x < 400; x += 24) star(c, x, 14, 5, 8, 0.55);
      for (let x = 0; x < 400; x += 24) diamond(c, x, 14, 2, 2);
      for (let x = 25; x < 400; x += 50) dotsAlong(c, x - 14, 40, x + 14, 40, 6, 0.9);
    });
  }, 70);
}

export function canopy(): HTMLCanvasElement {
  return silhouette(400, 46, 1.2, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(400, 0);
    ctx.lineTo(400, 16);
    for (let x = 400; x > 0; x -= 40) ctx.quadraticCurveTo(x - 20, 40, x - 40, 16);
    ctx.closePath();
    ctx.fill();
    for (let x = 0; x <= 400; x += 40) {
      line(ctx, 1, [x, 16, x, 34]);
      oval(ctx, x, 37, 2, 4);
    }
    punch(ctx, (c) => {
      for (let x = 20; x < 400; x += 40) {
        star(c, x, 12, 4.5, 8, 0.5);
        dotsAlong(c, x - 14, 22, x + 14, 22, 5, 0.8);
      }
    });
  }, 72);
}

export function hangingLantern(): HTMLCanvasElement {
  return silhouette(20, 50, 0.8, (ctx) => {
    line(ctx, 0.8, [10, 0, 10, 14]);
    poly(ctx, [4, 18, 16, 18, 10, 12]);
    ctx.beginPath();
    ctx.moveTo(4, 18);
    ctx.quadraticCurveTo(0, 30, 6, 40);
    ctx.lineTo(14, 40);
    ctx.quadraticCurveTo(20, 30, 16, 18);
    ctx.fill();
    poly(ctx, [7, 40, 13, 40, 10, 48]);
    punch(ctx, (c) => {
      star(c, 10, 29, 4.2, 8, 0.5);
      dotsAlong(c, 6, 21, 14, 21, 4, 0.6);
      dotsAlong(c, 7, 37, 13, 37, 3, 0.6);
    });
  });
}

export function treasureHeap(seed: number): HTMLCanvasElement {
  const rand = seeded(seed);
  return silhouette(200, 80, 1.5, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(0, 80);
    for (let x = 0; x <= 200; x += 6) {
      const h = 24 + Math.sin((x / 200) * Math.PI) * 40;
      ctx.arc(x, 80 - h, 6, Math.PI, 0);
    }
    ctx.lineTo(200, 80);
    ctx.closePath();
    ctx.fill();
    for (const [x, base, s] of [
      [46, 52, 1],
      [150, 50, 0.9],
    ] as const) vase(ctx, x, base, s);
    crown(ctx, 100, 22);
    punch(ctx, (c) => {
      for (let i = 0; i < 40; i++) {
        const x = 10 + rand() * 180;
        const top = 80 - (24 + Math.sin((x / 200) * Math.PI) * 40);
        const y = top + 8 + rand() * (80 - top - 12);
        if (rand() > 0.5) diamond(c, x, y, 1.6, 2.6);
        else circle(c, x, y, 1.3);
      }
      star(c, 100, 15, 2.2, 6, 0.4);
    });
  }, seed);
}

function vase(ctx: Ctx, x: number, base: number, s: number): void {
  ctx.beginPath();
  ctx.moveTo(x - 4 * s, base - 30 * s);
  ctx.lineTo(x + 4 * s, base - 30 * s);
  ctx.quadraticCurveTo(x + 2 * s, base - 22 * s, x + 10 * s, base - 14 * s);
  ctx.quadraticCurveTo(x + 12 * s, base - 4 * s, x + 6 * s, base);
  ctx.lineTo(x - 6 * s, base);
  ctx.quadraticCurveTo(x - 12 * s, base - 4 * s, x - 10 * s, base - 14 * s);
  ctx.quadraticCurveTo(x - 2 * s, base - 22 * s, x - 4 * s, base - 30 * s);
  ctx.fill();
  oval(ctx, x, base - 30 * s, 6 * s, 1.6 * s);
}

function crown(ctx: Ctx, x: number, y: number): void {
  poly(ctx, [x - 10, y + 14, x - 12, y, x - 6, y + 6, x, y - 4, x + 6, y + 6, x + 12, y, x + 10, y + 14]);
}

export function chest(open: boolean): HTMLCanvasElement {
  return silhouette(64, 56, 0.8, (ctx) => {
    ctx.fillRect(6, 30, 52, 24);
    if (open) {
      ctx.beginPath();
      ctx.moveTo(6, 30);
      ctx.lineTo(0, 6);
      ctx.quadraticCurveTo(30, -2, 56, 8);
      ctx.lineTo(58, 30);
      ctx.closePath();
      ctx.fill();
      for (let x = 10; x < 56; x += 7) circle(ctx, x, 28, 4);
    } else {
      ctx.beginPath();
      ctx.moveTo(6, 30);
      ctx.quadraticCurveTo(32, 12, 58, 30);
      ctx.fill();
    }
    punch(ctx, (c) => {
      dotsAlong(c, 10, 34, 54, 34, 9, 0.8);
      diamond(c, 32, 42, 3, 4);
      line(c, 0.8, [10, 50, 54, 50]);
      if (open) dotsAlong(c, 6, 14, 52, 12, 7, 1.4);
    });
  }, 77);
}

export function sack(): HTMLCanvasElement {
  return silhouette(40, 50, 0.65, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(14, 14);
    ctx.quadraticCurveTo(0, 30, 6, 46);
    ctx.quadraticCurveTo(20, 52, 34, 46);
    ctx.quadraticCurveTo(40, 30, 26, 14);
    ctx.closePath();
    ctx.fill();
    poly(ctx, [12, 4, 28, 4, 24, 14, 16, 14]);
    punch(ctx, (c) => {
      line(c, 0.8, [14, 15, 26, 15]);
      star(c, 20, 32, 4, 6, 0.45);
    });
  }, 81);
}

export function oilLamp(): HTMLCanvasElement {
  return silhouette(50, 24, 0.42, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(6, 16);
    ctx.quadraticCurveTo(10, 6, 26, 8);
    ctx.quadraticCurveTo(38, 8, 48, 4);
    ctx.quadraticCurveTo(44, 12, 36, 16);
    ctx.quadraticCurveTo(22, 22, 6, 16);
    ctx.fill();
    ctx.fillRect(16, 18, 14, 4);
    oval(ctx, 22, 7, 5, 2.5);
    ctx.beginPath();
    ctx.arc(6, 12, 4, Math.PI * 0.5, Math.PI * 1.6);
    ctx.lineWidth = 1.6;
    ctx.stroke();
    punch(ctx, (c) => dotsAlong(c, 14, 13, 32, 13, 5, 0.7));
  });
}

export function titleBanner(): HTMLCanvasElement {
  return silhouette(300, 92, 1.7, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(10, 88);
    ctx.lineTo(10, 30);
    ctx.bezierCurveTo(10, 12, 120, 16, 150, 2);
    ctx.bezierCurveTo(180, 16, 290, 12, 290, 30);
    ctx.lineTo(290, 88);
    ctx.closePath();
    ctx.fill();
    for (const x of [4, 296]) {
      line(ctx, 1.4, [x, 30, x, 76]);
      oval(ctx, x, 80, 2.4, 5);
    }
    punch(ctx, (c) => {
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.font = "900 34px 'Cinzel Decorative'";
      c.fillText("Ali Baba", 150, 42);
      c.font = "700 15px 'Cinzel Decorative'";
      c.fillText("and the Forty Thieves", 150, 70);
      star(c, 150, 15, 4, 8, 0.5);
      dotsAlong(c, 20, 83, 280, 83, 26, 0.9);
      star(c, 26, 52, 5, 8, 0.5);
      star(c, 274, 52, 5, 8, 0.5);
    });
  }, 90);
}

export function endBanner(): HTMLCanvasElement {
  return lettering("The End", "900 30px 'Cinzel Decorative'", 200, 54, 1.0, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(14, 50);
    ctx.lineTo(14, 16);
    ctx.quadraticCurveTo(100, -4, 186, 16);
    ctx.lineTo(186, 50);
    ctx.quadraticCurveTo(100, 40, 14, 50);
    ctx.fill();
    oval(ctx, 10, 32, 8, 18);
    oval(ctx, 190, 32, 8, 18);
  });
}

export function scroll(text: string, widthUnits = 200): HTMLCanvasElement {
  return lettering(text, "700 22px 'Cinzel Decorative'", widthUnits, 44, 0.62, (ctx) => {
    ctx.beginPath();
    ctx.roundRect(14, 6, widthUnits - 28, 32, 4);
    ctx.fill();
    oval(ctx, 12, 22, 7, 18);
    oval(ctx, widthUnits - 12, 22, 7, 18);
  });
}

export function veil(): HTMLCanvasElement {
  return silhouette(30, 90, 1.1, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(14, 0);
    ctx.bezierCurveTo(30, 20, 2, 44, 20, 66);
    ctx.quadraticCurveTo(28, 80, 16, 90);
    ctx.quadraticCurveTo(4, 76, 8, 62);
    ctx.bezierCurveTo(-4, 40, 20, 22, 8, 2);
    ctx.closePath();
    ctx.fill();
    punch(ctx, (c) => {
      for (let i = 0; i < 9; i++) circle(c, 13 + Math.sin(i * 0.9) * 4, 8 + i * 9, 1.1);
    });
  });
}

export function woodpile(): HTMLCanvasElement {
  return silhouette(70, 34, 0.55, (ctx) => {
    ctx.beginPath();
    ctx.roundRect(4, 18, 30, 16, 3);
    ctx.fill();
    for (let i = 0; i < 5; i++) {
      ctx.beginPath();
      ctx.roundRect(30 + (i % 3) * 12, 34 - 8 - Math.floor(i / 3) * 8, 14, 8, 4);
      ctx.fill();
    }
    punch(ctx, (c) => {
      for (let r = 2; r < 7; r += 2.4) {
        c.beginPath();
        c.arc(19, 26, r, 0, Math.PI * 2);
        c.lineWidth = 0.7;
        c.stroke();
      }
      for (let i = 0; i < 5; i++) circle(c, 42 + (i % 3) * 12 - 4, 30 - Math.floor(i / 3) * 8, 1.4);
    });
  });
}
