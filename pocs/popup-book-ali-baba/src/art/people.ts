import { type Ctx, circle, crescent, curve, diamond, dotsAlong, line, oval, poly, punch, silhouette, star } from "./canvas";

export interface FigureArt {
  canvas: HTMLCanvasElement;
  w: number;
  h: number;
  worldH: number;
  joints: Record<string, [number, number]>;
}

export type Turban = "round" | "tall" | "pointed" | "big";
export type Beard = "none" | "short" | "long" | "pointed";
export type Build = "slim" | "fat" | "broad";

export interface ManOptions {
  turban: Turban;
  beard: Beard;
  build: Build;
  mustache?: boolean;
  cape?: boolean;
  sword?: boolean;
  worldH?: number;
  seed?: number;
}

function turban(ctx: Ctx, kind: Turban, hx: number): void {
  if (kind === "round") {
    oval(ctx, hx - 1, 12, 11, 7.5, -0.15);
    circle(ctx, hx - 3, 5, 3);
  } else if (kind === "tall") {
    ctx.beginPath();
    ctx.moveTo(hx - 10, 15);
    ctx.bezierCurveTo(hx - 12, 4, hx - 4, 0.5, hx - 1, 0.5);
    ctx.bezierCurveTo(hx + 4, 0.5, hx + 11, 5, hx + 9, 15);
    ctx.fill();
    oval(ctx, hx - 1, 14, 11, 3.6);
  } else if (kind === "pointed") {
    ctx.beginPath();
    ctx.moveTo(hx - 10, 16);
    ctx.quadraticCurveTo(hx - 8, 6, hx + 1, 1.5);
    ctx.quadraticCurveTo(hx + 6, -0.5, hx + 7, 3);
    ctx.quadraticCurveTo(hx + 3, 3, hx + 3.5, 6);
    ctx.quadraticCurveTo(hx + 10, 9, hx + 9, 16);
    ctx.fill();
    oval(ctx, hx - 1, 15, 11, 3.4);
    circle(ctx, hx + 7.5, 2.5, 1.8);
  } else {
    oval(ctx, hx - 1, 11, 14, 9.5, -0.1);
    curve(ctx, 2.4, hx - 4, 4, hx - 9, 0.5, hx - 3, 1);
  }
}

function turbanHoles(ctx: Ctx, kind: Turban, hx: number): void {
  if (kind === "big") {
    star(ctx, hx + 4, 9, 3, 8, 0.5);
    dotsAlong(ctx, hx - 12, 13, hx + 10, 12, 7, 0.9);
  } else if (kind === "pointed") {
    dotsAlong(ctx, hx - 7, 13.5, hx + 6, 13, 5, 0.85);
    diamond(ctx, hx, 8, 1.4, 2.2);
  } else if (kind === "tall") {
    dotsAlong(ctx, hx - 8, 14, hx + 7, 14, 6, 0.8);
    star(ctx, hx - 1, 7, 2.4, 8, 0.5);
  } else {
    dotsAlong(ctx, hx - 9, 13, hx + 8, 11.5, 6, 0.85);
  }
}

function beard(ctx: Ctx, kind: Beard, hx: number, hy: number): void {
  if (kind === "none") return;
  ctx.beginPath();
  ctx.moveTo(hx + 6.5, hy + 3.5);
  if (kind === "short") {
    ctx.quadraticCurveTo(hx + 6.5, hy + 11, hx + 1, hy + 13);
  } else if (kind === "long") {
    ctx.quadraticCurveTo(hx + 7, hy + 16, hx + 2, hy + 22);
    ctx.quadraticCurveTo(hx - 1, hy + 15, hx - 3, hy + 12);
  } else {
    ctx.quadraticCurveTo(hx + 8, hy + 12, hx + 8.5, hy + 19);
    ctx.quadraticCurveTo(hx + 2, hy + 13, hx - 1, hy + 11);
  }
  ctx.quadraticCurveTo(hx - 5, hy + 11, hx - 6, hy + 4);
  ctx.closePath();
  ctx.fill();
}

function body(ctx: Ctx, build: Build): [number, number, number] {
  ctx.beginPath();
  if (build === "fat") {
    ctx.moveTo(27, 29);
    ctx.bezierCurveTo(19, 31, 17, 44, 18, 56);
    ctx.bezierCurveTo(18, 72, 16, 84, 15, 93);
    ctx.lineTo(56, 93);
    ctx.bezierCurveTo(57, 84, 63, 72, 62, 60);
    ctx.bezierCurveTo(61, 44, 52, 32, 42, 29);
    ctx.closePath();
    ctx.fill();
    return [20, 60, 56];
  }
  const s = build === "broad" ? 3 : 0;
  ctx.moveTo(28, 29);
  ctx.bezierCurveTo(20 - s, 30, 18 - s, 40, 20, 52);
  ctx.bezierCurveTo(21, 68, 16, 84, 13, 93);
  ctx.lineTo(51, 93);
  ctx.bezierCurveTo(48, 80, 45, 66, 45, 54);
  ctx.bezierCurveTo(46 + s, 42, 45 + s, 32, 38, 29);
  ctx.closePath();
  ctx.fill();
  return [22, 45, 55];
}

function slippers(ctx: Ctx): void {
  ctx.beginPath();
  ctx.moveTo(36, 91);
  ctx.lineTo(52, 91);
  ctx.quadraticCurveTo(57, 91, 59, 86);
  ctx.quadraticCurveTo(61, 92, 56, 96);
  ctx.lineTo(36, 97);
  ctx.closePath();
  ctx.fill();
  poly(ctx, [15, 91, 28, 91, 31, 97, 13, 97]);
}

export function man(options: ManOptions): FigureArt {
  const worldH = options.worldH ?? 1.7;
  const hx = options.build === "fat" ? 37 : 35;
  const hy = 20;
  const joints: Record<string, [number, number]> = { shoulder: options.build === "fat" ? [34, 35] : [32, 34], head: [hx, hy] };
  const canvas = silhouette(66, 100, worldH, (ctx) => {
    if (options.cape) {
      ctx.beginPath();
      ctx.moveTo(26, 29);
      ctx.bezierCurveTo(14, 40, 6, 70, 3, 90);
      ctx.quadraticCurveTo(12, 86, 20, 92);
      ctx.lineTo(30, 40);
      ctx.closePath();
      ctx.fill();
    }
    turban(ctx, options.turban, hx);
    circle(ctx, hx, hy, options.build === "fat" ? 8.5 : 7.5);
    if (options.turban === "pointed") poly(ctx, [hx + 5, hy - 3, hx + 12.5, hy + 3.5, hx + 9, hy + 4.5, hx + 6, hy + 4]);
    else if (options.build === "fat") oval(ctx, hx + 8, hy + 1.5, 3.4, 2.8);
    else poly(ctx, [hx + 5.5, hy - 3, hx + 11, hy + 2.5, hx + 6, hy + 4]);
    beard(ctx, options.beard, hx, hy);
    if (options.mustache) {
      curve(ctx, 1.7, hx + 6, hy + 5, hx + 12, hy + 6.5, hx + 13, hy + 2);
      curve(ctx, 1.7, hx + 5, hy + 5, hx - 1, hy + 7, hx - 2, hy + 3);
    }
    ctx.fillRect(hx - 5, hy + 4, 8, 8);
    const [back, front, sash] = body(ctx, options.build);
    slippers(ctx);
    if (options.sword) {
      ctx.beginPath();
      ctx.moveTo(back + 6, sash + 2);
      ctx.quadraticCurveTo(back - 6, sash + 14, back - 9, sash + 30);
      ctx.quadraticCurveTo(back - 2, sash + 16, back + 9, sash + 4);
      ctx.closePath();
      ctx.fill();
      oval(ctx, back + 8, sash + 1, 3.5, 1.6, -0.6);
    }
    punch(ctx, (c) => {
      circle(c, hx + 2, hy - 1.8, 1.15);
      turbanHoles(c, options.turban, hx);
      for (let x = back + 2.5; x < front - 1.5; x += 3.6) diamond(c, x, sash, 1.2, 1.7);
      dotsAlong(c, 18, 88.5, 48, 88.5, 10, 0.75);
      if (options.build === "fat") {
        for (const [x, y] of [
          [30, 42],
          [44, 46],
          [36, 64],
          [50, 70],
          [28, 76],
          [42, 82],
        ] as const)
          star(c, x, y, 2.3, 8, 0.5);
      } else {
        dotsAlong(c, 40, 34, 41, 50, 5, 0.75);
        for (const [x, y] of [
          [30, 66],
          [38, 74],
          [26, 80],
          [42, 84],
        ] as const)
          crescent(c, x, y, 1.8);
      }
      if (options.cape) line(c, 0.8, [9, 80, 22, 40]);
    });
  }, options.seed ?? 3);
  return { canvas, w: 66, h: 100, worldH, joints };
}

export interface ArmOptions {
  length?: number;
  sleeve?: "wide" | "fitted";
  hold?: "axe" | "scimitar" | "dagger" | "none";
  bangles?: boolean;
  worldScale: number;
}

export function arm(options: ArmOptions): FigureArt {
  const len = options.length ?? 34;
  const hold = options.hold ?? "none";
  const extra = hold === "axe" ? 26 : hold === "scimitar" ? 44 : hold === "dagger" ? 16 : 2;
  const w = 28;
  const h = len + extra + 2;
  const px = 12;
  const canvas = silhouette(w, h, (h / 100) * options.worldScale, (ctx) => {
    if (options.sleeve === "fitted") {
      poly(ctx, [px - 4, 1, px + 4, 1, px + 3, len * 0.82, px - 2.5, len * 0.82]);
    } else {
      poly(ctx, [px - 5, 1, px + 5, 1, px + 7, len * 0.6, px + 8.5, len * 0.72, px - 8.5, len * 0.72, px - 7, len * 0.6]);
      ctx.fillRect(px - 3, len * 0.7, 6, len * 0.16);
    }
    circle(ctx, px, len * 0.9, 3.6);
    if (hold === "axe") {
      line(ctx, 2.2, [px, len * 0.86, px, len + 22]);
      poly(ctx, [px, len + 12, px + 10, len + 9, px + 13, len + 17, px + 10, len + 25, px, len + 21]);
    } else if (hold === "scimitar") {
      ctx.fillRect(px - 5, len * 0.93, 10, 1.8);
      ctx.beginPath();
      ctx.moveTo(px - 1.6, len + 1);
      ctx.quadraticCurveTo(px - 3, len + 26, px + 10, len + 42);
      ctx.quadraticCurveTo(px + 4, len + 24, px + 2.2, len + 1);
      ctx.closePath();
      ctx.fill();
    } else if (hold === "dagger") {
      ctx.fillRect(px - 4, len * 0.93, 8, 1.6);
      poly(ctx, [px - 1.5, len + 1, px + 1.5, len + 1, px, len + 14]);
    }
    punch(ctx, (c) => {
      if (options.sleeve !== "fitted") dotsAlong(c, px - 6, len * 0.66, px + 6, len * 0.66, 5, 0.7);
      if (options.bangles) {
        line(c, 0.7, [px - 2.5, len * 0.74, px + 2.5, len * 0.74]);
        line(c, 0.7, [px - 2.4, len * 0.78, px + 2.4, len * 0.78]);
      }
      if (hold === "scimitar") line(c, 0.6, [px, len + 6, px + 1, len + 20]);
    });
  });
  return { canvas, w, h, worldH: (h / 100) * options.worldScale, joints: { pivot: [px, 3], hand: [px, len * 0.9] } };
}

export function woman(worldH = 1.62): FigureArt {
  const hx = 36;
  const hy = 19;
  const canvas = silhouette(66, 100, worldH, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(hx + 5, 12);
    ctx.bezierCurveTo(hx + 2, 6, hx - 10, 5, hx - 10, 14);
    ctx.bezierCurveTo(hx - 14, 30, hx - 22, 44, hx - 26, 60);
    ctx.quadraticCurveTo(hx - 20, 56, hx - 18, 62);
    ctx.quadraticCurveTo(hx - 14, 54, hx - 10, 58);
    ctx.bezierCurveTo(hx - 10, 42, hx - 6, 30, hx - 3, 26);
    ctx.closePath();
    ctx.fill();
    circle(ctx, hx, hy, 6.6);
    poly(ctx, [hx + 5, hy - 1, hx + 8.6, hy + 2.6, hx + 5.4, hy + 3.4]);
    oval(ctx, hx + 5, hy + 5, 1.6, 1.1);
    ctx.fillRect(hx - 4, hy + 4, 6, 7);
    ctx.beginPath();
    ctx.moveTo(31, 27);
    ctx.quadraticCurveTo(27, 30, 28, 38);
    ctx.lineTo(29, 48);
    ctx.bezierCurveTo(22, 62, 14, 80, 10, 92);
    for (let x = 10; x < 58; x += 6) ctx.quadraticCurveTo(x + 3, 96, x + 6, 92);
    ctx.bezierCurveTo(52, 78, 46, 62, 41, 48);
    ctx.bezierCurveTo(42, 42, 45, 36, 44, 33);
    ctx.quadraticCurveTo(42, 28, 38, 27);
    ctx.closePath();
    ctx.fill();
    poly(ctx, [40, 92, 50, 93, 53, 97, 40, 97]);
    punch(ctx, (c) => {
      circle(c, hx + 2, hy - 1.6, 1.05);
      dotsAlong(c, hx - 7, 11, hx + 3, 9, 5, 0.75);
      dotsAlong(c, hx - 13, 30, hx - 22, 54, 8, 0.75);
      for (let x = 31; x < 41; x += 2.8) diamond(c, x, 47.5, 0.9, 1.5);
      for (const [y, from, to, n] of [
        [60, 25, 44, 5],
        [71, 20, 48, 7],
        [82, 15, 52, 9],
      ] as const) {
        for (let i = 0; i < n; i++) {
          const x = from + ((to - from) * i) / (n - 1);
          if (i % 2 === 0) star(c, x, y, 1.7, 6, 0.45);
          else crescent(c, x, y, 1.5, -1.2);
        }
      }
      dotsAlong(c, 13, 90, 55, 90, 12, 0.6);
    });
  }, 9);
  return { canvas, w: 66, h: 100, worldH, joints: { shoulder: [34, 33], head: [hx, hy] } };
}

export function thiefHead(): FigureArt {
  const hx = 22;
  const hy = 18;
  const canvas = silhouette(40, 34, 0.62, (ctx) => {
    turban(ctx, "pointed", hx);
    circle(ctx, hx, hy, 7.5);
    poly(ctx, [hx + 5, hy - 3, hx + 12.5, hy + 3.5, hx + 9, hy + 4.5, hx + 6, hy + 4]);
    curve(ctx, 1.7, hx + 6, hy + 5, hx + 12, hy + 6.5, hx + 13, hy + 2);
    beard(ctx, "pointed", hx, hy);
    oval(ctx, hx - 1, 34, 13, 7);
    punch(ctx, (c) => {
      circle(c, hx + 2, hy - 1.8, 1.15);
      turbanHoles(c, "pointed", hx);
    });
  });
  return { canvas, w: 40, h: 34, worldH: 0.62, joints: {} };
}

export function musician(worldH = 1.15): FigureArt {
  const canvas = silhouette(80, 70, worldH, (ctx) => {
    turbanAt(ctx, 34, 10);
    circle(ctx, 36, 17, 6.5);
    poly(ctx, [41, 14, 46, 19, 41.5, 20.5]);
    ctx.fillRect(32, 21, 7, 6);
    ctx.beginPath();
    ctx.moveTo(30, 25);
    ctx.bezierCurveTo(20, 30, 20, 46, 24, 54);
    ctx.lineTo(10, 62);
    ctx.quadraticCurveTo(30, 70, 62, 66);
    ctx.lineTo(48, 52);
    ctx.bezierCurveTo(48, 40, 46, 28, 38, 25);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(48, 36);
    ctx.lineTo(70, 34);
    ctx.lineTo(66, 60);
    ctx.lineTo(52, 60);
    ctx.closePath();
    ctx.fill();
    oval(ctx, 60, 35, 10, 3);
    punch(ctx, (c) => {
      circle(c, 38, 15.5, 1);
      for (let x = 54; x < 66; x += 4) diamond(c, x, 47, 1.2, 4);
      dotsAlong(c, 52, 39, 67, 38, 5, 0.7);
    });
  }, 13);
  return { canvas, w: 80, h: 70, worldH, joints: { hand: [56, 30] } };
}

function turbanAt(ctx: Ctx, x: number, y: number): void {
  oval(ctx, x, y, 9, 6, -0.15);
  circle(ctx, x - 2, y - 5, 2.4);
}
