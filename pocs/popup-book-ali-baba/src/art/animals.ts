import { type Ctx, circle, curve, diamond, dotsAlong, line, oval, poly, punch, silhouette, star } from "./canvas";

type Leg = number[];

function legs(ctx: Ctx, width: number, set: Leg[]): void {
  for (const leg of set) {
    line(ctx, width, leg);
    const n = leg.length;
    oval(ctx, (leg[n - 2] ?? 0) + 1, (leg[n - 1] ?? 0) + 0.5, width * 0.75, width * 0.45);
  }
}

const HORSE_LEGS: Leg[][] = [
  [
    [80, 80, 92, 94, 102, 104],
    [74, 82, 82, 98, 86, 112],
    [40, 80, 28, 96, 18, 104],
    [46, 82, 44, 98, 40, 112],
  ],
  [
    [80, 80, 90, 92, 84, 104],
    [74, 82, 78, 98, 76, 112],
    [40, 80, 50, 96, 46, 112],
    [46, 82, 38, 98, 32, 110],
  ],
];

export function horse(frame: number, rider: boolean): HTMLCanvasElement {
  return silhouette(120, 116, 2.1, (ctx) => {
    oval(ctx, 58, 72, 30, 13.5);
    ctx.beginPath();
    ctx.moveTo(74, 64);
    ctx.quadraticCurveTo(84, 50, 90, 38);
    ctx.lineTo(96, 30);
    ctx.lineTo(98, 36);
    ctx.quadraticCurveTo(106, 38, 113, 46);
    ctx.quadraticCurveTo(114, 51, 108, 51);
    ctx.lineTo(99, 50);
    ctx.quadraticCurveTo(94, 58, 90, 70);
    ctx.closePath();
    ctx.fill();
    for (let i = 0; i < 6; i++) poly(ctx, [88 - i * 2.4, 40 + i * 4, 82 - i * 2.4, 40 + i * 4 + 1, 86 - i * 2.4, 44 + i * 4]);
    ctx.beginPath();
    ctx.moveTo(31, 66);
    ctx.quadraticCurveTo(16, 62, 10, 74);
    ctx.quadraticCurveTo(frame === 0 ? 4 : 8, 88, 12, 92);
    ctx.quadraticCurveTo(16, 78, 30, 72);
    ctx.closePath();
    ctx.fill();
    legs(ctx, 4.6, HORSE_LEGS[frame % 2] ?? []);
    if (rider) {
      ctx.beginPath();
      ctx.moveTo(52, 62);
      ctx.quadraticCurveTo(48, 44, 56, 34);
      ctx.lineTo(66, 34);
      ctx.quadraticCurveTo(70, 46, 66, 62);
      ctx.closePath();
      ctx.fill();
      line(ctx, 5, [58, 62, 64, 76, 70, 80]);
      circle(ctx, 62, 25, 6.4);
      poly(ctx, [66, 22, 72, 27, 67, 28.5]);
      ctx.beginPath();
      ctx.moveTo(54, 24);
      ctx.quadraticCurveTo(56, 14, 63, 11);
      ctx.quadraticCurveTo(67, 9, 68, 12);
      ctx.quadraticCurveTo(70, 18, 69, 24);
      ctx.fill();
      curve(ctx, 2, 56, 15, 50, 10, 44, 13);
      poly(ctx, [66, 28, 69, 34, 63, 32]);
      line(ctx, 4.2, [64, 38, 72, frame === 0 ? 26 : 30]);
      ctx.beginPath();
      const tipY = frame === 0 ? 4 : 10;
      ctx.moveTo(71, frame === 0 ? 27 : 31);
      ctx.quadraticCurveTo(76, 14, 86, tipY);
      ctx.quadraticCurveTo(80, 16, 74, frame === 0 ? 28 : 32);
      ctx.closePath();
      ctx.fill();
    }
    punch(ctx, (c) => {
      circle(c, 103, 41, 1.2);
      dotsAlong(c, 46, 70, 66, 70, 6, 1);
      for (let x = 48; x < 66; x += 5) diamond(c, x, 77, 1.4, 2.2);
      dotsAlong(c, 98, 38, 110, 46, 4, 0.7);
      if (rider) {
        circle(c, 64, 23.5, 1);
        star(c, 59, 47, 2.2, 8, 0.5);
      }
    });
  }, 17 + frame);
}

export type Load = "wood" | "jars" | "gold" | "none";

const DONKEY_LEGS: Leg[][] = [
  [
    [62, 46, 66, 58, 70, 66],
    [58, 47, 56, 58, 54, 66],
    [28, 46, 24, 58, 20, 66],
    [32, 47, 34, 58, 36, 66],
  ],
  [
    [62, 46, 62, 58, 62, 66],
    [58, 47, 60, 58, 62, 66],
    [28, 46, 28, 58, 28, 66],
    [32, 47, 30, 58, 28, 66],
  ],
];

export function donkey(frame: number, load: Load, worldH = 1.05): HTMLCanvasElement {
  return silhouette(96, 70, worldH, (ctx) => {
    oval(ctx, 44, 40, 21, 9.5);
    ctx.beginPath();
    ctx.moveTo(58, 36);
    ctx.quadraticCurveTo(66, 30, 70, 26);
    ctx.lineTo(80, 30);
    ctx.quadraticCurveTo(86, 34, 84, 38);
    ctx.lineTo(76, 38);
    ctx.quadraticCurveTo(70, 42, 64, 46);
    ctx.closePath();
    ctx.fill();
    poly(ctx, [70, 27, 66, 12, 73, 25]);
    poly(ctx, [73, 26, 74, 10, 77, 27]);
    line(ctx, 1.6, [24, 38, 18, 48]);
    oval(ctx, 18, 50, 1.8, 3);
    legs(ctx, 3.2, DONKEY_LEGS[frame % 2] ?? []);
    if (load === "wood") {
      ctx.beginPath();
      ctx.roundRect(26, 20, 40, 13, 6);
      ctx.fill();
      for (let i = 0; i < 3; i++) line(ctx, 2.2, [22 + i * 3, 21 + i * 4, 70 - i * 2, 18 + i * 5]);
    } else if (load === "jars") {
      for (const x of [34, 54]) {
        ctx.beginPath();
        ctx.moveTo(x - 3, 16);
        ctx.lineTo(x + 3, 16);
        ctx.quadraticCurveTo(x + 9, 22, x + 8, 32);
        ctx.quadraticCurveTo(x + 6, 40, x, 40);
        ctx.quadraticCurveTo(x - 6, 40, x - 8, 32);
        ctx.quadraticCurveTo(x - 9, 22, x - 3, 16);
        ctx.fill();
        ctx.fillRect(x - 4.5, 13, 9, 3);
      }
    } else if (load === "gold") {
      for (const x of [34, 52]) {
        oval(ctx, x, 27, 8, 8);
        poly(ctx, [x - 3, 20, x + 3, 20, x + 5, 15, x - 5, 15]);
      }
    }
    punch(ctx, (c) => {
      circle(c, 76, 31, 0.9);
      if (load === "wood") for (let i = 0; i < 3; i++) circle(c, 64 - i * 2, 21 + i * 4.5, 1.2);
      if (load === "jars") {
        for (const x of [34, 54]) {
          line(c, 0.7, [x - 7, 27, x + 7, 27]);
          dotsAlong(c, x - 5, 31, x + 5, 31, 4, 0.6);
        }
      }
      if (load === "gold") for (const x of [34, 52]) star(c, x, 28, 2.4, 6, 0.45);
      dotsAlong(c, 34, 43, 52, 43, 5, 0.8);
    });
  }, 23 + frame);
}

const CAMEL_LEGS: Leg[][] = [
  [
    [78, 66, 86, 84, 90, 104],
    [72, 66, 70, 86, 66, 104],
    [36, 66, 30, 86, 24, 104],
    [42, 66, 46, 86, 48, 104],
  ],
  [
    [78, 66, 80, 86, 80, 104],
    [72, 66, 76, 86, 78, 104],
    [36, 66, 38, 86, 38, 104],
    [42, 66, 38, 86, 36, 104],
  ],
];

export function camel(frame: number, rider: boolean): HTMLCanvasElement {
  return silhouette(124, 108, 1.75, (ctx) => {
    ctx.beginPath();
    ctx.moveTo(30, 64);
    ctx.quadraticCurveTo(28, 50, 40, 46);
    ctx.quadraticCurveTo(50, 26, 60, 44);
    ctx.quadraticCurveTo(70, 40, 80, 52);
    ctx.quadraticCurveTo(90, 54, 96, 40);
    ctx.quadraticCurveTo(98, 28, 104, 24);
    ctx.lineTo(114, 26);
    ctx.quadraticCurveTo(118, 30, 114, 33);
    ctx.lineTo(106, 33);
    ctx.quadraticCurveTo(100, 52, 86, 64);
    ctx.quadraticCurveTo(60, 72, 30, 64);
    ctx.fill();
    line(ctx, 2, [30, 58, 22, 70]);
    legs(ctx, 4, CAMEL_LEGS[frame % 2] ?? []);
    if (rider) {
      ctx.beginPath();
      ctx.moveTo(46, 42);
      ctx.quadraticCurveTo(44, 26, 52, 20);
      ctx.lineTo(60, 20);
      ctx.quadraticCurveTo(64, 30, 60, 42);
      ctx.closePath();
      ctx.fill();
      circle(ctx, 56, 13, 5.4);
      oval(ctx, 55, 7, 7, 4.6);
      poly(ctx, [60, 11, 64, 14, 60, 15]);
    }
    punch(ctx, (c) => {
      circle(c, 108, 27.5, 0.9);
      for (let x = 44; x < 74; x += 5) {
        circle(c, x, 52, 1.1);
        line(c, 0.7, [x, 55, x, 59]);
      }
      if (rider) circle(c, 57.5, 11.6, 0.8);
    });
  }, 31 + frame);
}

export function bird(frame: number): HTMLCanvasElement {
  return silhouette(40, 20, 0.22, (ctx) => {
    oval(ctx, 20, 12, 8, 3);
    poly(ctx, [27, 11, 33, 12.5, 27, 13.5]);
    if (frame % 2 === 0) {
      ctx.beginPath();
      ctx.moveTo(14, 11);
      ctx.quadraticCurveTo(10, 2, 2, 1);
      ctx.quadraticCurveTo(14, 4, 22, 11);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(14, 12);
      ctx.quadraticCurveTo(8, 18, 1, 19);
      ctx.quadraticCurveTo(14, 18, 22, 12);
      ctx.fill();
    }
    poly(ctx, [12, 11, 4, 9, 6, 13]);
  });
}
