import { RepeatWrapping, type Texture } from "three";
import { circle, diamond, makeCanvas, poly, seeded, star, toTexture } from "./canvas";

export function paperTexture(): Texture {
  const { canvas, ctx } = makeCanvas(512, 512);
  const rand = seeded(11);
  ctx.fillStyle = "#b8b8b8";
  ctx.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 2600; i++) {
    const v = 150 + Math.floor(rand() * 105);
    ctx.strokeStyle = `rgba(${v},${v},${v},0.35)`;
    ctx.lineWidth = 0.6 + rand() * 1.2;
    const x = rand() * 512;
    const y = rand() * 512;
    const a = rand() * Math.PI;
    const l = 6 + rand() * 30;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.quadraticCurveTo(x + Math.cos(a) * l * 0.5 + rand() * 6, y + Math.sin(a) * l * 0.5, x + Math.cos(a) * l, y + Math.sin(a) * l);
    ctx.stroke();
  }
  const texture = toTexture(canvas);
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  return texture;
}

export function woodTexture(): Texture {
  const { canvas, ctx } = makeCanvas(512, 512);
  const rand = seeded(5);
  ctx.fillStyle = "#4a2312";
  ctx.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 160; i++) {
    const y = rand() * 512;
    ctx.strokeStyle = rand() > 0.5 ? "rgba(120,62,30,0.35)" : "rgba(30,12,4,0.4)";
    ctx.lineWidth = 1 + rand() * 3;
    ctx.beginPath();
    ctx.moveTo(0, y);
    for (let x = 0; x <= 512; x += 32) ctx.lineTo(x, y + Math.sin(x * 0.02 + i) * 4);
    ctx.stroke();
  }
  const texture = toTexture(canvas);
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  return texture;
}

export function rugTexture(): Texture {
  const W = 1024;
  const H = 576;
  const { canvas, ctx } = makeCanvas(W, H);
  ctx.fillStyle = "#7d1420";
  ctx.fillRect(0, 0, W, H);
  const bands: [number, string][] = [
    [0, "#1c1f4a"],
    [26, "#d9a441"],
    [34, "#1c1f4a"],
    [78, "#d9a441"],
    [84, "#7d1420"],
  ];
  for (const [inset, color] of bands) {
    ctx.fillStyle = color;
    ctx.fillRect(inset, inset, W - inset * 2, H - inset * 2);
  }
  ctx.fillStyle = "#f0d9a8";
  for (let x = 50; x < W - 40; x += 34) {
    diamond(ctx, x, 56, 9, 14);
    diamond(ctx, x, H - 56, 9, 14);
  }
  for (let y = 90; y < H - 80; y += 34) {
    diamond(ctx, 56, y, 14, 9);
    diamond(ctx, W - 56, y, 14, 9);
  }
  ctx.fillStyle = "#1c1f4a";
  poly(ctx, [W / 2, 120, W / 2 + 300, H / 2, W / 2, H - 120, W / 2 - 300, H / 2]);
  ctx.fillStyle = "#d9a441";
  star(ctx, W / 2, H / 2, 110, 8, 0.62);
  ctx.fillStyle = "#7d1420";
  star(ctx, W / 2, H / 2, 70, 8, 0.6, 0.2);
  ctx.fillStyle = "#f0d9a8";
  circle(ctx, W / 2, H / 2, 22);
  for (const [x, y] of [
    [160, 150],
    [W - 160, 150],
    [160, H - 150],
    [W - 160, H - 150],
  ] as const) {
    ctx.fillStyle = "#d9a441";
    star(ctx, x, y, 38, 8, 0.6);
    ctx.fillStyle = "#1c1f4a";
    circle(ctx, x, y, 12);
  }
  const rand = seeded(3);
  ctx.fillStyle = "rgba(0,0,0,0.12)";
  for (let i = 0; i < 9000; i++) ctx.fillRect(rand() * W, rand() * H, 2, 2);
  return toTexture(canvas);
}

export function latticeTexture(): Texture {
  const { canvas, ctx } = makeCanvas(512, 256);
  ctx.fillStyle = "#c8902f";
  ctx.fillRect(0, 0, 512, 256);
  ctx.globalCompositeOperation = "destination-out";
  for (let x = 32; x < 512; x += 64) {
    for (let y = 32; y < 256; y += 64) {
      star(ctx, x, y, 22, 8, 0.55);
      if (x + 32 < 512 && y + 32 < 256) diamond(ctx, x + 32, y + 32, 7, 7);
    }
  }
  const texture = toTexture(canvas);
  texture.wrapS = RepeatWrapping;
  return texture;
}

export function audienceTexture(): Texture {
  const { canvas, ctx } = makeCanvas(2048, 256);
  const rand = seeded(21);
  ctx.fillStyle = "#000";
  for (let x = 40; x < 2048; x += 120 + rand() * 60) {
    const h = 120 + rand() * 50;
    const y = 256 - h;
    ctx.beginPath();
    ctx.ellipse(x, 256, 70, 70, 0, Math.PI, 0);
    ctx.fill();
    circle(ctx, x, y + 40, 32);
    if (rand() > 0.4) {
      ctx.beginPath();
      ctx.ellipse(x - 2, y + 14, 40, 26, 0, 0, Math.PI * 2);
      ctx.fill();
      circle(ctx, x - 4, y - 8, 10);
    } else {
      ctx.beginPath();
      ctx.moveTo(x - 34, y + 40);
      ctx.quadraticCurveTo(x - 20, y - 30, x + 6, y - 10);
      ctx.quadraticCurveTo(x + 40, y + 10, x + 34, y + 40);
      ctx.fill();
    }
  }
  return toTexture(canvas);
}
