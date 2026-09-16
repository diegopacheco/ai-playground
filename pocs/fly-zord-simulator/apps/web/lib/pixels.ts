import type { Palette, Sprite } from "@fly-zord/engine";

export interface DrawOptions {
  readonly flip?: boolean;
  readonly tint?: string;
  readonly alpha?: number;
}

export function drawSprite(ctx: CanvasRenderingContext2D, sprite: Sprite, palette: Palette, x: number, y: number, scale: number, options: DrawOptions = {}): void {
  const previousAlpha = ctx.globalAlpha;
  if (options.alpha !== undefined) ctx.globalAlpha = options.alpha;
  for (let row = 0; row < sprite.height; row += 1) {
    const line = sprite.rows[row] ?? "";
    for (let column = 0; column < sprite.width; column += 1) {
      const key = line[options.flip ? sprite.width - 1 - column : column];
      if (!key || key === ".") continue;
      const color = options.tint ?? palette[key];
      if (!color) continue;
      ctx.fillStyle = color;
      ctx.fillRect(x + column * scale, y + row * scale, scale, scale);
    }
  }
  ctx.globalAlpha = previousAlpha;
}

export function fillRect(ctx: CanvasRenderingContext2D, color: string, x: number, y: number, width: number, height: number): void {
  ctx.fillStyle = color;
  ctx.fillRect(Math.round(x), Math.round(y), Math.round(width), Math.round(height));
}

export function pixelText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, size = 8): void {
  ctx.font = `${size}px "Courier New", monospace`;
  ctx.textBaseline = "top";
  ctx.fillStyle = color;
  ctx.fillText(text, Math.round(x), Math.round(y));
}
