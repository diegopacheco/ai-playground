import { FLY_BUZZ, FLY_CALM, FLY_PALETTE, RANGER_CALM, RANGER_PALETTE } from "@fly-zord/engine";
import { drawSprite, fillRect, pixelText } from "./pixels";

export const CAM_WIDTH = 96;
export const CAM_HEIGHT = 84;

export type Mood = "idle" | "thinking" | "acting" | "hit" | "down";

export interface CamView {
  readonly kind: "human" | "fly";
  readonly mood: Mood;
  readonly accent: string;
  readonly label: string;
}

const SCALE = 3;

function drawInterior(ctx: CanvasRenderingContext2D, accent: string, frame: number, shake: number): void {
  fillRect(ctx, "#0b0b16", 0, 0, CAM_WIDTH, CAM_HEIGHT);
  fillRect(ctx, "#161630", 8 + shake, 6, 80, 54);
  fillRect(ctx, accent, 8 + shake, 6, 80, 1);
  for (let star = 0; star < 10; star += 1) fillRect(ctx, "#2b2b55", 12 + ((star * 17 + frame / 6) % 72), 10 + ((star * 11) % 44), 1, 1);
  fillRect(ctx, "#1d1d38", 0, 60, CAM_WIDTH, CAM_HEIGHT - 60);
  fillRect(ctx, "#33335c", 0, 60, CAM_WIDTH, 2);
  for (let dial = 0; dial < 5; dial += 1) {
    const lit = (frame / 8 + dial) % 5 < 2;
    fillRect(ctx, lit ? accent : "#2b2b55", 10 + dial * 16, 66, 6, 6);
  }
  fillRect(ctx, "#6b6b7a", 26 + shake, 54, 4, 12);
  fillRect(ctx, "#6b6b7a", 66 + shake, 54, 4, 12);
}

function drawOverlay(ctx: CanvasRenderingContext2D, view: CamView, frame: number): void {
  for (let y = 0; y < CAM_HEIGHT; y += 3) {
    ctx.fillStyle = "#00000044";
    ctx.fillRect(0, y, CAM_WIDTH, 1);
  }
  if (view.mood === "hit") for (let bar = 0; bar < 4; bar += 1) fillRect(ctx, "#ffffff55", 0, (frame * 7 + bar * 21) % CAM_HEIGHT, CAM_WIDTH, 2);
  if (frame % 30 < 18) fillRect(ctx, "#ff3b3b", 6, 4, 4, 4);
  pixelText(ctx, "REC", 13, 2, "#ff9b9b", 7);
  pixelText(ctx, view.label.slice(0, 14), 4, CAM_HEIGHT - 9, view.accent, 7);
  if (view.mood === "thinking" && frame % 24 < 14) pixelText(ctx, "COMPUTING", 34, 2, "#ffe14d", 7);
  if (view.mood === "down") {
    fillRect(ctx, "#000000aa", 0, 0, CAM_WIDTH, CAM_HEIGHT);
    pixelText(ctx, "SIGNAL LOST", 14, 36, "#ff3b3b", 9);
  }
}

export function renderCockpit(ctx: CanvasRenderingContext2D, view: CamView, frame: number): void {
  ctx.imageSmoothingEnabled = false;
  const shake = view.mood === "hit" ? (frame % 2 === 0 ? 2 : -2) : 0;
  drawInterior(ctx, view.accent, frame, shake);

  const flapRate = view.mood === "thinking" ? 3 : view.mood === "acting" ? 4 : 9;
  const sprite = view.kind === "human" ? RANGER_CALM : frame % (flapRate * 2) < flapRate ? FLY_CALM : FLY_BUZZ;
  const palette = view.kind === "human" ? RANGER_PALETTE : FLY_PALETTE;
  const hover = view.mood === "thinking" ? Math.round(Math.sin(frame / 4) * 3) : Math.round(Math.sin(frame / 12) * 2);
  const lean = view.mood === "acting" ? 4 : 0;
  const x = (CAM_WIDTH - sprite.width * SCALE) / 2 + shake;
  const y = 14 + hover + lean;

  if (view.mood === "acting") fillRect(ctx, "#ffffff22", 8, 6, 80, 54);
  drawSprite(ctx, sprite, palette, x, y, SCALE, view.mood === "hit" && frame % 6 < 3 ? { tint: "#ff8080" } : {});
  drawOverlay(ctx, view, frame);
}
