import type { Mood } from "./cockpit";
import { fillRect } from "./pixels";

export const STRIP_WIDTH = 132;
export const STRIP_HEIGHT = 30;

const AMPLITUDE: Readonly<Record<Mood, number>> = { idle: 0.3, thinking: 1, acting: 0.75, hit: 0.95, down: 0.05 };
export const WING_BEAT: Readonly<Record<Mood, number>> = { idle: 186, thinking: 244, acting: 212, hit: 268, down: 0 };

export const LOOP_STAGES: readonly string[] = ["read", "pick", "swing"];

export function loopStage(mood: Mood): number {
  if (mood === "thinking") return 1;
  if (mood === "acting") return 2;
  if (mood === "hit") return 0;
  return 0;
}

function noise(index: number): number {
  const value = Math.sin(index * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

export function renderNeural(ctx: CanvasRenderingContext2D, mood: Mood, accent: string, frame: number): void {
  ctx.imageSmoothingEnabled = false;
  fillRect(ctx, "#0b0b16", 0, 0, STRIP_WIDTH, STRIP_HEIGHT);
  for (let x = 0; x < STRIP_WIDTH; x += 11) fillRect(ctx, "#1c1c38", x, 0, 1, STRIP_HEIGHT);
  fillRect(ctx, "#1c1c38", 0, STRIP_HEIGHT / 2, STRIP_WIDTH, 1);
  const amplitude = AMPLITUDE[mood] * (STRIP_HEIGHT / 2 - 2);
  for (let x = 0; x < STRIP_WIDTH; x += 1) {
    const sample = x + frame * 2;
    const wave = Math.sin(sample / 3) * 0.6 + Math.sin(sample / 1.3) * 0.25 + (noise(sample) - 0.5) * 0.7;
    const height = Math.max(1, Math.round(Math.abs(wave) * amplitude));
    const top = Math.round(STRIP_HEIGHT / 2 - height);
    fillRect(ctx, x > STRIP_WIDTH - 12 ? "#ffffff" : accent, x, top, 1, height * 2);
  }
}
