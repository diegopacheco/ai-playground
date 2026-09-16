import { MOVES, ZORD_IDLE, ZORD_PALETTES, opponentOf } from "@fly-zord/engine";
import type { Battle, Side } from "@fly-zord/engine";
import type { Action } from "./battlefield";
import { drawSprite, fillRect, pixelText } from "./pixels";

export const VIEW_WIDTH = 132;
export const VIEW_HEIGHT = 92;

const HORIZON = 54;

function drawWorld(ctx: CanvasRenderingContext2D, seed: number, shake: number): void {
  fillRect(ctx, "#1b0f3c", 0, 0, VIEW_WIDTH, 26);
  fillRect(ctx, "#2c1860", 0, 26, VIEW_WIDTH, 16);
  fillRect(ctx, "#3a1f6b", 0, 42, VIEW_WIDTH, HORIZON - 42);
  for (let index = 0; index < 9; index += 1) {
    const width = 10 + ((seed + index * 23) % 12);
    const height = 12 + ((seed + index * 47) % 22);
    const x = index * 15 - 4 + shake;
    fillRect(ctx, index % 2 === 0 ? "#1a1a38" : "#242452", x, HORIZON - height, width, height);
    for (let window = 0; window < 4; window += 1) {
      if ((seed + index * 5 + window) % 3 !== 0) continue;
      fillRect(ctx, "#ffe14d", x + 2 + (window % 2) * 5, HORIZON - height + 3 + Math.floor(window / 2) * 6, 2, 3);
    }
  }
  fillRect(ctx, "#20203a", 0, HORIZON, VIEW_WIDTH, VIEW_HEIGHT - HORIZON);
  for (let lane = 0; lane < 7; lane += 1) {
    const depth = lane / 7;
    const y = HORIZON + depth * depth * (VIEW_HEIGHT - HORIZON);
    const width = 4 + depth * 22;
    fillRect(ctx, "#4a4a70", VIEW_WIDTH / 2 - width / 2 + shake, y, width, 1 + depth * 2);
  }
}

function drawTarget(ctx: CanvasRenderingContext2D, battle: Battle, side: Side, action: Action | null, frame: number): void {
  const target = battle[opponentOf(side)];
  const palette = ZORD_PALETTES[target.palette] ?? ZORD_PALETTES.crimson!;
  const phase = action ? frame / 54 : 0;
  const charging = action?.side === opponentOf(side) && MOVES[action.move].kind === "attack";
  const scale = charging && phase > 0.3 && phase < 0.75 ? 5 : 4;
  const bob = Math.round(Math.sin(frame / 10) * 2);
  const x = Math.round((VIEW_WIDTH - ZORD_IDLE.width * scale) / 2);
  const y = HORIZON - ZORD_IDLE.height * scale + 30 + bob;
  const struck = action?.side === side && action.damage > 0 && phase > 0.45 && phase < 0.7;
  drawSprite(ctx, ZORD_IDLE, palette, x, y, scale, { flip: side === "right", ...(struck ? { tint: "#ffffff" } : {}) });
  if (target.guarding) fillRect(ctx, frame % 8 < 4 ? "#3de0e0" : "#a0f4ff", x - 5, y + 30, 4, 46);
}

function drawOutgoing(ctx: CanvasRenderingContext2D, action: Action, frame: number): void {
  const phase = frame / 54;
  const center = VIEW_WIDTH / 2;
  if (action.move === "missiles") {
    for (let rocket = 0; rocket < 3; rocket += 1) {
      const travel = Math.min(1, Math.max(0, (phase - 0.2 - rocket * 0.07) / 0.35));
      if (travel <= 0 || travel >= 1) continue;
      const size = Math.round(7 - travel * 5);
      const x = center + (rocket - 1) * 26 * (1 - travel);
      const y = VIEW_HEIGHT - 8 - travel * (VIEW_HEIGHT - HORIZON + 14);
      fillRect(ctx, "#e8e8f0", x, y, size, size);
      fillRect(ctx, frame % 2 === 0 ? "#ffe14d" : "#ff5ad0", x, y + size, size, Math.max(1, size - 2));
    }
  }
  if (action.move === "jab" && phase > 0.2 && phase < 0.7) {
    const punch = Math.min(1, (phase - 0.2) / 0.3);
    const size = Math.round(22 - punch * 10);
    fillRect(ctx, "#f2a03d", center - 30 + punch * 18, VIEW_HEIGHT - 26 - punch * 24, size, size);
    fillRect(ctx, "#6b0f1a", center - 30 + punch * 18, VIEW_HEIGHT - 26 - punch * 24 + size - 4, size, 4);
  }
  if (action.move === "saber" && phase > 0.25 && phase < 0.7) {
    const sweep = (phase - 0.25) / 0.45;
    for (let slice = 0; slice < 6; slice += 1) {
      fillRect(ctx, slice % 2 === 0 ? "#ffffff" : "#3de0e0", 12 + slice * 18 - sweep * 10, 20 + sweep * 26 + slice * 2, 16, 3);
    }
  }
  if (action.move === "charge") {
    for (let spark = 0; spark < 6; spark += 1) {
      const rise = (frame * 3 + spark * 15) % 60;
      fillRect(ctx, spark % 2 === 0 ? "#3de0a0" : "#ffe14d", 10 + spark * 20, VIEW_HEIGHT - 10 - rise, 3, 4);
    }
  }
  if (action.move === "guard") {
    const alpha = frame % 10 < 5 ? "#3de0e066" : "#3de0e033";
    fillRect(ctx, alpha, 6, 6, VIEW_WIDTH - 12, HORIZON + 10);
  }
}

function drawGlass(ctx: CanvasRenderingContext2D, battle: Battle, side: Side, accent: string, frame: number, hurt: boolean): void {
  const target = battle[opponentOf(side)];
  fillRect(ctx, "#0b0b16", 0, 0, VIEW_WIDTH, 10);
  fillRect(ctx, "#0b0b16", 0, VIEW_HEIGHT - 14, VIEW_WIDTH, 14);
  fillRect(ctx, "#0b0b16", 0, 0, 8, VIEW_HEIGHT);
  fillRect(ctx, "#0b0b16", VIEW_WIDTH - 8, 0, 8, VIEW_HEIGHT);
  fillRect(ctx, "#2b2b55", 8, 10, VIEW_WIDTH - 16, 1);
  fillRect(ctx, "#2b2b55", 8, VIEW_HEIGHT - 15, VIEW_WIDTH - 16, 1);
  for (let rivet = 0; rivet < 6; rivet += 1) {
    fillRect(ctx, "#3a3a66", 10 + rivet * 22, 4, 2, 2);
    fillRect(ctx, "#3a3a66", 10 + rivet * 22, VIEW_HEIGHT - 6, 2, 2);
  }
  const center = Math.round(VIEW_WIDTH / 2);
  fillRect(ctx, accent, center - 7, 40, 5, 1);
  fillRect(ctx, accent, center + 3, 40, 5, 1);
  fillRect(ctx, accent, center - 1, 34, 1, 5);
  fillRect(ctx, accent, center - 1, 42, 1, 5);
  pixelText(ctx, "TGT", 10, VIEW_HEIGHT - 11, accent, 7);
  fillRect(ctx, "#101028", 30, VIEW_HEIGHT - 11, 42, 7);
  fillRect(ctx, target.hp > 40 ? "#3de0a0" : "#ff5a5a", 31, VIEW_HEIGHT - 10, Math.max(1, Math.round((target.hp / 100) * 40)), 5);
  pixelText(ctx, `${target.hp}`, 76, VIEW_HEIGHT - 11, "#e8e8f8", 7);
  if (hurt) {
    fillRect(ctx, "#ff000033", 8, 10, VIEW_WIDTH - 16, VIEW_HEIGHT - 25);
    for (let crack = 0; crack < 5; crack += 1) fillRect(ctx, "#ffffff88", 20 + crack * 18, 16 + ((frame + crack * 9) % 40), 10, 1);
  }
  if (frame % 40 < 20) pixelText(ctx, "ONBOARD", VIEW_WIDTH - 52, 2, "#9a94c8", 7);
}

export function renderOnboard(ctx: CanvasRenderingContext2D, battle: Battle, side: Side, action: Action | null, frame: number, accent: string): void {
  ctx.imageSmoothingEnabled = false;
  const phase = action ? frame / 54 : 0;
  const hurt = action !== null && action.side !== side && action.damage > 0 && phase > 0.45 && phase < 0.8;
  const shake = hurt ? (frame % 2 === 0 ? 3 : -3) : 0;
  drawWorld(ctx, battle.seed, shake);
  drawTarget(ctx, battle, side, action, frame);
  if (action && action.side === side) drawOutgoing(ctx, action, frame);
  drawGlass(ctx, battle, side, accent, frame, hurt);
}
