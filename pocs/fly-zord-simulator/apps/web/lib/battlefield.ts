import { MOVES, ZORD_ATTACK, ZORD_GUARD, ZORD_IDLE, ZORD_PALETTES, opponentOf, skyline } from "@fly-zord/engine";
import type { Battle, MoveId, Side } from "@fly-zord/engine";
import { drawSprite, fillRect, pixelText } from "./pixels";

export const WIDTH = 320;
export const HEIGHT = 180;
export const ACTION_FRAMES = 54;

const GROUND = 152;
const SCALE = 4;
const LEFT_X = 36;
const RIGHT_X = 220;

export interface Action {
  readonly side: Side;
  readonly move: MoveId;
  readonly damage: number;
  readonly critical: boolean;
  readonly blocked: boolean;
}

function star(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, color: string): void {
  fillRect(ctx, color, x - size, y - 1, size * 2, 2);
  fillRect(ctx, color, x - 1, y - size, 2, size * 2);
  fillRect(ctx, color, x - size / 2, y - size / 2, size, size);
}

function drawSky(ctx: CanvasRenderingContext2D, seed: number, shake: number): void {
  fillRect(ctx, "#150c30", 0, 0, WIDTH, 60);
  fillRect(ctx, "#241452", 0, 60, WIDTH, 40);
  fillRect(ctx, "#3a1f6b", 0, 100, WIDTH, 52);
  for (let index = 0; index < 40; index += 1) {
    const x = (seed * 13 + index * 61) % WIDTH;
    const y = (seed * 7 + index * 29) % 90;
    fillRect(ctx, index % 5 === 0 ? "#ffe14d" : "#8f8fd0", x, y, 1, 1);
  }
  fillRect(ctx, "#ffe14d", 268 + shake, 18, 16, 16);
  fillRect(ctx, "#3a1f6b", 262 + shake, 14, 10, 12);
}

function drawCity(ctx: CanvasRenderingContext2D, seed: number, shake: number): void {
  for (const building of skyline(seed, WIDTH, 14)) {
    const top = GROUND - building.height;
    fillRect(ctx, building.shade === "d" ? "#141428" : building.shade === "m" ? "#22224a" : "#2e2e63", building.x + shake, top, building.width, building.height);
    fillRect(ctx, "#0a0a18", building.x + shake, top, building.width, 2);
    const columns = Math.max(1, Math.floor(building.width / 4));
    for (const slot of building.windows) {
      const x = building.x + (slot % columns) * 4 + 1 + shake;
      const y = top + Math.floor(slot / columns) * 5 + 3;
      if (y > GROUND - 4) continue;
      fillRect(ctx, slot % 7 === 0 ? "#3de0a0" : "#ffe14d", x, y, 2, 3);
    }
  }
  fillRect(ctx, "#1b1b2e", 0, GROUND, WIDTH, HEIGHT - GROUND);
  fillRect(ctx, "#2e2e4a", 0, GROUND, WIDTH, 2);
  for (let x = 4; x < WIDTH; x += 16) fillRect(ctx, "#4a4a70", x + shake, GROUND + 12, 8, 2);
}

function poseOf(move: MoveId | null, phase: number): typeof ZORD_IDLE {
  if (!move) return ZORD_IDLE;
  if (move === "guard") return ZORD_GUARD;
  if (MOVES[move].kind === "attack" && phase > 0.2 && phase < 0.8) return ZORD_ATTACK;
  return ZORD_IDLE;
}

function lungeOf(move: MoveId | null, phase: number): number {
  if (!move || MOVES[move].kind !== "attack") return 0;
  const curve = phase < 0.35 ? -phase * 20 : phase < 0.65 ? (phase - 0.35) * 90 - 7 : (1 - phase) * 60;
  return Math.round(curve);
}

function drawZord(ctx: CanvasRenderingContext2D, battle: Battle, side: Side, action: Action | null, frame: number): void {
  const zord = battle[side];
  const palette = ZORD_PALETTES[zord.palette] ?? ZORD_PALETTES.crimson!;
  const facingRight = side === "left";
  const phase = action ? frame / ACTION_FRAMES : 0;
  const acting = action?.side === side;
  const struck = action !== null && action.side !== side && action.damage > 0 && phase > 0.45 && phase < 0.75;
  const bob = Math.round(Math.sin(frame / 9 + (side === "left" ? 0 : 2)) * 1.5);
  const lunge = acting ? lungeOf(action.move, phase) * (facingRight ? 1 : -1) : 0;
  const shake = struck ? (frame % 2 === 0 ? 3 : -3) : 0;
  const base = side === "left" ? LEFT_X : RIGHT_X;
  const x = base + lunge + shake;
  const y = GROUND - ZORD_IDLE.height * SCALE + bob;
  const pose = acting ? poseOf(action.move, phase) : zord.guarding ? ZORD_GUARD : ZORD_IDLE;

  if (battle.winner !== "none" && battle.winner !== side) {
    ctx.save();
    ctx.translate(x + 32, GROUND - 20);
    ctx.rotate(facingRight ? -1.4 : 1.4);
    drawSprite(ctx, ZORD_IDLE, palette, -32, -44, SCALE, { flip: !facingRight, alpha: 0.85 });
    ctx.restore();
    for (let puff = 0; puff < 6; puff += 1) {
      const offset = (frame + puff * 11) % 40;
      fillRect(ctx, puff % 2 === 0 ? "#6b6b7a" : "#3a3a4a", x + 10 + puff * 9, GROUND - 40 - offset, 5, 5);
    }
    return;
  }

  drawSprite(ctx, pose, palette, x, y, SCALE, { flip: !facingRight, ...(struck ? { tint: "#ffffff" } : {}) });
  if (zord.guarding && !acting) {
    const edge = facingRight ? x + 64 : x - 8;
    fillRect(ctx, frame % 8 < 4 ? "#3de0e0" : "#a0f4ff", edge, y + 20, 4, 52);
  }
}

function drawEffect(ctx: CanvasRenderingContext2D, battle: Battle, action: Action, frame: number): void {
  const phase = frame / ACTION_FRAMES;
  const attacker = action.side;
  const defender = opponentOf(attacker);
  const fromX = attacker === "left" ? LEFT_X + 60 : RIGHT_X + 4;
  const toX = defender === "left" ? LEFT_X + 40 : RIGHT_X + 24;
  const chest = GROUND - 56;

  if (action.move === "missiles") {
    for (let rocket = 0; rocket < 3; rocket += 1) {
      const travel = Math.min(1, Math.max(0, (phase - 0.25 - rocket * 0.06) / 0.3));
      if (travel <= 0 || travel >= 1) continue;
      const x = fromX + (toX - fromX) * travel;
      const y = chest + rocket * 9 - 9 + Math.sin(travel * 6) * 3;
      fillRect(ctx, "#e8e8f0", x, y, 6, 3);
      fillRect(ctx, frame % 2 === 0 ? "#ffe14d" : "#ff5ad0", x + (attacker === "left" ? -5 : 6), y, 5, 3);
    }
  }
  if (action.move === "saber" && phase > 0.35 && phase < 0.65) {
    const sweep = (phase - 0.35) / 0.3;
    for (let slice = 0; slice < 5; slice += 1) {
      const y = chest - 24 + slice * 12 + sweep * 10;
      fillRect(ctx, slice % 2 === 0 ? "#ffffff" : "#3de0e0", toX - 12 + slice * 3, y, 22 - slice * 2, 3);
    }
  }
  if (phase > 0.45 && phase < 0.68 && action.damage > 0) {
    const burst = (phase - 0.45) / 0.23;
    star(ctx, toX + 8, chest, 6 + burst * 14, burst < 0.5 ? "#ffe14d" : "#ff5ad0");
    if (action.blocked) star(ctx, toX + 8, chest, 8 + burst * 8, "#3de0e0");
  }
  if (action.move === "charge") {
    for (let spark = 0; spark < 5; spark += 1) {
      const rise = (frame * 2 + spark * 13) % 44;
      fillRect(ctx, spark % 2 === 0 ? "#3de0a0" : "#ffe14d", fromX - (attacker === "left" ? 30 : -30) + spark * 6, GROUND - 12 - rise, 3, 3);
    }
  }
  if (action.damage > 0 && phase > 0.5) {
    const float = (phase - 0.5) * 60;
    pixelText(ctx, `-${action.damage}${action.critical ? "!" : ""}`, toX, chest - 30 - float, action.critical ? "#ff5ad0" : "#ffffff", 14);
  }
}

export function renderBattle(ctx: CanvasRenderingContext2D, battle: Battle, action: Action | null, frame: number): void {
  const impact = action !== null && action.damage > 0 && frame / ACTION_FRAMES > 0.45 && frame / ACTION_FRAMES < 0.6;
  const shake = impact ? (frame % 2 === 0 ? 2 : -2) : 0;
  ctx.imageSmoothingEnabled = false;
  drawSky(ctx, battle.seed, shake);
  drawCity(ctx, battle.seed, shake);
  drawZord(ctx, battle, "left", action, frame);
  drawZord(ctx, battle, "right", action, frame);
  if (action) drawEffect(ctx, battle, action, frame);
  if (battle.winner !== "none") {
    fillRect(ctx, "#00000099", 0, 14, WIDTH, 32);
    pixelText(ctx, `${battle[battle.winner].name} STANDS`, 62, 22, frame % 20 < 10 ? "#ffe14d" : "#ffffff", 18);
  }
}
