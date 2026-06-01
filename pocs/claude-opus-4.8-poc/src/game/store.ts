import { Store } from "@tanstack/store";
import type { Axis, GameState, Piece, Vec3 } from "./types";
import { WIDTH, DEPTH, HEIGHT, cellIndex } from "./types";
import { SHAPES } from "./pieces";

const LINE_SCORES = [0, 100, 300, 600, 1000];
const LEVEL_STEP = 8;

const emptyBoard = (): number[] => new Array(WIDTH * DEPTH * HEIGHT).fill(0);

const randomShape = (): number => Math.floor(Math.random() * SHAPES.length);

const rotateCell = (c: Vec3, axis: Axis): Vec3 => {
  if (axis === "x") return { x: c.x, y: -c.z, z: c.y };
  if (axis === "y") return { x: c.z, y: c.y, z: -c.x };
  return { x: -c.y, y: c.x, z: c.z };
};

const normalize = (cells: Vec3[]): Vec3[] => {
  const minX = Math.min(...cells.map((c) => c.x));
  const minY = Math.min(...cells.map((c) => c.y));
  const minZ = Math.min(...cells.map((c) => c.z));
  return cells.map((c) => ({ x: c.x - minX, y: c.y - minY, z: c.z - minZ }));
};

const sizeOf = (cells: Vec3[], key: keyof Vec3): number =>
  Math.max(...cells.map((c) => c[key])) + 1;

const canPlace = (board: number[], cells: Vec3[], pos: Vec3): boolean => {
  for (const c of cells) {
    const x = pos.x + c.x;
    const y = pos.y + c.y;
    const z = pos.z + c.z;
    if (x < 0 || x >= WIDTH || z < 0 || z >= DEPTH || y < 0 || y >= HEIGHT) {
      return false;
    }
    if (board[cellIndex(x, y, z)] !== 0) return false;
  }
  return true;
};

const makePiece = (shapeIndex: number): Piece => {
  const shape = SHAPES[shapeIndex];
  const cells = normalize(shape.cells);
  const pos: Vec3 = {
    x: Math.floor((WIDTH - sizeOf(cells, "x")) / 2),
    y: HEIGHT - sizeOf(cells, "y"),
    z: Math.floor((DEPTH - sizeOf(cells, "z")) / 2),
  };
  return { cells, pos, color: shape.color };
};

const scoreFor = (n: number, level: number): number => {
  const base = LINE_SCORES[n] ?? 1000 + (n - 4) * 500;
  return base * level;
};

const clearLayers = (board: number[]): { board: number[]; cleared: number } => {
  const kept: number[][] = [];
  for (let y = 0; y < HEIGHT; y++) {
    const layer: number[] = [];
    let full = true;
    for (let z = 0; z < DEPTH; z++) {
      for (let x = 0; x < WIDTH; x++) {
        const val = board[cellIndex(x, y, z)];
        layer.push(val);
        if (val === 0) full = false;
      }
    }
    if (!full) kept.push(layer);
  }
  const next = emptyBoard();
  for (let y = 0; y < kept.length; y++) {
    const layer = kept[y];
    for (let i = 0; i < layer.length; i++) {
      const x = i % WIDTH;
      const z = Math.floor(i / WIDTH);
      next[cellIndex(x, y, z)] = layer[i];
    }
  }
  return { board: next, cleared: HEIGHT - kept.length };
};

export const ghostPos = (state: GameState): Vec3 | null => {
  if (!state.piece) return null;
  let pos = state.piece.pos;
  while (canPlace(state.board, state.piece.cells, { ...pos, y: pos.y - 1 })) {
    pos = { ...pos, y: pos.y - 1 };
  }
  return pos;
};

export const store = new Store<GameState>({
  board: emptyBoard(),
  piece: null,
  nextShape: randomShape(),
  score: 0,
  level: 1,
  cleared: 0,
  status: "idle",
});

export const start = (): void => {
  const shape = store.state.nextShape;
  const piece = makePiece(shape);
  store.setState((s) => ({
    ...s,
    board: emptyBoard(),
    piece,
    nextShape: randomShape(),
    score: 0,
    level: 1,
    cleared: 0,
    status: "playing",
  }));
};

export const pause = (): void => {
  store.setState((s) => {
    if (s.status === "playing") return { ...s, status: "paused" };
    if (s.status === "paused") return { ...s, status: "playing" };
    return s;
  });
};

export const move = (dx: number, dy: number, dz: number): void => {
  const s = store.state;
  if (s.status !== "playing" || !s.piece) return;
  const pos = { x: s.piece.pos.x + dx, y: s.piece.pos.y + dy, z: s.piece.pos.z + dz };
  if (canPlace(s.board, s.piece.cells, pos)) {
    store.setState((prev) => (prev.piece ? { ...prev, piece: { ...prev.piece, pos } } : prev));
  }
};

export const rotate = (axis: Axis, dir: 1 | -1 = 1): void => {
  const s = store.state;
  if (s.status !== "playing" || !s.piece) return;
  const times = dir === 1 ? 1 : 3;
  let cells = s.piece.cells;
  for (let i = 0; i < times; i++) cells = cells.map((c) => rotateCell(c, axis));
  cells = normalize(cells);
  if (canPlace(s.board, cells, s.piece.pos)) {
    store.setState((prev) => (prev.piece ? { ...prev, piece: { ...prev.piece, cells } } : prev));
  }
};

const lockAndSpawn = (): void => {
  const s = store.state;
  if (!s.piece) return;
  const board = s.board.slice();
  for (const c of s.piece.cells) {
    board[cellIndex(s.piece.pos.x + c.x, s.piece.pos.y + c.y, s.piece.pos.z + c.z)] = s.piece.color;
  }
  const result = clearLayers(board);
  const totalCleared = s.cleared + result.cleared;
  const level = 1 + Math.floor(totalCleared / LEVEL_STEP);
  const gained = scoreFor(result.cleared, s.level);
  const piece = makePiece(s.nextShape);
  if (!canPlace(result.board, piece.cells, piece.pos)) {
    store.setState((prev) => ({
      ...prev,
      board: result.board,
      piece: null,
      status: "over",
      score: prev.score + gained,
      cleared: totalCleared,
      level,
    }));
    return;
  }
  store.setState((prev) => ({
    ...prev,
    board: result.board,
    piece,
    nextShape: randomShape(),
    score: prev.score + gained,
    cleared: totalCleared,
    level,
  }));
};

export const tick = (): void => {
  const s = store.state;
  if (s.status !== "playing" || !s.piece) return;
  const pos = { ...s.piece.pos, y: s.piece.pos.y - 1 };
  if (canPlace(s.board, s.piece.cells, pos)) {
    store.setState((prev) => (prev.piece ? { ...prev, piece: { ...prev.piece, pos } } : prev));
  } else {
    lockAndSpawn();
  }
};

export const drop = (): void => {
  const s = store.state;
  if (s.status !== "playing" || !s.piece) return;
  const pos = ghostPos(s);
  if (!pos) return;
  store.setState((prev) => (prev.piece ? { ...prev, piece: { ...prev.piece, pos } } : prev));
  lockAndSpawn();
};

export const intervalFor = (level: number): number => Math.max(120, 800 - (level - 1) * 70);
