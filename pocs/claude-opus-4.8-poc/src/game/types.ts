export type Vec3 = { x: number; y: number; z: number };

export type Axis = "x" | "y" | "z";

export type Piece = {
  cells: Vec3[];
  pos: Vec3;
  color: number;
};

export type Status = "idle" | "playing" | "paused" | "over";

export type GameState = {
  board: number[];
  piece: Piece | null;
  nextShape: number;
  score: number;
  level: number;
  cleared: number;
  status: Status;
};

export const WIDTH = 5;
export const DEPTH = 5;
export const HEIGHT = 12;

export const cellIndex = (x: number, y: number, z: number): number =>
  x + z * WIDTH + y * WIDTH * DEPTH;
