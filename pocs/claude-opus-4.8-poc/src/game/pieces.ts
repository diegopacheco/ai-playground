import type { Vec3 } from "./types";

export type Shape = {
  cells: Vec3[];
  color: number;
};

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

export const SHAPES: Shape[] = [
  { color: 1, cells: [v(0, 0, 0), v(1, 0, 0), v(2, 0, 0), v(3, 0, 0)] },
  { color: 2, cells: [v(0, 0, 0), v(1, 0, 0), v(0, 0, 1), v(1, 0, 1)] },
  { color: 3, cells: [v(0, 0, 0), v(1, 0, 0), v(2, 0, 0), v(0, 1, 0)] },
  { color: 4, cells: [v(0, 0, 0), v(1, 0, 0), v(2, 0, 0), v(1, 1, 0)] },
  { color: 5, cells: [v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(2, 0, 1)] },
  { color: 6, cells: [v(0, 0, 0), v(1, 0, 0), v(0, 0, 1), v(0, 1, 0)] },
  { color: 7, cells: [v(0, 0, 0), v(1, 0, 0), v(0, 1, 0), v(0, 0, 1)] },
];

export const COLORS: string[] = [
  "#000000",
  "#22d3ee",
  "#facc15",
  "#a855f7",
  "#34d399",
  "#fb7185",
  "#60a5fa",
  "#f97316",
];

export const colorOf = (index: number): string => COLORS[index] ?? "#ffffff";
