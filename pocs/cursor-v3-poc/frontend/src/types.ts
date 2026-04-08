export type Difficulty = "easy" | "medium" | "hard";
export type ThemeName = "classic" | "neon" | "pastel" | "dark" | "ocean";

export interface GameConfig {
  difficulty: Difficulty;
  theme: ThemeName;
  timer_enabled: boolean;
  timer_minutes: number;
}

export interface ScoreEntry {
  id: string;
  player_name: string;
  score: number;
  level: number;
  lines_cleared: number;
  difficulty: string;
  created_at: string;
}

export interface Theme {
  background: string;
  boardBg: string;
  gridLine: string;
  text: string;
  accent: string;
  pieces: string[];
}

export const THEMES: Record<ThemeName, Theme> = {
  classic: {
    background: "#1a1a2e",
    boardBg: "#16213e",
    gridLine: "#1a2744",
    text: "#e0e0e0",
    accent: "#e94560",
    pieces: ["#00f0f0", "#f0f000", "#a000f0", "#00f000", "#f00000", "#0000f0", "#f0a000"],
  },
  neon: {
    background: "#0a0a0a",
    boardBg: "#111111",
    gridLine: "#1a1a1a",
    text: "#ffffff",
    accent: "#ff00ff",
    pieces: ["#00ffff", "#ffff00", "#ff00ff", "#00ff00", "#ff0044", "#4444ff", "#ff8800"],
  },
  pastel: {
    background: "#f5e6d3",
    boardBg: "#faf0e6",
    gridLine: "#e8d5c4",
    text: "#4a4a4a",
    accent: "#e88d97",
    pieces: ["#a8d8ea", "#f6e58d", "#dda0dd", "#98d8c8", "#f7a8a8", "#87ceeb", "#ffcc99"],
  },
  dark: {
    background: "#0d1117",
    boardBg: "#161b22",
    gridLine: "#21262d",
    text: "#c9d1d9",
    accent: "#58a6ff",
    pieces: ["#79c0ff", "#e3b341", "#bc8cff", "#56d364", "#f85149", "#6e7681", "#f0883e"],
  },
  ocean: {
    background: "#0c2233",
    boardBg: "#0f3460",
    gridLine: "#164677",
    text: "#e0f0ff",
    accent: "#00d2ff",
    pieces: ["#00d2ff", "#48c9b0", "#5dade2", "#76d7c4", "#2980b9", "#1abc9c", "#3498db"],
  },
};

export const BOARD_WIDTH = 10;
export const BOARD_HEIGHT = 20;
export const CELL_SIZE = 30;

export type Board = (string | null)[][];

export interface Position {
  x: number;
  y: number;
}

export interface Piece {
  shape: number[][];
  color: string;
}

export const PIECES: number[][][] = [
  [[1, 1, 1, 1]],
  [[1, 1], [1, 1]],
  [[0, 1, 0], [1, 1, 1]],
  [[1, 0, 0], [1, 1, 1]],
  [[0, 0, 1], [1, 1, 1]],
  [[0, 1, 1], [1, 1, 0]],
  [[1, 1, 0], [0, 1, 1]],
];

export const LEVEL_SPEEDS = [800, 720, 630, 540, 450, 370, 290, 220, 160, 100];

export const DIFFICULTY_MULTIPLIER: Record<Difficulty, number> = {
  easy: 0.7,
  medium: 1.0,
  hard: 1.5,
};

export const LINES_PER_LEVEL = 10;
