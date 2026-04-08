import { useState, useCallback, useEffect, useRef } from "react";
import {
  Board, Piece, Position, PIECES, BOARD_WIDTH, BOARD_HEIGHT,
  LEVEL_SPEEDS, LINES_PER_LEVEL, DIFFICULTY_MULTIPLIER,
  Difficulty, ThemeName, THEMES,
} from "./types";

function createBoard(): Board {
  return Array.from({ length: BOARD_HEIGHT }, () => Array(BOARD_WIDTH).fill(null));
}

function randomPiece(theme: ThemeName): Piece {
  const idx = Math.floor(Math.random() * PIECES.length);
  return { shape: PIECES[idx], color: THEMES[theme].pieces[idx] };
}

function rotate(shape: number[][]): number[][] {
  const rows = shape.length;
  const cols = shape[0].length;
  const rotated: number[][] = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      rotated[c][rows - 1 - r] = shape[r][c];
    }
  }
  return rotated;
}

function collides(board: Board, piece: Piece, pos: Position): boolean {
  for (let r = 0; r < piece.shape.length; r++) {
    for (let c = 0; c < piece.shape[r].length; c++) {
      if (!piece.shape[r][c]) continue;
      const nx = pos.x + c;
      const ny = pos.y + r;
      if (nx < 0 || nx >= BOARD_WIDTH || ny >= BOARD_HEIGHT) return true;
      if (ny >= 0 && board[ny][nx]) return true;
    }
  }
  return false;
}

function merge(board: Board, piece: Piece, pos: Position): Board {
  const newBoard = board.map((row) => [...row]);
  for (let r = 0; r < piece.shape.length; r++) {
    for (let c = 0; c < piece.shape[r].length; c++) {
      if (!piece.shape[r][c]) continue;
      const ny = pos.y + r;
      const nx = pos.x + c;
      if (ny >= 0 && ny < BOARD_HEIGHT && nx >= 0 && nx < BOARD_WIDTH) {
        newBoard[ny][nx] = piece.color;
      }
    }
  }
  return newBoard;
}

function clearLines(board: Board): { board: Board; cleared: number } {
  const remaining = board.filter((row) => row.some((cell) => !cell));
  const cleared = BOARD_HEIGHT - remaining.length;
  const emptyRows = Array.from({ length: cleared }, () => Array(BOARD_WIDTH).fill(null));
  return { board: [...emptyRows, ...remaining], cleared };
}

function calcScore(linesCleared: number, level: number): number {
  const base = [0, 100, 300, 500, 800];
  return (base[linesCleared] || 0) * (level + 1);
}

export interface GameState {
  board: Board;
  score: number;
  level: number;
  linesCleared: number;
  gameOver: boolean;
  playing: boolean;
  currentPiece: Piece | null;
  currentPos: Position;
  nextPiece: Piece | null;
  timerSeconds: number;
}

export function useGame(difficulty: Difficulty, theme: ThemeName, timerEnabled: boolean, timerMinutes: number) {
  const [state, setState] = useState<GameState>({
    board: createBoard(),
    score: 0,
    level: 0,
    linesCleared: 0,
    gameOver: false,
    playing: false,
    currentPiece: null,
    currentPos: { x: 0, y: 0 },
    nextPiece: null,
    timerSeconds: timerMinutes * 60,
  });

  const stateRef = useRef(state);
  stateRef.current = state;
  const dropInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  const spawnPiece = useCallback((board: Board, next: Piece | null): Partial<GameState> | null => {
    const piece = next || randomPiece(theme);
    const nextP = randomPiece(theme);
    const pos: Position = { x: Math.floor((BOARD_WIDTH - piece.shape[0].length) / 2), y: -1 };
    if (collides(board, piece, { ...pos, y: 0 })) {
      return { currentPiece: piece, currentPos: pos, nextPiece: nextP, gameOver: true, playing: false };
    }
    return { currentPiece: piece, currentPos: pos, nextPiece: nextP };
  }, [theme]);

  const startGame = useCallback(() => {
    const board = createBoard();
    const spawned = spawnPiece(board, null);
    if (!spawned) return;
    setState({
      board,
      score: 0,
      level: 0,
      linesCleared: 0,
      gameOver: false,
      playing: true,
      currentPiece: spawned.currentPiece!,
      currentPos: spawned.currentPos!,
      nextPiece: spawned.nextPiece!,
      timerSeconds: timerMinutes * 60,
    });
  }, [spawnPiece, timerMinutes]);

  const drop = useCallback(() => {
    setState((prev) => {
      if (!prev.playing || prev.gameOver || !prev.currentPiece) return prev;
      const newPos = { ...prev.currentPos, y: prev.currentPos.y + 1 };
      if (!collides(prev.board, prev.currentPiece, newPos)) {
        return { ...prev, currentPos: newPos };
      }
      const merged = merge(prev.board, prev.currentPiece, prev.currentPos);
      const { board: clearedBoard, cleared } = clearLines(merged);
      const totalLines = prev.linesCleared + cleared;
      const newLevel = Math.min(Math.floor(totalLines / LINES_PER_LEVEL), 9);
      const points = calcScore(cleared, prev.level);
      const diffMult = DIFFICULTY_MULTIPLIER[difficulty];
      const newScore = prev.score + Math.floor(points * diffMult);
      const spawned = spawnPiece(clearedBoard, prev.nextPiece);
      if (!spawned) return { ...prev, gameOver: true, playing: false };
      return {
        ...prev,
        board: clearedBoard,
        score: newScore,
        level: newLevel,
        linesCleared: totalLines,
        ...spawned,
      };
    });
  }, [difficulty, spawnPiece]);

  const moveLeft = useCallback(() => {
    setState((prev) => {
      if (!prev.playing || !prev.currentPiece) return prev;
      const newPos = { ...prev.currentPos, x: prev.currentPos.x - 1 };
      if (collides(prev.board, prev.currentPiece, newPos)) return prev;
      return { ...prev, currentPos: newPos };
    });
  }, []);

  const moveRight = useCallback(() => {
    setState((prev) => {
      if (!prev.playing || !prev.currentPiece) return prev;
      const newPos = { ...prev.currentPos, x: prev.currentPos.x + 1 };
      if (collides(prev.board, prev.currentPiece, newPos)) return prev;
      return { ...prev, currentPos: newPos };
    });
  }, []);

  const rotatePiece = useCallback(() => {
    setState((prev) => {
      if (!prev.playing || !prev.currentPiece) return prev;
      const rotated = { ...prev.currentPiece, shape: rotate(prev.currentPiece.shape) };
      if (collides(prev.board, rotated, prev.currentPos)) return prev;
      return { ...prev, currentPiece: rotated };
    });
  }, []);

  const hardDrop = useCallback(() => {
    setState((prev) => {
      if (!prev.playing || !prev.currentPiece) return prev;
      let pos = { ...prev.currentPos };
      while (!collides(prev.board, prev.currentPiece, { ...pos, y: pos.y + 1 })) {
        pos.y += 1;
      }
      const merged = merge(prev.board, prev.currentPiece, pos);
      const { board: clearedBoard, cleared } = clearLines(merged);
      const totalLines = prev.linesCleared + cleared;
      const newLevel = Math.min(Math.floor(totalLines / LINES_PER_LEVEL), 9);
      const points = calcScore(cleared, prev.level);
      const diffMult = DIFFICULTY_MULTIPLIER[difficulty];
      const newScore = prev.score + Math.floor(points * diffMult);
      const spawned = spawnPiece(clearedBoard, prev.nextPiece);
      if (!spawned) return { ...prev, board: clearedBoard, score: newScore, gameOver: true, playing: false };
      return {
        ...prev,
        board: clearedBoard,
        score: newScore,
        level: newLevel,
        linesCleared: totalLines,
        ...spawned,
      };
    });
  }, [difficulty, spawnPiece]);

  useEffect(() => {
    if (!state.playing) {
      if (dropInterval.current) clearInterval(dropInterval.current);
      if (timerInterval.current) clearInterval(timerInterval.current);
      return;
    }
    const speed = LEVEL_SPEEDS[state.level] || 100;
    if (dropInterval.current) clearInterval(dropInterval.current);
    dropInterval.current = setInterval(drop, speed);

    if (timerEnabled) {
      if (timerInterval.current) clearInterval(timerInterval.current);
      timerInterval.current = setInterval(() => {
        setState((prev) => {
          if (prev.timerSeconds <= 1) {
            return { ...prev, timerSeconds: 0, gameOver: true, playing: false };
          }
          return { ...prev, timerSeconds: prev.timerSeconds - 1 };
        });
      }, 1000);
    }

    return () => {
      if (dropInterval.current) clearInterval(dropInterval.current);
      if (timerInterval.current) clearInterval(timerInterval.current);
    };
  }, [state.playing, state.level, drop, timerEnabled]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!stateRef.current.playing) return;
      switch (e.key) {
        case "ArrowLeft": moveLeft(); break;
        case "ArrowRight": moveRight(); break;
        case "ArrowDown": drop(); break;
        case "ArrowUp": rotatePiece(); break;
        case " ": hardDrop(); break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [moveLeft, moveRight, drop, rotatePiece, hardDrop]);

  return { state, startGame, drop, moveLeft, moveRight, rotatePiece, hardDrop };
}
