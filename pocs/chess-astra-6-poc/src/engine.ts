import { Chess, type Move, type PieceSymbol } from 'chess.js';

export type Difficulty = 'apprentice' | 'wizard' | 'grandmaster';
export type SearchRequest = { history: string[]; difficulty: Difficulty };
export type SearchReply = { from: string; to: string; promotion?: string } | null;
const values: Record<PieceSymbol, number> = { p: 100, n: 320, b: 335, r: 500, q: 900, k: 0 };
const depths: Record<Difficulty, number> = { apprentice: 1, wizard: 2, grandmaster: 3 };

export function restoreGame(history: string[]): Chess {
  const game = new Chess();
  for (const move of history) game.move(move);
  return game;
}

function evaluate(game: Chess): number {
  let score = 0;
  for (const row of game.board()) {
    for (const piece of row) {
      if (!piece) continue;
      const file = piece.square.charCodeAt(0) - 97;
      const rank = Number(piece.square[1]) - 1;
      const center = 3.5 - (Math.abs(file - 3.5) + Math.abs(rank - 3.5)) / 2;
      const advance = piece.color === 'w' ? rank : 7 - rank;
      const position = piece.type === 'p' ? advance * 9 + center * 4 : piece.type === 'k' ? -center * 8 : center * 14;
      score += (values[piece.type] + position) * (piece.color === game.turn() ? 1 : -1);
    }
  }
  return score;
}

function ordered(moves: Move[]): Move[] {
  return moves.sort((a, b) => priority(b) - priority(a));
}

function priority(move: Move): number {
  return (move.captured ? values[move.captured] * 10 - values[move.piece] : 0) + (move.promotion ? values[move.promotion] : 0) + (move.san.includes('+') ? 40 : 0);
}

export function chooseMove(game: Chess, difficulty: Difficulty, budget = 1400): SearchReply {
  if (game.isGameOver()) return null;
  const deadline = performance.now() + budget;
  const search = (depth: number, alpha: number, beta: number, ply: number): number => {
    if (game.isCheckmate()) return -100000 + ply;
    if (game.isDraw()) return 0;
    if (depth === 0 || performance.now() > deadline) return evaluate(game);
    let best = -Infinity;
    for (const move of ordered(game.moves({ verbose: true }))) {
      game.move(move);
      const score = -search(depth - 1, -beta, -alpha, ply + 1);
      game.undo();
      best = Math.max(best, score);
      alpha = Math.max(alpha, score);
      if (alpha >= beta) break;
    }
    return best;
  };
  const moves = ordered(game.moves({ verbose: true }));
  let best = moves[0];
  for (let depth = 1; depth <= depths[difficulty]; depth++) {
    let roundBest = best;
    let score = -Infinity;
    let complete = true;
    for (const move of moves) {
      if (performance.now() > deadline) { complete = false; break; }
      game.move(move);
      const value = -search(depth - 1, -Infinity, -score, 1);
      game.undo();
      if (value > score) { score = value; roundBest = move; }
    }
    if (complete) best = roundBest;
    else break;
  }
  return { from: best.from, to: best.to, promotion: best.promotion };
}
