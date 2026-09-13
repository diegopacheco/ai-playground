import { describe, expect, test } from 'bun:test';
import { Chess } from 'chess.js';
import { chooseMove, restoreGame, skipStuckTurn, type Difficulty } from '../src/engine';

describe('Guardian search', () => {
  for (const difficulty of ['apprentice', 'wizard', 'grandmaster'] as Difficulty[]) {
    test(`${difficulty} chooses a legal reply without changing the match`, () => {
      const game = restoreGame(['e4']);
      const fen = game.fen();
      const history = game.history();
      const move = chooseMove(game, difficulty);
      expect(game.fen()).toBe(fen);
      expect(game.history()).toEqual(history);
      expect(move).not.toBeNull();
      expect(() => game.move(move!)).not.toThrow();
      expect(game.turn()).toBe('w');
    });
  }
  test('takes a forced checkmate instead of material', () => {
    const game = restoreGame(['f3', 'e5', 'g4']);
    game.move(chooseMove(game, 'wizard')!);
    expect(game.isCheckmate()).toBe(true);
  });
  test('does not move when a match has ended', () => {
    const game = restoreGame(['f3', 'e5', 'g4', 'Qh4#']);
    expect(chooseMove(game, 'grandmaster')).toBeNull();
    expect(chooseMove(new Chess('8/8/8/8/8/8/2k5/K7 w - - 0 1'), 'wizard')).toBeNull();
  });
  test('a short budget still returns a legal move and restores state', () => {
    const game = new Chess();
    const fen = game.fen();
    const move = chooseMove(game, 'grandmaster', 0);
    expect(game.fen()).toBe(fen);
    expect(() => game.move(move!)).not.toThrow();
  });
});

describe('rules and persisted match', () => {
  test('saved history preserves repetition and undo', () => {
    const history = ['Nf3', 'Nf6', 'Ng1', 'Ng8', 'Nf3', 'Nf6', 'Ng1', 'Ng8'];
    const game = restoreGame(history);
    expect(game.isThreefoldRepetition()).toBe(true);
    game.undo();
    expect(game.isThreefoldRepetition()).toBe(false);
  });
  test('castling moves king and rook together', () => {
    const game = restoreGame(['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6', 'O-O']);
    expect(game.get('g1')?.type).toBe('k');
    expect(game.get('f1')?.type).toBe('r');
  });
  test('en passant removes the captured pawn from its actual square', () => {
    const game = restoreGame(['e4', 'a6', 'e5', 'd5']);
    const move = game.move('exd6');
    expect(move.isEnPassant()).toBe(true);
    expect(game.get('d5')).toBeUndefined();
    expect(game.get('d6')?.color).toBe('w');
  });
  test('promotion permits a knight and rejects an illegal king move', () => {
    const game = new Chess('7k/P7/8/8/8/8/8/7K w - - 0 1');
    game.move({ from: 'a7', to: 'a8', promotion: 'n' });
    expect(game.get('a8')?.type).toBe('n');
    expect(() => new Chess().move('e1e3')).toThrow();
  });
  test('corrupt saved moves fail loudly', () => {
    expect(() => restoreGame(['e4', 'invalid'])).toThrow();
  });
});

describe('no stalemate: a stuck side skips its turn', () => {
  const beforeStalemate = 'k7/2B5/2K5/6p1/6N1/pP6/P1P5/8 w - - 0 59';

  test('a side with no legal move and no check passes, so a winning player keeps playing instead of drawing', () => {
    const game = new Chess(beforeStalemate);
    game.move('Kb6');
    expect(skipStuckTurn(game)).toBe(true);
    expect(game.turn()).toBe('w');
    expect(game.isGameOver()).toBe(false);
    expect(game.history()).toEqual(['Kb6', '--']);
    game.move('Kb5');
    expect(game.moves().length).toBeGreaterThan(0);
  });

  test('a side that can move or is in check never skips', () => {
    const open = restoreGame(['e4']);
    expect(skipStuckTurn(open)).toBe(false);
    expect(open.history()).toEqual(['e4']);
    const checked = new Chess('4k3/8/8/8/8/8/8/4R2K b - - 0 1');
    expect(skipStuckTurn(checked)).toBe(false);
  });

  test('skips survive a save and reload because they replay from the history', () => {
    const played = restoreGame(['e3', 'a5', 'Qh5', 'Ra6', 'Qxa5', 'h5', 'h4', 'Rah6', 'Qxc7', 'f6', 'Qxd7+', 'Kf7', 'Qxb7', 'Qd3', 'Qxb8', 'Qh7', 'Qxc8', 'Kg6', 'Qe6']);
    expect(skipStuckTurn(played)).toBe(true);
    played.move('Ke2');
    const restored = restoreGame(played.history());
    expect(restored.fen()).toBe(played.fen());
    expect(restored.history().filter(move => move === '--')).toHaveLength(played.history().filter(move => move === '--').length);
  });

  test('the Guardian does not score a stuck opponent as a draw when it is losing', () => {
    const game = new Chess('k7/2B5/2K5/6p1/6N1/pP6/P1P5/8 w - - 0 59');
    const fen = game.fen();
    const move = chooseMove(game, 'grandmaster');
    expect(game.fen()).toBe(fen);
    expect(() => game.move(move!)).not.toThrow();
    const stuck = new Chess('k7/2B5/1K6/6p1/6N1/pP6/P1P5/8 b - - 0 60');
    expect(chooseMove(stuck, 'wizard')).toBeNull();
  });
});
