import test from 'node:test';
import assert from 'node:assert/strict';
import { spots, createGame, startGame, jump, performTrick, togglePause, updateGame } from '../js/game.js';

function advance(game, seconds) {
  for (let elapsed = 0; elapsed < seconds; elapsed += 1 / 60) updateGame(game, 1 / 60);
}
test('four distinct waterfront spots support either rider', () => {
  assert.equal(new Set(spots.map(spot => spot.id)).size, 4);
  for (const spot of spots) for (const rider of ['boy', 'girl']) {
    const game = createGame(spot, rider);
    assert.equal(game.spot, spot);
    assert.equal(game.rider, rider);
    assert.equal(game.status, 'ready');
  }
});
test('a jump cannot repeat in midair and awards points only on landing', () => {
  const game = createGame();
  assert.equal(jump(game), false);
  startGame(game);
  assert.equal(jump(game), true);
  assert.equal(jump(game), false);
  advance(game, .2);
  assert.ok(game.y > 30);
  assert.equal(game.score, 0);
  advance(game, 1);
  assert.equal(game.y, 0);
  assert.equal(game.score, 50);
});
test('different airborne tricks build a combo and duplicates cannot farm points', () => {
  const game = createGame();
  startGame(game);
  assert.equal(performTrick(game, 'kickflip'), false);
  jump(game);
  advance(game, .2);
  assert.equal(performTrick(game, 'kickflip'), true);
  assert.equal(performTrick(game, 'kickflip'), false);
  assert.equal(performTrick(game, 'heelflip'), true);
  assert.equal(performTrick(game, 'spin'), true);
  assert.equal(game.score, 0);
  advance(game, 1);
  assert.equal(game.score, 1950);
  assert.equal(game.landed, 1);
});
test('a collision loses the combo and applies one penalty per obstacle', () => {
  const game = createGame();
  startGame(game);
  game.score = 200;
  game.tricks = ['spin'];
  game.obstacles.push({ x: 290, width: 35, height: 43, passed: false, type: 'cone' });
  advance(game, .1);
  assert.equal(game.score, 100);
  assert.equal(game.bails, 1);
  assert.deepEqual(game.tricks, []);
});
test('clearing an obstacle awards points without a collision', () => {
  const game = createGame();
  startGame(game);
  jump(game);
  advance(game, .2);
  game.obstacles.push({ x: 295, width: 35, height: 43, passed: false, type: 'cone' });
  advance(game, .3);
  assert.equal(game.bails, 0);
  assert.equal(game.score, 75);
});
test('pause freezes physics and time and resume continues the run', () => {
  const game = createGame();
  startGame(game);
  jump(game);
  togglePause(game);
  const snapshot = { time: game.time, y: game.y, distance: game.distance };
  advance(game, 3);
  assert.deepEqual({ time: game.time, y: game.y, distance: game.distance }, snapshot);
  togglePause(game);
  advance(game, .1);
  assert.ok(game.time < snapshot.time);
});
test('a session ends at sixty seconds and cannot score afterward', () => {
  const game = createGame();
  startGame(game);
  advance(game, 61);
  assert.equal(game.status, 'finished');
  assert.equal(game.time, 0);
  const score = game.score;
  assert.equal(jump(game), false);
  advance(game, 1);
  assert.equal(game.score, score);
});
