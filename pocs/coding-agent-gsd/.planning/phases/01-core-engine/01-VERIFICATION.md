# Phase 1 Verification: Core Engine

## Phase Goal

Playable Tetris — pieces fall, player controls them, lines clear, game ends when stacked.

## Status: passed

## Must-Haves Verification

| Must-Have | Evidence | Status |
|-----------|----------|--------|
| Canvas renders at correct size | index.html loads canvas, render.js sets 300x600 with high-DPI scaling | ✓ Verified |
| 7 tetrominoes with correct shapes | pieces.js defines I,O,T,S,Z,J,L with 4 rotations each | ✓ Verified |
| Pieces fall automatically | main.js dropInterval = 1000, dropCounter in update() | ✓ Verified |
| Player can move pieces | input.js handles arrow keys, main.js movePiece() | ✓ Verified |
| Player can rotate with wall kicks | pieces.js WALL_KICKS, main.js rotatePiece() tries 5 offsets | ✓ Verified |
| Player can hard drop | main.js hardDrop() moves until collision then locks | ✓ Verified |
| Pieces lock on landing | main.js lockDelay = 500, lockPieceToBoard() calls board.js lockPiece() | ✓ Verified |
| Complete rows clear | board.js checkLines() finds full rows, clearLines() removes and shifts | ✓ Verified |
| Game ends when stacked | main.js spawnPiece() checks isValidPosition, sets gameOver = true | ✓ Verified |
| Game can restart | main.js resetGame() on 'KeyR' when gameOver | ✓ Verified |
| 60fps with rAF | main.js uses requestAnimationFrame(gameLoop) | ✓ Verified |
| No external dependencies | Only vanilla JS files, no npm packages | ✓ Verified |
| High-DPI support | render.js uses devicePixelRatio for canvas scaling | ✓ Verified |

## Requirements Coverage

| Requirement | Status |
|-------------|--------|
| CORE-01 | ✓ Complete |
| CORE-02 | ✓ Complete |
| CORE-03 | ✓ Complete |
| CORE-04 | ✓ Complete |
| CORE-05 | ✓ Complete |
| CORE-06 | ✓ Complete |
| CORE-07 | ✓ Complete |
| CORE-08 | ✓ Complete |
| CORE-09 | ✓ Complete |
| CORE-10 | ✓ Complete |
| CORE-11 | ✓ Complete |
| TECH-01 | ✓ Complete |
| TECH-02 | ✓ Complete |
| TECH-04 | ✓ Complete |
| TECH-05 | ✓ Complete |

**Coverage:** 15/15 requirements verified (100%)

## Human Verification Checklist

The following should be manually tested in a browser:

- [ ] Open index.html in browser
- [ ] Grid displays correctly (10 columns, 20 rows)
- [ ] Pieces spawn at top and fall
- [ ] Arrow keys move piece left/right
- [ ] Down arrow soft drops
- [ ] Spacebar hard drops
- [ ] Up arrow rotates
- [ ] Pieces lock after landing
- [ ] Full rows clear with white flash
- [ ] Game over screen appears when stacked
- [ ] R key restarts the game

## Gaps Found

None.

---
*Verified: 2026-02-02*
