# Summary: 01-02 Tetrominoes & Movement

## Status: Complete

## What Was Built

- **js/pieces.js** — All 7 tetrominoes (I, O, T, S, Z, J, L) with 4 rotation states each, SRS wall kick data
- **js/board.js** — Added isValidPosition() for collision detection
- **js/input.js** — Keyboard handling with DAS (170ms initial delay, 50ms repeat)
- **js/render.js** — Added drawPiece() function
- **js/main.js** — 7-bag randomizer, piece spawning, movement, rotation with wall kicks, hard drop
- **index.html** — Updated script order to include pieces.js and input.js

## Commits

| Task | Commit | Files |
|------|--------|-------|
| Task 1 | 3b04b712 | js/pieces.js |
| Task 2 | 335e6734 | js/board.js, eslint.config.js |
| Task 3 | 435f33be | js/input.js |
| Task 4 | 8b880ea5 | js/render.js |
| Task 5-6 | cd80ddc9 | js/main.js, index.html |

## Requirements Addressed

- [x] CORE-02: 7 standard tetrominoes spawn (I, O, T, S, Z, J, L)
- [x] CORE-04: Player can move piece left/right with arrow keys
- [x] CORE-05: Player can soft drop with down arrow (framework ready, timing in Wave 3)
- [x] CORE-06: Player can hard drop with spacebar
- [x] CORE-07: Player can rotate piece clockwise with up arrow
- [x] CORE-08: Rotation uses wall kicks when near edges

## Verification

- [x] All 7 piece types defined with correct shapes and colors
- [x] Left/right arrows move piece horizontally
- [x] Spacebar instantly drops piece to bottom
- [x] Up arrow rotates piece clockwise
- [x] Wall kicks work near walls
- [x] Pieces cannot move through walls or floor

## Deviations

- Added eslint.config.js to fix ESLint hook errors
