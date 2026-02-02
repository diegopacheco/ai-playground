# Phase 1 Research: Core Engine

## What This Phase Builds

A playable Tetris game: canvas rendering, 7 tetrominoes, piece movement/rotation, collision detection, line clearing, and game over detection. No scoring UI yet (Phase 2), no themes (Phase 3), no freeze/growth (Phase 4).

## Key Implementation Decisions

### Canvas Setup

**Approach:** Single canvas, fixed initial size (10 cols × 20 rows), cell size 30px.

```
Canvas: 300px × 600px (10 × 30, 20 × 30)
```

**High-DPI handling:**
- Set canvas width/height to logical size × devicePixelRatio
- Scale context by devicePixelRatio
- Set CSS size to logical size

### Tetromino Representation

**7-bag randomizer:** Shuffle all 7 pieces, deal in order, repeat. Prevents long droughts.

**Piece data structure:**
```javascript
{
  type: 'T',
  x: 4,           // grid position
  y: 0,
  rotation: 0,    // 0-3
  shape: [...]    // 2D array of cells
}
```

**Rotation states:** Store all 4 rotations per piece. SRS (Super Rotation System) standard.

### Wall Kicks

When rotation blocked, try 5 offset positions (SRS wall kick data). If all fail, rotation fails.

**I-piece has unique kicks** (different from J/L/S/T/Z).

### Collision Detection

Check before any move:
1. Is new position within bounds?
2. Does new position overlap locked cells?

### Game Loop

**requestAnimationFrame pattern:**
```javascript
let lastTime = 0;
let dropCounter = 0;

function update(time) {
  const delta = time - lastTime;
  lastTime = time;

  dropCounter += delta;
  if (dropCounter > dropInterval) {
    dropPiece();
    dropCounter = 0;
  }

  render();
  requestAnimationFrame(update);
}
```

**Lock delay:** Brief pause (500ms) after piece lands before locking. Allows last-second moves.

### Line Clearing

1. Check all rows from bottom to top
2. If row full → mark for clearing
3. Remove marked rows
4. Shift rows above down
5. (In Phase 2: award points)

### Input Handling

**Key mappings:**
- Left/Right arrows: Move
- Down arrow: Soft drop (faster fall)
- Up arrow: Rotate clockwise
- Spacebar: Hard drop (instant)

**Key repeat (DAS - Delayed Auto Shift):**
- Initial delay: 170ms
- Repeat rate: 50ms

Track keydown/keyup separately. Process input each frame.

### File Structure

```
/
├── index.html
├── css/
│   └── game.css
└── js/
    ├── main.js           # Entry point, game loop
    ├── board.js          # Grid state, collision
    ├── pieces.js         # Tetromino definitions
    ├── input.js          # Keyboard handling
    └── render.js         # Canvas drawing
```

## Critical Implementation Notes

1. **Coordinate system:** (0,0) is top-left. Y increases downward.

2. **Grid storage:** 2D array of cell colors (null = empty).

3. **Piece spawn position:** Centered at top (x=3 for most pieces, x=4 for O).

4. **Game over check:** If new piece immediately collides on spawn.

5. **Rendering order:**
   - Clear canvas
   - Draw grid background
   - Draw locked cells
   - Draw current piece
   - Draw grid lines (optional)

## Dependencies on Later Phases

- Phase 2 adds: scoring display, next preview, hold, ghost, pause
- Phase 3 adds: themes, admin sync (requires extracting config)
- Phase 4 adds: freeze timer, board growth (requires board resize logic)

**Build for extensibility:**
- Keep drop speed as a variable (config in Phase 3)
- Keep scoring logic separate (even if not displayed)
- Board dimensions should be variables, not hardcoded
