# Phase 2 Research: Scoring & Polish

## What This Phase Builds

Game polish features: ghost piece, hold piece, next preview, score/level display, pause functionality. These are expected in any modern Tetris clone.

## Key Implementation Decisions

### Ghost Piece

**What it is:** Semi-transparent preview showing where piece will land.

**Implementation:**
1. Calculate landing position: drop current piece virtually until collision
2. Draw ghost at landing position with reduced opacity (30-40%)
3. Must update every frame as piece moves
4. Color should match current piece but lighter/transparent

```javascript
function getGhostY(board, piece) {
    let ghostY = piece.y;
    while (isValidPosition(board, piece.type, piece.x, ghostY + 1, piece.rotation)) {
        ghostY++;
    }
    return ghostY;
}
```

### Hold Piece

**What it is:** Player can swap current piece with held piece (once per drop).

**Implementation:**
1. Hold slot stores one piece type (initially null)
2. On hold key (C or Shift):
   - If no held piece: store current, spawn next
   - If held piece exists: swap current with held
3. Can only hold once per piece (reset on lock)
4. Need UI area to display held piece

**State:**
```javascript
let heldPiece = null;      // Piece type string or null
let canHold = true;        // Reset to true on spawn
```

### Next Piece Preview

**What it is:** Shows upcoming piece(s) so player can plan.

**Implementation:**
1. Need to look ahead in the bag without consuming
2. Draw preview area (right side of board)
3. Draw next piece shape in preview area
4. May show 1-3 next pieces (1 for v1)

**Canvas layout change:**
- Current: 300px wide (board only)
- New: Need sidebar for score, next, hold
- Suggestion: Board + 120px sidebar = 420px total

### Pause

**What it is:** P key pauses/resumes game.

**Implementation:**
1. `isPaused` boolean flag
2. When paused:
   - Stop game loop updates
   - Still render (show paused state)
   - Show "PAUSED" overlay
   - Block input except P to resume
3. When unpaused: resume normal loop

### Score Display

**What it is:** Show current score and level.

**Implementation:**
1. Already tracking `score` in main.js
2. Add level calculation: `level = Math.floor(score / 100) + 1`
3. Draw in sidebar area
4. Update font, position

### Level-Up Theme Change

**What it is:** Visual change when level increases.

**For Phase 2:** Just track level and trigger event. Actual theme system in Phase 3.

**Implementation:**
```javascript
let currentLevel = 1;

function checkLevelUp() {
    const newLevel = Math.floor(score / 100) + 1;
    if (newLevel > currentLevel) {
        currentLevel = newLevel;
        onLevelUp();  // Trigger theme change (Phase 3 will implement)
    }
}
```

## UI Layout Changes

**Current layout:**
```
┌────────────┐
│            │
│   Board    │
│  (300px)   │
│            │
└────────────┘
```

**New layout:**
```
┌────────────┬──────────┐
│            │  HOLD    │
│            │  ┌────┐  │
│            │  │    │  │
│            │  └────┘  │
│   Board    │          │
│  (300px)   │  NEXT    │
│            │  ┌────┐  │
│            │  │    │  │
│            │  └────┘  │
│            │          │
│            │  SCORE   │
│            │  12340   │
│            │  LEVEL 3 │
└────────────┴──────────┘
```

**Canvas dimensions:**
- Board: 300 x 600 (unchanged)
- Sidebar: 120 x 600
- Total: 420 x 600

## File Changes

| File | Changes |
|------|---------|
| index.html | Possibly widen canvas container |
| css/game.css | Adjust for wider canvas |
| js/render.js | Add drawGhost, drawSidebar, drawHold, drawNext, drawScore, drawPaused |
| js/main.js | Add hold logic, pause logic, level tracking, sidebar state |
| js/input.js | Add C/Shift for hold, P for pause |

## Dependencies on Phase 1

Uses from Phase 1:
- `currentPiece`, `board` state
- `isValidPosition()` for ghost calculation
- `PIECES` definitions for rendering previews
- `spawnPiece()` needs modification for hold swap

## Dependencies on Phase 3

Phase 3 will add:
- Actual theme definitions
- `onLevelUp()` will trigger theme switch
- For now, just track level and log/no-op on level up
