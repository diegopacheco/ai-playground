# Phase 4: Unique Mechanics - Research

**Researched:** 2026-02-02
**Domain:** Time-based game state cycles, dynamic board growth, canvas overlay UI
**Confidence:** MEDIUM

## Summary

This phase implements two signature mechanics for the Tetris variant: freeze cycles and growing board. The freeze cycle alternates between 10-second play and 10-second freeze periods, requiring a state machine pattern to manage transitions and visual indicators. The growing board mechanic adds rows at intervals, requiring dynamic 2D array manipulation and coordinate preservation.

The existing codebase already has a solid foundation with requestAnimationFrame-based game loop, deltaTime tracking, and BroadcastChannel infrastructure. The admin panel's growth interval slider is implemented, so this phase focuses on implementing the mechanics themselves.

**Primary recommendation:** Use enum-based state machine for freeze/play cycles, accumulator pattern for precise timing, and array push method for board growth. Implement visual overlays with semi-transparent canvas drawing and countdown text.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vanilla JS | ES6+ | State machines, timers | No dependencies requirement from project constraints |
| Canvas API | Native | Visual overlays, countdown text | Already in use, no additional dependencies |
| requestAnimationFrame | Native | Game loop timing | Industry standard for smooth game loops |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| BroadcastChannel | Native | State sync to admin panel | Already implemented for other features |
| Array methods | Native | Board row manipulation | Built-in, performant for 2D arrays |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Enum states | Boolean flags | Multiple booleans create 2^n complexity vs 4 clear states |
| Array.push | Array.splice | push() is simpler and clearer for bottom-row addition |
| globalAlpha | rgba() colors | globalAlpha affects all subsequent draws, rgba is more precise |

**Installation:**
No installation needed - all vanilla JavaScript.

## Architecture Patterns

### Recommended Project Structure
```
js/
├── main.js           # Add freeze state, cycle timers, growth timer
├── board.js          # Add growBoard() function
├── render.js         # Add drawFreezeOverlay(), drawCountdown()
└── sync.js           # Already supports state sync
```

### Pattern 1: Enum-Based State Machine
**What:** Use string constants or object with frozen properties to represent game states
**When to use:** Managing mutually exclusive states like PLAYING, FROZEN, PAUSED, GAME_OVER

**Example:**
```javascript
const GameState = Object.freeze({
    PLAYING: 'PLAYING',
    FROZEN: 'FROZEN',
    PAUSED: 'PAUSED',
    GAME_OVER: 'GAME_OVER'
});

let gameState = GameState.PLAYING;

function update(deltaTime) {
    if (gameState === GameState.FROZEN) {
        return;
    }
    if (gameState === GameState.PAUSED) {
        return;
    }
    if (gameState === GameState.GAME_OVER) {
        return;
    }
}
```

**Why this pattern:**
- Eliminates boolean flag combinations (3 bools = 8 states vs 4 clear states)
- Single source of truth for current state
- Clear transition logic
- Easier debugging and logging

### Pattern 2: Accumulator Pattern for Timers
**What:** Accumulate deltaTime in counters, trigger events when threshold reached
**When to use:** Time-based cycles, intervals, countdowns

**Example:**
```javascript
let cycleTimer = 0;
const PLAY_DURATION = 10000;
const FREEZE_DURATION = 10000;

function update(deltaTime) {
    cycleTimer += deltaTime;

    if (gameState === GameState.PLAYING && cycleTimer >= PLAY_DURATION) {
        gameState = GameState.FROZEN;
        cycleTimer = 0;
    } else if (gameState === GameState.FROZEN && cycleTimer >= FREEZE_DURATION) {
        gameState = GameState.PLAYING;
        cycleTimer = 0;
    }
}
```

**Why this pattern:**
- Frame-rate independent timing
- Precise control over intervals
- Works with existing requestAnimationFrame loop

### Pattern 3: Dynamic 2D Array Growth
**What:** Add rows to bottom of 2D array while preserving existing content
**When to use:** Growing playfield, expanding game board

**Example:**
```javascript
function growBoard(board, cols) {
    const newRow = new Array(cols).fill(null);
    board.push(newRow);
    return board;
}

let growthTimer = 0;
let boardGrowthInterval = 30000;
const MAX_ROWS = 30;

function update(deltaTime) {
    growthTimer += deltaTime;

    if (growthTimer >= boardGrowthInterval && board.length < MAX_ROWS) {
        board = growBoard(board, COLS);
        growthTimer = 0;
    }
}
```

**Why this pattern:**
- Simple and clear intent
- Preserves existing array references
- No complex coordinate transformation needed

### Pattern 4: Canvas Overlay with Transparency
**What:** Draw semi-transparent overlay over game area with text indicators
**When to use:** Visual state indicators, countdowns, game pauses

**Example:**
```javascript
function drawFreezeOverlay() {
    ctx.fillStyle = 'rgba(0, 100, 200, 0.6)';
    ctx.fillRect(0, 0, COLS * CELL_SIZE, ROWS * CELL_SIZE);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('FROZEN', (COLS * CELL_SIZE) / 2, (ROWS * CELL_SIZE) / 2);
}

function drawCountdown(seconds) {
    ctx.fillStyle = '#00ffff';
    ctx.font = 'bold 36px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(seconds + 's', (COLS * CELL_SIZE) / 2, (ROWS * CELL_SIZE) / 2 + 60);
}
```

**Why this pattern:**
- Clear visual distinction between states
- Non-intrusive countdown display
- Consistent with existing pause/game-over overlays

### Anti-Patterns to Avoid
- **Multiple boolean flags for state:** Instead of `isFrozen`, `isPlaying`, `isPaused`, use single state enum to avoid impossible state combinations
- **Using setInterval/setTimeout:** Breaks synchronization with game loop, causes drift over time
- **Modifying canvas dimensions:** Canvas resize clears content; grow the board array, update ROWS constant, recalculate canvas height once
- **Direct time comparisons:** Use accumulators instead of `if (Date.now() - startTime > 10000)` to avoid frame skip issues
- **Splice for bottom-row addition:** Unnecessarily complex; use push() for clarity

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Time tracking | Custom Date.now() tracking | deltaTime from requestAnimationFrame | Frame-independent, no drift, already implemented |
| State transitions | Ad-hoc if/else chains | Enum-based state machine | Prevents impossible states, clearer logic |
| Array manipulation | Custom row insertion logic | Array.push() for bottom rows | Built-in, optimized, clear intent |
| Text centering | Manual pixel calculation | textAlign = 'center' + textBaseline | Canvas API handles this correctly |
| Transparency | Pixel manipulation | rgba() colors or globalAlpha | Hardware-accelerated, correct compositing |

**Key insight:** The canvas and array APIs provide all necessary primitives. Custom solutions add complexity without benefit.

## Common Pitfalls

### Pitfall 1: Freeze State Doesn't Stop Input Processing
**What goes wrong:** Players can still rotate/move pieces during freeze, queue up inputs that execute when unfrozen
**Why it happens:** Input processing runs independently of game state checks
**How to avoid:** Add state check at top of processInput() - return early if gameState is FROZEN
**Warning signs:** Pieces move or rotate immediately after unfreeze

### Pitfall 2: Timer Drift from Frame Skips
**What goes wrong:** Cycles become inconsistent (9.7s play, 10.3s freeze), especially on slower devices
**Why it happens:** Using frame count instead of accumulated time, or not capping deltaTime max
**How to avoid:** Always use deltaTime accumulator, cap maximum deltaTime to prevent "spiral of death"
**Warning signs:** Timers become unreliable when tab loses focus or device lags

### Pitfall 3: Board Growth Breaks Collision Detection
**What goes wrong:** After board grows, collision detection fails, pieces fall through bottom
**Why it happens:** ROWS constant not updated, or collision checks hardcoded to old height
**How to avoid:** Either make ROWS mutable and update it, or use board.length in collision checks
**Warning signs:** Pieces don't lock at new bottom row after growth

### Pitfall 4: Canvas Height Not Updated After Growth
**What goes wrong:** New rows render outside visible canvas, or grid doesn't extend
**Why it happens:** Canvas dimensions set once in setupCanvas(), not recalculated on growth
**How to avoid:** Recalculate and resize canvas when board grows, or pre-size canvas to MAX_ROWS
**Warning signs:** New bottom rows invisible or grid lines don't extend

### Pitfall 5: Countdown Shows Decimal Milliseconds
**What goes wrong:** Countdown displays "3.847s" instead of "3s" or "4s"
**Why it happens:** Rendering raw timer value without rounding
**How to avoid:** Use Math.ceil() to round up to nearest second for countdown display
**Warning signs:** Jittery, hard-to-read countdown numbers

### Pitfall 6: Freeze Cycle Starts Mid-Game
**What goes wrong:** Game freezes unexpectedly within first 10 seconds of play
**Why it happens:** cycleTimer not reset in resetGame(), starts from accumulated value
**How to avoid:** Initialize cycleTimer = 0 in resetGame() function
**Warning signs:** First freeze happens at wrong time after reset

### Pitfall 7: Growth Timer Not Synced from Admin Panel
**What goes wrong:** Changing growth interval in admin panel doesn't affect next growth
**Why it happens:** Timer not reset when interval changes, or already-accumulated time triggers immediate growth
**How to avoid:** When receiving GROWTH_INTERVAL_CHANGE, optionally reset growthTimer or clamp it: `growthTimer = Math.min(growthTimer, newInterval)`
**Warning signs:** Board grows immediately after interval decrease

## Code Examples

Verified patterns from official sources:

### State Machine Setup
```javascript
const GameState = Object.freeze({
    PLAYING: 'PLAYING',
    FROZEN: 'FROZEN',
    PAUSED: 'PAUSED',
    GAME_OVER: 'GAME_OVER'
});

let gameState = GameState.PLAYING;
let cycleTimer = 0;
const PLAY_DURATION = 10000;
const FREEZE_DURATION = 10000;
```

### Freeze Cycle Update Logic
```javascript
function update(deltaTime) {
    if (gameState === GameState.GAME_OVER) return;
    if (gameState === GameState.PAUSED) return;

    cycleTimer += deltaTime;

    if (gameState === GameState.PLAYING) {
        if (cycleTimer >= PLAY_DURATION) {
            gameState = GameState.FROZEN;
            cycleTimer = 0;
        }
    } else if (gameState === GameState.FROZEN) {
        if (cycleTimer >= FREEZE_DURATION) {
            gameState = GameState.PLAYING;
            cycleTimer = 0;
            canHold = true;
        }
        return;
    }

    if (clearingLines.length > 0) {
    }
}
```

### Board Growth with Safety Checks
```javascript
const MAX_ROWS = 30;
let growthTimer = 0;

function growBoard(board, cols) {
    const newRow = new Array(cols).fill(null);
    board.push(newRow);
    return board;
}

function update(deltaTime) {
    if (gameState !== GameState.PLAYING) return;

    growthTimer += deltaTime;

    if (growthTimer >= boardGrowthInterval && board.length < MAX_ROWS) {
        board = growBoard(board, COLS);
        growthTimer = 0;

        const height = board.length * CELL_SIZE;
        canvas.style.height = height + 'px';
        canvas.height = height * (window.devicePixelRatio || 1);
    }
}
```

### Freeze Overlay Rendering
```javascript
function drawFreezeOverlay() {
    ctx.fillStyle = 'rgba(50, 150, 255, 0.5)';
    ctx.fillRect(0, 0, COLS * CELL_SIZE, board.length * CELL_SIZE);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('FROZEN', (COLS * CELL_SIZE) / 2, (board.length * CELL_SIZE) / 2 - 30);

    const remainingSeconds = Math.ceil((FREEZE_DURATION - cycleTimer) / 1000);
    ctx.fillStyle = '#00ffff';
    ctx.font = 'bold 36px Arial';
    ctx.fillText(remainingSeconds + 's', (COLS * CELL_SIZE) / 2, (board.length * CELL_SIZE) / 2 + 30);
}

function render() {
    drawGrid();
    drawBoard(board);

    if (gameState === GameState.FROZEN) {
        drawFreezeOverlay();
    }

    drawSidebar();
}
```

### Input Blocking During Freeze
```javascript
function processInput() {
    const input = getInput();

    if (input.pause) {
        togglePause();
        return;
    }

    if (gameState === GameState.FROZEN) return;
    if (gameState === GameState.PAUSED) return;
    if (gameOver || clearingLines.length > 0) return;

    if (!currentPiece) return;

    if (input.hold) {
        holdPiece();
    }
}
```

### Reset Game State
```javascript
function resetGame() {
    board = createBoard();
    bag = [];
    nextPiece = null;
    heldPiece = null;
    canHold = true;
    score = 0;
    level = 1;
    gameState = GameState.PLAYING;
    cycleTimer = 0;
    growthTimer = 0;
    clearingLines = [];
    clearingTimer = 0;
    dropCounter = 0;
    lockCounter = 0;
    isLocking = false;
    spawnPiece();
}
```

### Collision Detection with Dynamic Board
```javascript
function isValidPosition(board, pieceType, x, y, rotation) {
    const shape = PIECES[pieceType].shapes[rotation];
    for (let row = 0; row < shape.length; row++) {
        for (let col = 0; col < shape[row].length; col++) {
            if (shape[row][col]) {
                const newX = x + col;
                const newY = y + row;
                if (newX < 0 || newX >= COLS || newY >= board.length) {
                    return false;
                }
                if (newY >= 0 && board[newY][newX]) {
                    return false;
                }
            }
        }
    }
    return true;
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| setInterval for cycles | requestAnimationFrame + accumulator | ~2010-2012 | Frame-independent timing, no drift |
| Boolean state flags | Enum-based state machines | ~2015+ | Clearer logic, prevents impossible states |
| Array.splice for growth | Array.push for bottom addition | Always | Simpler, more readable |
| Date.now() comparisons | deltaTime accumulators | ~2012+ | No tab-out issues, precise timing |
| Manual text positioning | textAlign/textBaseline | Canvas API v1 | Hardware-accelerated, correct rendering |

**Deprecated/outdated:**
- setInterval/setTimeout for game loops: Causes drift, not synced with rendering
- Direct Date.now() tracking: Breaks when tab loses focus
- Hardcoded ROWS constant: Prevents dynamic board growth

## Open Questions

Things that couldn't be fully resolved:

1. **Board Growth Visual Transition**
   - What we know: Array push is instantaneous
   - What's unclear: Whether smooth animation (slide-in) is desired vs instant appearance
   - Recommendation: Start with instant growth for simplicity, can add slide animation later if desired

2. **Freeze Cycle Pause Interaction**
   - What we know: Game can be manually paused, freeze is automatic
   - What's unclear: Should manual pause stop the freeze/play cycle timer, or should cycle continue
   - Recommendation: Pause should stop all timers including cycle timer, preserving time remaining

3. **Board Growth During Freeze**
   - What we know: Growth timer accumulates, freeze timer accumulates
   - What's unclear: Should board growth be blocked during freeze state
   - Recommendation: Allow growth during freeze for fairness, only block piece movement

4. **Canvas Resize Strategy**
   - What we know: Canvas resize clears content, dynamic height needed
   - What's unclear: Pre-size to MAX_ROWS vs resize on each growth
   - Recommendation: Pre-size canvas to MAX_ROWS for simplicity, avoid resize clears

## Sources

### Primary (HIGH confidence)
- Canvas API - fillText, strokeText, globalAlpha from MDN Web Docs
- Array methods - push, splice from MDN JavaScript Reference
- requestAnimationFrame timing from MDN Web APIs

### Secondary (MEDIUM confidence)
- State machine patterns: [Game Programming Patterns - State](https://gameprogrammingpatterns.com/state.html)
- Finite state machines in JavaScript: [DEV Community - FSM in JavaScript](https://dev.to/spukas/finite-state-machine-in-javascript-1ki1)
- Time-based animation: [NickNagel.com - Time-Based Animation with requestAnimationFrame](https://dr-nick-nagel.github.io/blog/raf-time.html)
- Countdown timer patterns: [A countdown clock using requestAnimationFrame and state machine](https://gotofritz.net/blog/countdown-clock-state-machine-requestanimationframe-vanilla-js/)
- 2D array manipulation: [JavaScript 2D Arrays - freeCodeCamp](https://www.freecodecamp.org/news/javascript-2d-arrays/)
- Canvas text rendering: [How to Draw Text on an HTML Canvas](https://jsdev.space/howto/filltext-canvas/)
- Transparency patterns: [Canvas API - Applying styles and colors - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Applying_styles_and_colors)

### Tertiary (LOW confidence - WebSearch unverified)
- Game loop patterns: [Performant Game Loops in JavaScript](https://www.aleksandrhovhannisyan.com/blog/javascript-game-loop/)
- DeltaTime common mistakes: [Fix Your Timestep - Gaffer On Games](https://gafferongames.com/post/fix_your_timestep/)
- Modern state management: [Modern State Management in Vanilla JavaScript 2026](https://medium.com/@orami98/modern-state-management-in-vanilla-javascript-2026-patterns-and-beyond-ce00425f7ac5)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Vanilla JS and Canvas API verified as project constraints, no dependencies allowed
- Architecture: MEDIUM - Patterns verified from multiple sources, specific implementation adapted to existing codebase
- Pitfalls: MEDIUM - Derived from general game programming patterns and WebSearch sources, not Tetris-specific

**Research date:** 2026-02-02
**Valid until:** 2026-03-02 (30 days - stable patterns, vanilla JS APIs don't change rapidly)
