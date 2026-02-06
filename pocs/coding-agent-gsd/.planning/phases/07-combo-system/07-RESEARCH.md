# Phase 7: Combo System - Research

**Researched:** 2026-02-05
**Domain:** Tetris combo scoring, Back-to-Back mechanics, visual feedback
**Confidence:** HIGH

## Summary

Research focused on implementing the Tetris combo system following Guideline standards. The combo system awards bonus points for consecutive line clears, using the formula `50 x combo x level`. The Back-to-Back (B2B) system awards a 1.5x multiplier for consecutive "difficult" clears (Tetris and T-Spin line clears).

The existing codebase already has the integration points needed: `lockPieceToBoard()` in main.js handles piece locking and line clearing, `stats.js` tracks game statistics, and `render.js` handles all visual display including the sidebar. The implementation requires adding combo state tracking, B2B state tracking, and visual feedback rendering.

**Primary recommendation:** Add `combo` and `b2bActive` state variables to track combo/B2B chains, update scoring in the line clear handler, and add visual indicators to the sidebar rendering.

## Standard Stack

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| Vanilla JS | ES6+ | State management | Already used in codebase |
| HTML5 Canvas | - | Visual rendering | Already used for all rendering |

### Supporting
| Component | Purpose | When to Use |
|-----------|---------|-------------|
| stats.js | Combo statistics tracking | Extend with combo/b2b counters |
| render.js | Visual combo display | Add combo indicator rendering |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Sidebar display | Floating text animation | Floating text more complex, sidebar simpler and consistent with existing UI |
| Global state | Module pattern | Global state already used throughout codebase |

## Architecture Patterns

### Recommended State Structure
```javascript
let combo = 0;
let b2bActive = false;
let lastClearWasDifficult = false;
```

### Pattern 1: Combo State Machine
**What:** Track combo counter that increments on line clears, resets on piece lock without clear
**When to use:** Every piece lock event

**State transitions:**
```
PIECE_LOCKS_WITH_LINES -> combo++
PIECE_LOCKS_NO_LINES -> combo = 0
```

### Pattern 2: Back-to-Back Tracking
**What:** Track whether last line clear was "difficult" (4-line Tetris)
**When to use:** After each line clear to determine if B2B bonus applies

**State transitions:**
```
TETRIS_CLEAR && b2bActive -> award 1.5x bonus, keep b2bActive = true
TETRIS_CLEAR && !b2bActive -> normal scoring, set b2bActive = true
NON_TETRIS_CLEAR -> normal scoring, set b2bActive = false
NO_CLEAR -> b2bActive unchanged (does NOT break chain)
```

### Pattern 3: Scoring Integration
**What:** Calculate combo bonus and B2B multiplier in existing score calculation
**When to use:** In the line clearing handler after `clearLines()` returns

```javascript
function calculateScore(linesCleared, level, combo, b2bActive) {
    let baseScore = getBaseScore(linesCleared) * level;
    let comboBonus = 50 * combo * level;
    if (b2bActive && linesCleared === 4) {
        baseScore = Math.floor(baseScore * 1.5);
    }
    return baseScore + comboBonus;
}
```

### Anti-Patterns to Avoid
- **Resetting combo on non-clearing placement:** The combo only breaks when a piece LOCKS without clearing lines, not just when placed
- **Breaking B2B on zero-line placement:** B2B chain persists through non-clearing placements
- **Starting combo at 1:** Combo counter starts at 0 or -1; first line clear is "1-combo" but bonus only applies from 2nd consecutive clear

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Scoring formula | Custom point calculation | Guideline formula (50 x combo x level) | Well-tested standard |
| B2B detection | Complex clear type analysis | Simple line count check (4 = difficult) | T-Spin detection not in requirements |
| Animation system | Complex tween library | Simple alpha fade with requestAnimationFrame | Already using canvas rendering |

**Key insight:** The requirements only mention "consecutive Tetris clears" for B2B, not T-Spins. Since T-Spin detection is complex and not in requirements, implement B2B for 4-line clears (Tetris) only.

## Common Pitfalls

### Pitfall 1: Combo Counter Off-By-One
**What goes wrong:** First consecutive clear shows "2x Combo" instead of "1x Combo"
**Why it happens:** Confusion about when combo counter increments
**How to avoid:** Increment combo BEFORE calculating bonus; first clear shows "1x", awards 50 x 1 x level bonus on SECOND consecutive clear
**Warning signs:** Combo display doesn't match expected sequence

### Pitfall 2: B2B Breaking on Non-Clears
**What goes wrong:** B2B chain breaks when placing pieces without clearing
**Why it happens:** Misunderstanding that B2B only breaks on NON-TETRIS line clears
**How to avoid:** Only check B2B break condition when lines ARE cleared
**Warning signs:** B2B indicator disappears after placing a piece that doesn't clear

### Pitfall 3: Score Calculation Order
**What goes wrong:** Combo bonus uses old combo value
**Why it happens:** Incrementing combo after calculating score instead of before
**How to avoid:** Increment combo first, then calculate score with new value
**Warning signs:** First consecutive clear shows combo but awards no bonus

### Pitfall 4: Visual Display Timing
**What goes wrong:** Combo indicator shows briefly then disappears immediately
**Why it happens:** Clearing combo state before rendering
**How to avoid:** Display combo for duration of clearing animation, reset after
**Warning signs:** Combo text flashes too quickly to read

## Code Examples

### Combo State Update
```javascript
function lockPieceToBoard() {
    if (!currentPiece) return;

    updatePiecePlaced();
    board = lockPiece(board, currentPiece);
    const lines = checkLines(board);

    if (lines.length > 0) {
        combo++;

        let isDifficultClear = (lines.length === 4);
        let b2bBonus = (b2bActive && isDifficultClear);

        clearingLines = lines;
        clearingTimer = 100;
        currentPiece = null;

        pendingScoreCalc = { lines: lines.length, b2bBonus: b2bBonus };

        if (isDifficultClear) {
            b2bActive = true;
        } else {
            b2bActive = false;
        }
    } else {
        combo = 0;
        spawnPiece();
    }
}
```

### Score Calculation with Combo and B2B
```javascript
function calculateLineScore(linesCleared, level, currentCombo, hasB2bBonus) {
    const baseScores = { 1: 100, 2: 300, 3: 500, 4: 800 };
    let lineScore = baseScores[linesCleared] * level;

    if (hasB2bBonus) {
        lineScore = Math.floor(lineScore * 1.5);
    }

    let comboBonus = 0;
    if (currentCombo > 1) {
        comboBonus = 50 * (currentCombo - 1) * level;
    }

    return lineScore + comboBonus;
}
```

### Visual Combo Display in Sidebar
```javascript
function drawComboIndicator(combo, b2bActive) {
    if (combo <= 0 && !b2bActive) return;

    var sidebarX = COLS * CELL_SIZE;
    var comboY = 550;

    if (combo > 0) {
        ctx.fillStyle = '#ff00ff';
        ctx.font = 'bold 18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(combo + 'x COMBO', sidebarX + SIDEBAR_WIDTH / 2, comboY);
    }

    if (b2bActive) {
        ctx.fillStyle = '#ffff00';
        ctx.font = 'bold 14px Arial';
        ctx.fillText('BACK-TO-BACK', sidebarX + SIDEBAR_WIDTH / 2, comboY + 25);
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Simple line scoring | Combo + B2B systems | Tetris DS (2006) | Modern games expect combo scoring |
| Fixed point values | Level multipliers | Guideline adoption | Score scales with progression |

**Deprecated/outdated:**
- NES Tetris scoring (no combos): Modern players expect combo rewards
- Immediate score display: Visual feedback with animations preferred

## Integration Points

### main.js Integration
| Location | Current Code | Required Change |
|----------|--------------|-----------------|
| Line 14 | `let score = 0;` | Add `let combo = 0; let b2bActive = false;` |
| Line 193-206 | `lockPieceToBoard()` | Add combo/b2b state updates |
| Line 278-289 | Line clear score calculation | Use new scoring formula |
| Line 352-372 | `resetGame()` | Reset combo and b2bActive |

### render.js Integration
| Location | Current Code | Required Change |
|----------|--------------|-----------------|
| After Line 218 | End of `drawSessionStats()` | Add `drawComboIndicator()` call |
| New function | N/A | Add `drawComboIndicator(combo, b2bActive)` |

### stats.js Integration
| Location | Current Code | Required Change |
|----------|--------------|-----------------|
| Line 1-9 | stats object | Add `maxCombo: 0, b2bCount: 0` |
| New function | N/A | Add `updateComboStats(combo)` |

## Open Questions

1. **Combo display duration**
   - What we know: Combo should display during active combo
   - What's unclear: Should combo number persist briefly after chain breaks?
   - Recommendation: Display during active combo only; reset visually when combo resets

2. **B2B visual indicator style**
   - What we know: Requirements say "Back-to-Back indicator"
   - What's unclear: Exact visual treatment (text label vs icon)
   - Recommendation: Use text label "BACK-TO-BACK" below combo counter

3. **Combo bonus edge case**
   - What we know: Formula is 50 x combo x level
   - What's unclear: Does first clear in combo (1x) award 50 points or start at 2x?
   - Recommendation: Award bonus starting from second consecutive clear (combo > 1)

## Sources

### Primary (HIGH confidence)
- [TetrisWiki Scoring](https://tetris.wiki/Scoring) - Complete scoring tables, combo formula
- [TetrisWiki Combo](https://tetris.wiki/Combo) - Combo counter mechanics
- [TetrisWiki T-Spin](https://tetris.wiki/T-Spin) - T-Spin detection rules
- [Hard Drop Back-to-Back](https://harddrop.com/wiki/Back-to-Back) - B2B chain mechanics

### Secondary (MEDIUM confidence)
- Existing codebase analysis - Integration points verified by reading source

### Tertiary (LOW confidence)
- None - all claims verified with authoritative sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using existing codebase patterns
- Architecture: HIGH - Based on Tetris Guideline standards
- Pitfalls: HIGH - Well-documented in community wikis
- Scoring formulas: HIGH - Official Guideline specification

**Research date:** 2026-02-05
**Valid until:** 2026-03-05 (30 days - stable domain, established standards)
