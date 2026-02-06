# Phase 8: T-Spin Detection - Research

**Researched:** 2026-02-05
**Domain:** Tetris T-spin detection algorithms and scoring mechanics
**Confidence:** HIGH

## Summary

T-spin detection is a standardized mechanic in modern Tetris games following the Tetris Guideline. The most widely adopted algorithm is the "3-corner T rule" which checks two conditions: the last movement was a rotation, and three of four diagonal corners around the T-piece center are occupied. The system distinguishes between "mini" and "full" T-spins based on which corners are filled, with different point values for each type. T-spins integrate with the back-to-back (B2B) system, treating any T-spin that clears lines as a "difficult clear."

The implementation requires tracking rotation state (to distinguish rotation from drops), checking corner occupancy after piece locks, and calculating bonus points based on T-spin type and lines cleared. Visual feedback displays the T-spin type to the player.

**Primary recommendation:** Use the 3-corner T rule with front/back corner classification for mini vs full detection. Track last action type (rotation vs movement) before piece locks. Integrate with existing B2B system by treating line-clearing T-spins as difficult clears.

## Standard Stack

The established approach for T-spin detection in Tetris:

### Core Algorithm
| Component | Implementation | Purpose | Why Standard |
|-----------|---------------|---------|--------------|
| 3-corner T rule | Check 3+ diagonal corners occupied | Detect T-spin | Used in Tetris DS, modern Guideline games |
| Last action tracking | Flag rotation vs movement/drop | Verify rotation requirement | Prevents false positives from drops |
| Front/back corner check | Classify which corners filled | Distinguish mini vs full | Standard since Tetris Guideline 2006 |
| Point calculation | Base + line clear bonus | Award scores | Consistent across Guideline games |

### Supporting Components
| Component | Purpose | When to Use |
|-----------|---------|-------------|
| Visual indicator | Display "T-Spin Mini" or "T-Spin" text | After detection |
| B2B integration | Track T-spin as difficult clear | When lines cleared |
| Wall kick upgrade | Promote mini to full for 1x2 kick | Special SRS rotation offset |

### Algorithm Variations
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 3-corner T rule | Immobility test | Less accurate, false positives in enclosed spaces |
| Front/back corners | 3-corner T no kick | Simpler but excludes valid wall-kicked T-spins |
| Standard scoring | Custom point values | Not Guideline-compliant |

**No special libraries needed:** Pure JavaScript implementation using existing board collision detection.

## Architecture Patterns

### Detection Flow
```
Piece lock sequence:
1. lockPieceToBoard() called
2. Check if piece type === 'T'
3. Check if lastAction === 'rotation'
4. Check corner occupancy (3+ of 4)
5. Classify mini vs full (front corners)
6. Apply wall kick upgrade if applicable
7. Calculate bonus points
8. Set B2B if lines cleared
9. Show visual indicator
```

### State Tracking Pattern
```javascript
let lastAction = null;
let lastKickOffset = null;

function rotatePiece() {
    const kickUsed = performRotationWithKicks();
    if (kickUsed) {
        lastAction = 'rotation';
        lastKickOffset = kickUsed;
    }
}

function movePiece(dx, dy) {
    lastAction = 'movement';
    lastKickOffset = null;
}
```

### Corner Check Pattern
The T-piece rotates around its center block. The four diagonal corners are relative to this center:

```
Rotation 0 (spawn):     Rotation 1 (CW):        Rotation 2 (180):       Rotation 3 (CCW):
    [0,1,0]                 [0,1,0]                 [0,0,0]                 [0,1,0]
    [1,1,1]                 [0,1,1]                 [1,1,1]                 [1,1,0]
    [0,0,0]                 [0,1,0]                 [0,1,0]                 [0,1,0]

Center: (1,1)           Center: (1,1)           Center: (1,0)           Center: (1,1)
Front corners:          Front corners:          Front corners:          Front corners:
  (0,0), (2,0)            (2,0), (2,2)            (0,2), (2,2)            (0,0), (0,2)
Back corners:           Back corners:           Back corners:           Back corners:
  (0,2), (2,2)            (0,0), (0,2)            (0,0), (2,0)            (2,0), (2,2)
```

### Corner Position Calculator
```javascript
function getCornerPositions(piece) {
    const cx = piece.x + 1;
    const cy = piece.y + 1;

    const corners = [
        [cx - 1, cy - 1],
        [cx + 1, cy - 1],
        [cx - 1, cy + 1],
        [cx + 1, cy + 1]
    ];

    const frontCorners = getFrontCornersForRotation(piece.rotation);
    const backCorners = getBackCornersForRotation(piece.rotation);

    return { corners, frontCorners, backCorners };
}

function getFrontCornersForRotation(rotation) {
    const frontIndices = [
        [0, 1],
        [1, 3],
        [2, 3],
        [0, 2]
    ];
    return frontIndices[rotation];
}
```

### Detection Implementation
```javascript
function detectTSpin(piece, board, lastAction, lastKickOffset) {
    if (piece.type !== 'T') return null;
    if (lastAction !== 'rotation') return null;

    const { corners, frontCorners, backCorners } = getCornerPositions(piece);

    let occupiedCount = 0;
    let frontOccupied = 0;

    corners.forEach((corner, idx) => {
        const [x, y] = corner;
        const isOccupied = (x < 0 || x >= COLS || y >= board.length ||
                           (y >= 0 && board[y][x]));
        if (isOccupied) {
            occupiedCount++;
            if (frontCorners.includes(idx)) frontOccupied++;
        }
    });

    if (occupiedCount < 3) return null;

    const isMini = frontOccupied < 2;
    const isWallKickUpgrade = lastKickOffset &&
        (Math.abs(lastKickOffset[0]) + Math.abs(lastKickOffset[1]) === 3);

    return isMini && !isWallKickUpgrade ? 'mini' : 'full';
}
```

### Scoring Pattern
```javascript
function calculateTSpinScore(tSpinType, linesCleared, level, hasB2B) {
    const basePoints = {
        mini: { 0: 100, 1: 200, 2: 400 },
        full: { 0: 400, 1: 800, 2: 1200, 3: 1600 }
    };

    let score = basePoints[tSpinType][linesCleared] || 0;
    score *= level;

    if (hasB2B && linesCleared > 0) {
        score = Math.floor(score * 1.5);
    }

    return score;
}
```

### Anti-Patterns to Avoid

- **Checking T-spin on every piece type:** Only check when piece.type === 'T'
- **Not tracking last action:** Simple drops into T-shaped holes should not count as T-spins
- **Ignoring wall boundaries:** Walls and floor count as occupied for corner checks
- **Hard-coding corner positions:** Corners change based on rotation state
- **Breaking B2B on non-clearing T-spins:** T-spins with 0 lines cleared don't break B2B chain

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Corner occupancy logic | Manual coordinate math per rotation | Lookup table with rotation-relative offsets | 4 rotation states with different corner positions |
| T-piece center tracking | Recalculate from shape array | Use fixed center offset in piece definition | Center must stay consistent across rotations |
| Wall kick detection | Custom rotation collision system | Extend existing wall kick implementation | Already implemented in Phase 1 (SRS) |
| Visual indicator timing | Custom animation system | Reuse existing combo indicator pattern | Consistent with B2B/combo display |

**Key insight:** T-spin detection builds on existing rotation and collision systems. The main additions are corner checking and score calculation, not new infrastructure.

## Common Pitfalls

### Pitfall 1: False Positives from Drops
**What goes wrong:** Player drops T-piece into T-shaped hole, gets T-spin credit without rotating
**Why it happens:** Not tracking whether last action was rotation vs movement/drop
**How to avoid:** Set lastAction flag in rotatePiece(), movePiece(), and gravity drop. Only detect T-spin if lastAction === 'rotation'
**Warning signs:** T-spins triggering when player doesn't rotate piece

### Pitfall 2: Incorrect Corner Classification
**What goes wrong:** Mini T-spins classified as full or vice versa
**Why it happens:** Front/back corners switch with rotation state, hard-coded positions wrong
**How to avoid:** Use lookup table mapping rotation state to front corner indices
**Warning signs:** T-spin mini showing as full T-spin in certain rotations

### Pitfall 3: Wall Boundaries Not Counted
**What goes wrong:** T-spins against walls not detected
**Why it happens:** Only checking board cells, not boundary conditions
**How to avoid:** Consider x < 0, x >= COLS, y >= board.length as occupied
**Warning signs:** T-spins work in center board but not near edges

### Pitfall 4: Center Position Miscalculation
**What goes wrong:** Corner positions offset from actual T-piece center
**Why it happens:** T-piece shapes don't have center at same position in bounding box
**How to avoid:** Verify T-piece rotates around same grid cell across all 4 rotations
**Warning signs:** Corner checks failing in specific rotations but working in others

### Pitfall 5: Breaking B2B on Zero-Line T-Spins
**What goes wrong:** Player performs T-spin without clearing lines, loses B2B chain
**Why it happens:** Setting b2bActive = false on all non-Tetris clears
**How to avoid:** Only break B2B if no lines cleared AND not a T-spin rotation
**Warning signs:** B2B chain breaks when experimenting with T-spin setups

### Pitfall 6: Wall Kick Upgrade Not Applied
**What goes wrong:** Valid full T-spins classified as mini
**Why it happens:** Not checking if last rotation used the 1x2 wall kick offset
**How to avoid:** Track which wall kick offset was used, check if abs(x) + abs(y) === 3
**Warning signs:** Certain advanced T-spin setups (STSD) showing as mini instead of full

## Code Examples

### Complete T-Spin Detection Integration

```javascript
let lastAction = null;
let lastKickOffset = null;

function rotatePiece() {
    if (!currentPiece) return false;

    const newRotation = (currentPiece.rotation + 1) % 4;
    const kicks = getWallKicks(currentPiece.type, currentPiece.rotation, newRotation);

    for (const kick of kicks) {
        const newX = currentPiece.x + kick[0];
        const newY = currentPiece.y - kick[1];

        if (isValidPosition(board, currentPiece.type, newX, newY, newRotation)) {
            currentPiece.x = newX;
            currentPiece.y = newY;
            currentPiece.rotation = newRotation;
            lastAction = 'rotation';
            lastKickOffset = kick;
            if (isLocking) {
                lockCounter = 0;
            }
            return true;
        }
    }
    return false;
}

function movePiece(dx, dy) {
    if (!currentPiece) return false;

    const newX = currentPiece.x + dx;
    const newY = currentPiece.y + dy;

    if (isValidPosition(board, currentPiece.type, newX, newY, currentPiece.rotation)) {
        currentPiece.x = newX;
        currentPiece.y = newY;
        lastAction = 'movement';
        lastKickOffset = null;
        if (isLocking && (dx !== 0 || dy < 0)) {
            lockCounter = 0;
        }
        return true;
    }
    return false;
}

function getTSpinCorners(piece) {
    const cx = piece.x + 1;
    const cy = piece.y + 1;

    const corners = [
        [cx - 1, cy - 1],
        [cx + 1, cy - 1],
        [cx - 1, cy + 1],
        [cx + 1, cy + 1]
    ];

    const frontCornersMap = [
        [0, 1],
        [1, 3],
        [2, 3],
        [0, 2]
    ];

    return {
        corners: corners,
        frontIndices: frontCornersMap[piece.rotation]
    };
}

function detectTSpin(piece, board, lastAction, lastKickOffset) {
    if (piece.type !== 'T') return null;
    if (lastAction !== 'rotation') return null;

    const { corners, frontIndices } = getTSpinCorners(piece);

    let occupiedCount = 0;
    let frontOccupiedCount = 0;

    corners.forEach((corner, idx) => {
        const [x, y] = corner;
        const isOccupied = (
            x < 0 ||
            x >= COLS ||
            y >= board.length ||
            (y >= 0 && board[y][x])
        );

        if (isOccupied) {
            occupiedCount++;
            if (frontIndices.includes(idx)) {
                frontOccupiedCount++;
            }
        }
    });

    if (occupiedCount < 3) return null;

    const isMini = frontOccupiedCount < 2;

    const isWallKickUpgrade = lastKickOffset &&
        (Math.abs(lastKickOffset[0]) + Math.abs(lastKickOffset[1]) === 3);

    if (isMini && !isWallKickUpgrade) {
        return 'mini';
    } else {
        return 'full';
    }
}

function lockPieceToBoard() {
    if (!currentPiece) return;

    const tSpinType = detectTSpin(currentPiece, board, lastAction, lastKickOffset);

    updatePiecePlaced();
    board = lockPiece(board, currentPiece);
    const lines = checkLines(board);

    if (lines.length > 0) {
        combo++;
        updateComboStats(combo);

        let isDifficultClear = lines.length === 4 || tSpinType !== null;
        const hasB2bBonus = b2bActive && isDifficultClear;

        pendingScoreCalc = {
            linesCleared: lines.length,
            comboValue: combo,
            hasB2bBonus: hasB2bBonus,
            tSpinType: tSpinType
        };

        b2bActive = isDifficultClear;
        clearingLines = lines;
        clearingTimer = 100;
        currentPiece = null;
    } else {
        if (tSpinType === null) {
            combo = 0;
        }
        spawnPiece();
    }

    lastAction = null;
    lastKickOffset = null;
}
```

### T-Spin Score Calculation

```javascript
function calculateTSpinScore(tSpinType, linesCleared, level) {
    const baseScores = {
        mini: [100, 200, 400],
        full: [400, 800, 1200, 1600]
    };

    const lineIndex = linesCleared;
    const baseScore = baseScores[tSpinType][lineIndex] || 0;

    return baseScore * level;
}

function update(deltaTime) {
    if (clearingLines.length > 0) {
        clearingTimer -= deltaTime;
        if (clearingTimer <= 0) {
            const result = clearLines(board, clearingLines);
            board = result.board;

            let scoreToAdd = 0;

            if (pendingScoreCalc && pendingScoreCalc.tSpinType) {
                scoreToAdd = calculateTSpinScore(
                    pendingScoreCalc.tSpinType,
                    result.linesCleared,
                    level
                );

                if (pendingScoreCalc.hasB2bBonus) {
                    scoreToAdd = Math.floor(scoreToAdd * 1.5);
                    incrementB2bCount();
                }
            } else {
                scoreToAdd = result.linesCleared * pointsPerRow;

                if (pendingScoreCalc && pendingScoreCalc.hasB2bBonus && result.linesCleared === 4) {
                    scoreToAdd = Math.floor(scoreToAdd * 1.5);
                    incrementB2bCount();
                }
            }

            let comboBonus = 0;
            if (pendingScoreCalc && pendingScoreCalc.comboValue > 1) {
                comboBonus = 50 * (pendingScoreCalc.comboValue - 1) * level;
            }

            score += scoreToAdd + comboBonus;
            pendingScoreCalc = null;
            stats.score = score;
            stats.lines += result.linesCleared;
            checkLevelUp();
            clearingLines = [];
            spawnPiece();
        }
        return;
    }
}
```

### Visual Indicator Display

```javascript
function drawTSpinIndicator(tSpinType, linesCleared) {
    if (!tSpinType) return;

    const ctx = canvas.getContext('2d');
    ctx.save();

    ctx.font = 'bold 24px monospace';
    ctx.fillStyle = '#ff00ff';
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 3;

    let text = tSpinType === 'mini' ? 'T-Spin Mini' : 'T-Spin';
    if (linesCleared > 0) {
        const lineNames = ['', 'Single', 'Double', 'Triple'];
        text += ' ' + lineNames[linesCleared];
    }

    const x = canvas.width - 200;
    const y = 150;

    ctx.strokeText(text, x, y);
    ctx.fillText(text, x, y);

    ctx.restore();
}

function drawComboIndicator(combo, b2bActive, tSpinType, tSpinLines) {
    const ctx = canvas.getContext('2d');
    ctx.save();

    const x = canvas.width - 200;
    let y = 100;

    if (tSpinType && performance.now() % 1000 < 500) {
        drawTSpinIndicator(tSpinType, tSpinLines);
        y += 40;
    }

    if (combo > 1) {
        ctx.font = 'bold 20px monospace';
        ctx.fillStyle = '#ffff00';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        const comboText = combo + ' Combo';
        ctx.strokeText(comboText, x, y);
        ctx.fillText(comboText, x, y);
        y += 30;
    }

    if (b2bActive) {
        ctx.font = 'bold 18px monospace';
        ctx.fillStyle = '#ff8800';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        const b2bText = 'Back-to-Back';
        ctx.strokeText(b2bText, x, y);
        ctx.fillText(b2bText, x, y);
    }

    ctx.restore();
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Immobility test (can't move any direction) | 3-corner T rule | 2001 (Tetris Worlds) | More accurate detection, fewer false positives |
| No mini vs full distinction | Front/back corner classification | 2006 (Tetris Guideline) | Balanced scoring, rewards proper technique |
| T-spins don't count for B2B | T-spins are difficult clears | Post-2006 Guideline | Increased strategic value of T-spins |
| Wall kicks excluded from T-spins | Wall kick upgrade rule (1x2 offset) | Modern Guideline | Enables advanced techniques like STSD |
| No rotation requirement | Last action must be rotation | 2001+ | Prevents cheese from simple drops |

**Deprecated/outdated:**
- Immobility-only detection: Produces false positives in enclosed spaces, replaced by 3-corner rule
- 3-corner T no kick: Too restrictive, excludes valid advanced techniques
- Tetris Worlds GBA wall treatment: Walls not counted as occupied, inconsistent with modern games

## Open Questions

1. **Visual indicator duration**
   - What we know: Indicators should flash/display when T-spin detected
   - What's unclear: Exact duration (500ms? 1000ms? until next piece spawns?)
   - Recommendation: Use 1000ms duration consistent with combo indicator timing

2. **T-spin with 0 lines cleared**
   - What we know: Awards 100 points (mini) or 400 points (full), doesn't break B2B
   - What's unclear: Should it trigger visual indicator?
   - Recommendation: Show indicator for all T-spins to teach players the mechanic

3. **Multiple simultaneous indicators**
   - What we know: T-spin + B2B + combo can all trigger together
   - What's unclear: Optimal layout to avoid overlap
   - Recommendation: Stack vertically in sidebar: T-spin (top), combo (middle), B2B (bottom)

## Sources

### Primary (HIGH confidence)
- [Hard Drop Tetris Wiki - T-Spin](https://harddrop.com/wiki/T-Spin) - 3-corner rule algorithm, mini vs full detection
- [TetrisWiki - T-Spin](https://tetris.wiki/T-Spin) - Official Guideline scoring, B2B rules
- [Hard Drop Tetris Wiki - SRS](https://harddrop.com/wiki/SRS) - T-piece rotation center, wall kick offsets
- [TetrisWiki - Super Rotation System](https://tetris.wiki/Super_Rotation_System) - SRS wall kick implementation

### Secondary (MEDIUM confidence)
- [Katy's Code - Coding for T-Spins](https://katyscode.wordpress.com/2012/10/13/tetris-aside-coding-for-t-spins/) - Implementation details, edge cases
- [TETRIS-FAQ - T-Spin Guide](https://winternebs.github.io/TETRIS-FAQ/tspin/) - Detection methods, scoring
- [FOUR - T-Spin](https://four.lol/srs/t-spin/) - Modern implementation patterns

### Tertiary (LOW confidence)
- General WebSearch results for implementation examples - marked for validation

## Metadata

**Confidence breakdown:**
- Standard algorithm (3-corner T rule): HIGH - Verified across multiple authoritative sources (Hard Drop Wiki, TetrisWiki)
- Scoring values: HIGH - Consistent across official documentation
- Mini vs full classification: HIGH - Front/back corner method documented in Guideline
- Wall kick upgrade: MEDIUM - Mentioned in sources but less detail on exact implementation
- Visual indicator specifics: LOW - General patterns found, but exact timing/positioning not standardized

**Research date:** 2026-02-05
**Valid until:** 2026-03-05 (30 days - stable mechanic, unlikely to change)
