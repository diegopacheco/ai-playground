# Architecture Research: Tetris Twist v2.0

**Researched:** 2026-02-03
**Scope:** T-spin detection, combo multipliers, session statistics, additional themes, sound effects, keyboard remapping
**Confidence:** HIGH

## Current Architecture Summary

Tetris Twist v1.0 uses a modular vanilla JavaScript architecture with:

- **Game tab**: main.js (game loop, state machine), board.js (grid, collision, line clearing), pieces.js (tetrominoes, rotation, wall kicks), render.js (canvas drawing), input.js (keyboard handling with DAS/ARR), themes.js (3 color schemes)
- **Admin tab**: admin.html (control panel UI), admin.js (BroadcastChannel sender)
- **State machine**: GameState enum (PLAYING, FROZEN, PAUSED, GAME_OVER)
- **Sync mechanism**: BroadcastChannel for real-time tab communication
- **Canvas rendering**: Direct 2D context drawing at 30x30 cell size

Key architectural patterns validated in v1.0:
- Single global state in main.js
- Pure functions in board.js and pieces.js
- Render functions called from game loop
- Input polled via getInput() in processInput()
- Message-based config updates via BroadcastChannel

## New Components Needed for v2.0

### 1. T-Spin Detection Module (tspin.js)

**Purpose:** Detect T-spin moves and calculate bonus points

**Integration point:** Called after rotatePiece() in main.js

**Core function:**
```javascript
function detectTSpin(board, piece, lastMove, rotation) {
  if (piece.type !== 'T') return { isTSpin: false };
  if (lastMove !== 'ROTATE') return { isTSpin: false };

  const corners = checkCorners(board, piece);
  const filledCorners = corners.filter(c => c).length;
  const pointingCorners = checkPointingCorners(corners, rotation);

  if (filledCorners >= 3 && pointingCorners >= 2) {
    return { isTSpin: true, type: 'FULL' };
  }
  if (filledCorners >= 3) {
    return { isTSpin: true, type: 'MINI' };
  }
  return { isTSpin: false };
}
```

**Data flow:**
```
rotatePiece() → piece rotates → detectTSpin() → returns { isTSpin, type }
lockPieceToBoard() → if isTSpin → apply bonus multiplier to score
```

**State changes required:**
- Add `lastMove` tracking to main.js ('LEFT', 'RIGHT', 'DOWN', 'ROTATE', 'HARD_DROP')
- Add `tSpinDetected` flag for rendering visual feedback

**Reference:** 3-corner T algorithm with pointing-side corner validation (Tetris DS standard)

### 2. Combo System Module (combo.js)

**Purpose:** Track consecutive line clears and apply multipliers

**Integration point:** Called in lockPieceToBoard() after line clearing

**Core function:**
```javascript
function updateCombo(previousComboCount, linesCleared) {
  if (linesCleared === 0) {
    return { comboCount: 0, comboBonus: 0 };
  }

  const newCount = previousComboCount + 1;
  const comboBonus = calculateComboBonus(newCount, linesCleared);

  return { comboCount: newCount, comboBonus };
}

function calculateComboBonus(comboCount, linesCleared) {
  return 50 * comboCount * linesCleared;
}
```

**Data flow:**
```
lockPieceToBoard() → clearLines() → lines cleared count
  → updateCombo(currentCombo, linesCleared)
  → returns { comboCount, comboBonus }
  → add comboBonus to score
  → update comboCount in state
```

**State changes required:**
- Add `comboCount` to main.js state (initialized to 0)
- Reset comboCount when linesCleared === 0

**Rendering integration:**
- Add drawCombo(comboCount) to render.js
- Display in sidebar below score (if comboCount > 0)

**Reference:** Standard Tetris scoring formula: 50 × combo count × level

### 3. Session Statistics Module (stats.js)

**Purpose:** Track cumulative session metrics

**Integration point:** Event-driven updates throughout main.js

**Data structure:**
```javascript
const sessionStats = {
  startTime: Date.now(),
  totalPieces: 0,
  totalLines: 0,
  totalScore: 0,
  pieceDistribution: { I: 0, O: 0, T: 0, S: 0, Z: 0, J: 0, L: 0 },
  tSpins: { full: 0, mini: 0 },
  maxCombo: 0,
  currentPlayTime: 0
};
```

**Update triggers:**
- `spawnPiece()` → increment totalPieces, pieceDistribution
- `clearLines()` → increment totalLines
- `score` changes → update totalScore
- `detectTSpin()` returns true → increment tSpins
- `updateCombo()` → track maxCombo
- `update(deltaTime)` → accumulate currentPlayTime (only when PLAYING)

**Functions:**
```javascript
function resetStats()
function getStats()
function updateStat(key, value)
```

**Rendering integration:**
- Optional stats panel overlay (toggle with 'S' key)
- Display in admin panel via BroadcastChannel
- Format playtime as MM:SS

**No external dependencies:** Pure JavaScript object

### 4. Audio Manager Module (audio.js)

**Purpose:** Play sound effects for game events

**Integration point:** Event-driven calls from main.js and board.js

**Architecture pattern:** AudioContext with pre-loaded sound buffers

**Core structure:**
```javascript
const audioManager = {
  context: null,
  sounds: {},
  gainNode: null,
  enabled: true,

  init() {
    this.context = new AudioContext();
    this.gainNode = this.context.createGain();
    this.gainNode.connect(this.context.destination);
    this.loadSounds();
  },

  play(soundName) {
    if (!this.enabled || !this.sounds[soundName]) return;

    const source = this.context.createBufferSource();
    source.buffer = this.sounds[soundName];
    source.connect(this.gainNode);
    source.start(0);
  },

  setVolume(level) {
    this.gainNode.gain.value = level;
  }
};
```

**Sound events:**
- `move` - piece moves left/right
- `rotate` - piece rotates
- `softDrop` - piece moves down
- `hardDrop` - piece hard drops
- `lock` - piece locks to board
- `lineClear` - lines cleared (vary by count)
- `tSpin` - T-spin detected
- `levelUp` - level increases
- `gameOver` - game ends

**Sound generation approach:**
- Use Web Audio API OscillatorNode for synthesized sounds (no audio file dependencies)
- Create simple 8-bit style beeps with different frequencies
- Keep sounds under 200ms for responsiveness

**Integration points:**
```
movePiece() → audioManager.play('move')
rotatePiece() → audioManager.play('rotate')
hardDrop() → audioManager.play('hardDrop')
lockPieceToBoard() → audioManager.play('lock')
clearLines() → audioManager.play('lineClear' + count)
detectTSpin() → audioManager.play('tSpin')
checkLevelUp() → audioManager.play('levelUp')
gameState = GAME_OVER → audioManager.play('gameOver')
```

**User control:**
- Mute toggle (keyboard 'M')
- Volume slider in admin panel
- Sync mute state via BroadcastChannel

**Browser compatibility:** AudioContext requires user gesture - initialize on first keypress

**Reference:** Web Audio API best practices - single AudioContext, GainNode for volume control, pre-loaded buffers

### 5. Key Remapping Module (keybindings.js)

**Purpose:** Allow users to customize keyboard controls

**Integration point:** Replaces hardcoded keys in input.js

**Data structure:**
```javascript
const defaultBindings = {
  moveLeft: ['ArrowLeft'],
  moveRight: ['ArrowRight'],
  softDrop: ['ArrowDown'],
  rotate: ['ArrowUp', 'KeyX'],
  hardDrop: ['Space'],
  hold: ['KeyC', 'ShiftLeft', 'ShiftRight'],
  pause: ['KeyP', 'Escape']
};

let currentBindings = { ...defaultBindings };
```

**Core functions:**
```javascript
function setBinding(action, keyCodes)
function getBinding(action)
function isKeyBound(keyCode, action)
function saveBindings()
function loadBindings()
function resetToDefaults()
```

**Integration with input.js:**
- Modify getInput() to check currentBindings instead of hardcoded keys
- Use keyCode lookup: `isKeyBound(e.code, 'moveLeft')`

**UI for remapping:**
- Add settings modal (toggle with 'K')
- Click action → wait for keypress → bind key
- Conflict detection (warn if key already bound)
- Save to localStorage for persistence

**Data flow:**
```
User presses 'K' → show settings modal
User clicks "Move Left" → capture mode
User presses new key → setBinding('moveLeft', [e.code])
saveBindings() → localStorage.setItem('tetris-bindings', JSON.stringify(currentBindings))
getInput() → checks currentBindings for all actions
```

**localStorage schema:**
```javascript
localStorage.getItem('tetris-bindings')
// Returns: {"moveLeft":["KeyA"],"moveRight":["KeyD"],...}
```

**Validation:**
- Prevent unbinding all keys for an action
- Allow multiple keys per action (e.g., both ArrowUp and KeyX for rotate)
- Prevent binding modifier-only keys (Shift, Ctrl, Alt alone)

**Reference:** Game input architecture pattern - use e.code (physical position) not e.key (character value) for consistent cross-keyboard behavior

### 6. Theme Expansion (themes.js enhancement)

**Purpose:** Add 2+ additional themes for total of 5+

**Integration point:** Extend existing THEMES object in themes.js

**No new module needed:** Extend existing themes.js

**New themes to add:**
- `ocean` - Blue/teal underwater palette
- `sunset` - Orange/purple gradient colors
- `matrix` - Green monochrome terminal aesthetic

**Data structure (existing):**
```javascript
THEMES.ocean = {
  name: 'Ocean',
  colors: {
    background: '#001f3f',
    grid: '#0074D9',
    sidebar: '#003366',
    I: '#7FDBFF',
    O: '#39CCCC',
    T: '#3D9970',
    S: '#2ECC40',
    Z: '#01FF70',
    J: '#0074D9',
    L: '#001f3f'
  }
};
```

**Admin panel integration:**
- Add new theme radio buttons to admin.html
- Update THEME_ORDER array in themes.js
- No code changes needed (existing theme system handles it)

**Build effort:** LOW - just data addition, no architectural changes

## Integration Points Summary

| Feature | Modifies Existing | New Files | Integration Complexity |
|---------|-------------------|-----------|------------------------|
| T-spin detection | main.js, render.js | tspin.js | MEDIUM - requires lastMove tracking |
| Combo system | main.js, render.js | combo.js | LOW - simple state addition |
| Session stats | main.js, render.js, admin.js | stats.js | MEDIUM - many update points |
| Audio manager | main.js, board.js | audio.js | MEDIUM - AudioContext initialization |
| Key remapping | input.js | keybindings.js | HIGH - replaces core input logic |
| Additional themes | themes.js | none | LOW - data only |

## Data Flow Changes

### Extended Game State

```javascript
let tSpinResult = null;
let comboCount = 0;
let lastMove = null;
let sessionStats = { ... };

const GameState = Object.freeze({
  PLAYING: 'PLAYING',
  FROZEN: 'FROZEN',
  PAUSED: 'PAUSED',
  GAME_OVER: 'GAME_OVER'
});
```

### New Message Types (BroadcastChannel)

```javascript
{ type: 'AUDIO_MUTE', muted: true }
{ type: 'AUDIO_VOLUME', volume: 0.7 }
{ type: 'STATS_UPDATE', stats: sessionStats }
```

### Score Calculation Enhancement

```javascript
function calculateScore(linesCleared, tSpinResult, comboCount, level) {
  let baseScore = linesCleared * pointsPerRow;

  if (tSpinResult && tSpinResult.isTSpin) {
    const multiplier = tSpinResult.type === 'FULL' ? 2.0 : 1.5;
    baseScore *= multiplier;
  }

  const comboBonus = 50 * comboCount * level;

  return baseScore + comboBonus;
}
```

## Suggested Build Order

Build features in dependency order to minimize rework:

### Phase 1: Foundation Enhancements
1. **Additional themes** (LOW complexity, no dependencies)
   - Add 2-3 new theme objects to themes.js
   - Update admin.html radio buttons
   - Test theme switching

2. **Session statistics** (MEDIUM complexity, foundation for other features)
   - Create stats.js module
   - Add stat tracking to spawnPiece(), clearLines()
   - Add stats display to render.js sidebar
   - Wire to admin panel via BroadcastChannel

### Phase 2: Scoring Enhancements
3. **Combo system** (LOW complexity, needed before T-spin scoring)
   - Create combo.js module
   - Add comboCount state to main.js
   - Integrate with clearLines() logic
   - Add combo display to render.js

4. **T-spin detection** (MEDIUM complexity, depends on combo for scoring)
   - Create tspin.js module
   - Add lastMove tracking to main.js
   - Integrate detectTSpin() after rotatePiece()
   - Add T-spin visual feedback to render.js
   - Update stats tracking for T-spins

### Phase 3: User Experience
5. **Audio manager** (MEDIUM complexity, independent)
   - Create audio.js module
   - Generate synthesized sounds with OscillatorNode
   - Add play() calls to game events
   - Add mute toggle and volume control
   - Wire admin panel audio controls

6. **Key remapping** (HIGH complexity, refactors input.js)
   - Create keybindings.js module
   - Refactor input.js to use binding lookups
   - Build settings modal UI
   - Implement localStorage persistence
   - Add conflict detection

## Architectural Concerns

### Performance

**Audio playback:** Multiple simultaneous sounds could cause AudioContext congestion
- **Mitigation:** Limit to 5 concurrent sounds, use sound pooling

**Stats tracking:** Frequent object updates on every piece spawn
- **Mitigation:** Stats are lightweight primitives, no concern

**Combo rendering:** Additional sidebar element
- **Mitigation:** Only render when comboCount > 0

### State Management

**Increasing global state:** Adding tSpinResult, comboCount, lastMove, sessionStats to main.js
- **Current pattern:** All state in main.js works well for single-file architecture
- **Recommendation:** Continue pattern, no refactor needed
- **Future consideration:** If state grows significantly (10+ more variables), consider state object

### Browser Compatibility

**AudioContext:** Requires HTTPS or localhost in modern browsers
- **Mitigation:** Development on localhost, production on HTTPS

**BroadcastChannel:** 95%+ browser support (2026)
- **Mitigation:** Already validated in v1.0

**localStorage:** Universal support
- **Mitigation:** No concern

### Testing Complexity

**T-spin detection:** Requires specific board configurations
- **Test approach:** Manual testing with known T-spin setups

**Combo system:** Needs consecutive line clears
- **Test approach:** Use admin panel to slow fall speed for testing

**Audio:** Difficult to automate
- **Test approach:** Manual verification of sound events

## Anti-Patterns to Avoid

### 1. AudioContext Creation on Import
**Bad:**
```javascript
const audioContext = new AudioContext();
```

**Good:**
```javascript
let audioContext = null;

function initAudio() {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
}
```

**Reason:** Browsers require user gesture before AudioContext creation

### 2. Synchronous localStorage in Game Loop
**Bad:**
```javascript
function update(deltaTime) {
  localStorage.setItem('stats', JSON.stringify(stats));
}
```

**Good:**
```javascript
function saveStats() {
  setTimeout(() => {
    localStorage.setItem('stats', JSON.stringify(stats));
  }, 0);
}

window.addEventListener('beforeunload', saveStats);
```

**Reason:** localStorage is synchronous and blocks game loop

### 3. Mutable Shared State Between Modules
**Bad:**
```javascript
export let stats = { score: 0 };
```

**Good:**
```javascript
let stats = { score: 0 };

export function getStats() {
  return { ...stats };
}

export function updateStats(key, value) {
  stats[key] = value;
}
```

**Reason:** Prevents unintended mutations, clearer data flow

### 4. Complex T-Spin Detection in Render Loop
**Bad:**
```javascript
function drawPiece(piece) {
  const tSpinResult = detectTSpin(board, piece);
  if (tSpinResult.isTSpin) { }
}
```

**Good:**
```javascript
function rotatePiece() {
  tSpinResult = detectTSpin(board, currentPiece);
}

function render() {
  if (tSpinResult && tSpinResult.isTSpin) { }
}
```

**Reason:** Detection should occur once at rotation, not every frame

## File Structure After v2.0

```
js/
├── main.js          (game loop, state machine) [MODIFIED]
├── board.js         (grid, collision, line clearing) [UNCHANGED]
├── pieces.js        (tetrominoes, rotation, wall kicks) [UNCHANGED]
├── render.js        (canvas drawing) [MODIFIED - new draw functions]
├── input.js         (keyboard handling) [MODIFIED - binding lookups]
├── themes.js        (5+ color schemes) [MODIFIED - new themes]
├── sync.js          (BroadcastChannel) [UNCHANGED]
├── admin.js         (admin panel logic) [MODIFIED - audio controls]
├── tspin.js         [NEW]
├── combo.js         [NEW]
├── stats.js         [NEW]
├── audio.js         [NEW]
└── keybindings.js   [NEW]
```

## Confidence Assessment

| Feature | Confidence | Rationale |
|---------|-----------|-----------|
| T-spin detection | HIGH | Well-documented algorithm, existing rotation system in place |
| Combo system | HIGH | Simple state tracking, standard formula |
| Session stats | HIGH | Straightforward event tracking, no dependencies |
| Audio manager | MEDIUM | Web Audio API requires user gesture handling, synthesis approach reduces complexity |
| Key remapping | MEDIUM | Input system refactor is complex, localStorage well-understood |
| Additional themes | HIGH | Existing theme system proven, just data addition |

## Sources

T-Spin Detection:
- [T-Spin - TetrisWiki](https://tetris.wiki/T-Spin)
- [T-Spin - Hard Drop Tetris Wiki](https://harddrop.com/wiki/T-Spin)
- [Tetris Aside: Coding for T-Spins | Katy's Code](https://katyscode.wordpress.com/2012/10/13/tetris-aside-coding-for-t-spins/)

Combo Systems:
- [Combo - TetrisWiki](https://tetris.wiki/Combo)
- [Scoring - TetrisWiki](https://tetris.wiki/Scoring)
- [Mechanics - Wiki for TETR.IO](https://tetrio.wiki.gg/wiki/Mechanics)

Audio:
- [Audio for Web games - Game development | MDN](https://developer.mozilla.org/en-US/docs/Games/Techniques/Audio_for_Web_Games)
- [Web Audio API best practices - Web APIs | MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices)

Input Handling:
- [Create a keyboard input mapping for a 2D javascript typescript game](https://stephendoddtech.com/blog/game-design/keyboard-event-game-input-map)
- [Handling Keyboard Input with JavaScript KeyboardEvent Properties](https://medium.com/@AlexanderObregon/handling-keyboard-input-with-javascript-keyboardevent-properties-8f558597a853)

Session Statistics:
- [GitHub - ejona86/taus: Tetris - Actually Useful Statistics](https://github.com/ejona86/taus)

State Management:
- [Javascript Game Foundations - State Management | Jake Gordon](https://codeincomplete.com/articles/javascript-game-foundations-state-management/)
- [Modern State Management in Vanilla JavaScript: 2026 Patterns and Beyond | by Orami | Medium](https://medium.com/@orami98/modern-state-management-in-vanilla-javascript-2026-patterns-and-beyond-ce00425f7ac5)
