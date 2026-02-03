# Stack Research: Tetris Twist v2.0 Enhancements

**Project:** Tetris Twist
**Researched:** 2026-02-03
**Scope:** Stack additions for v2.0 enhanced features
**Confidence:** HIGH

## v1.0 Validated Stack (No Changes Needed)

| Technology | Purpose | Status |
|------------|---------|--------|
| Vanilla JavaScript (ES6+) | Game logic, UI | Validated |
| HTML5 Canvas | Game rendering | Validated |
| CSS3 | Admin UI, themes | Validated |
| BroadcastChannel API | Tab sync | Validated |
| requestAnimationFrame | 60 FPS game loop | Validated |

**Constraint:** No external dependencies per project guidelines.

## v2.0 Stack Additions

All v2.0 features can be implemented using native browser APIs that integrate cleanly with the existing vanilla JS stack.

### Audio (Sound Effects & Music)

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| Web Audio API | Sound effects playback | Low latency, unlimited concurrent sounds, precise timing control |
| AudioContext | Audio environment | Core API for sound management |
| AudioBufferSourceNode | Playing pre-loaded effects | Designed for short game audio clips |
| GainNode | Volume control | Per-sound and master volume |

**Browser Support:** Baseline widely available since April 2021. 92% browser compatibility score in 2026.

**Why Web Audio API over HTML5 Audio Element:**
- HTML5 Audio was "good for long audio like music or a podcast, but ill-suited for the demands of gaming" with looping problems and concurrent sound limits
- Web Audio API provides "low-latency playback for games" and supports "high number of concurrent effects" without the 32-64 sound ceiling
- Fieldrunners HTML5 port "encountered issues with audio playback with the Audio tag and early on decided to focus on the Web Audio API instead"
- Web Audio's buffer source is designed specifically for "short audio clips for games and sound effects"

**Integration Pattern:**

```javascript
const audioContext = new (window.AudioContext || window.webkitAudioContext)();
const masterGain = audioContext.createGain();
masterGain.connect(audioContext.destination);

async function loadSound(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    return await audioContext.decodeAudioData(arrayBuffer);
}

function playSound(buffer, volume = 1.0) {
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
    const source = audioContext.createBufferSource();
    const gain = audioContext.createGain();
    source.buffer = buffer;
    gain.gain.value = volume;
    source.connect(gain);
    gain.connect(masterGain);
    source.start(0);
}
```

**Autoplay Policy:** Requires user interaction before first sound. Resume audioContext on first keypress or click. Already handled by existing game input system.

**Sound Asset Formats:**

| Format | Support | Use |
|--------|---------|-----|
| OGG | Wide | Sound effects (better compression) |
| MP3 | Universal | Music tracks (if added) |
| WAV | Universal | Fallback for effects |

**Recommendation:** Use OGG for effects (smaller), MP3 for music. Load with fetch + decodeAudioData.

**Integration Point:** Create new js/audio.js module. Call playSound() from existing game events (line clear, piece lock, rotation, etc.).

### Keyboard Remapping

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| KeyboardEvent.code | Physical key detection | Layout-agnostic, perfect for remapping |
| localStorage | Persistent key bindings | 5-10MB storage, survives browser restart |
| JSON.stringify/parse | Serialize key mappings | Simple object persistence |

**Why event.code:**
- "Useful when you want to handle keys based on their physical positions on the input device rather than the characters associated with those keys; this is especially common when writing code to handle input for games"
- event.code is always the same regardless of keyboard layout (KeyW is physical W key position)
- "If you are building a game that uses WASD keys, then use .code, as it will work across any keyboard layout"
- Existing codebase already uses event.code ('ArrowUp', 'KeyC', 'Space', etc.) in js/input.js

**Storage Pattern:**

```javascript
const DEFAULT_BINDINGS = {
    moveLeft: ['ArrowLeft'],
    moveRight: ['ArrowRight'],
    moveDown: ['ArrowDown'],
    rotate: ['ArrowUp'],
    hardDrop: ['Space'],
    hold: ['KeyC', 'ShiftLeft', 'ShiftRight'],
    pause: ['KeyP']
};

function loadKeyBindings() {
    const stored = localStorage.getItem('tetris_keybindings');
    return stored ? JSON.parse(stored) : DEFAULT_BINDINGS;
}

function saveKeyBindings(bindings) {
    localStorage.setItem('tetris_keybindings', JSON.stringify(bindings));
}
```

**Integration Point:** Modify existing getInput() in js/input.js to use configurable bindings instead of hardcoded codes.

**Security:** No sensitive data. XSS risk irrelevant for key bindings (no authentication tokens).

**Alternative Considered:** event.key - REJECTED because it changes with keyboard layout (AZERTY vs QWERTY produces different characters).

### Session Statistics Tracking

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| JavaScript Object | Session stat tracking | Fast, no serialization overhead |
| localStorage | Optional persistence | Preserve stats across page refresh |
| Performance.now() | High-precision timing | Accurate session duration |

**Why In-Memory:**
- Zero overhead during gameplay
- Stats updated every frame without storage I/O
- Session-scoped by default (v1.0 is session-only gameplay per PROJECT.md)

**Why Optional localStorage:**
- Preserve stats on accidental refresh
- Allow stats history across sessions
- User can clear localStorage to reset

**Data Structure:**

```javascript
const sessionStats = {
    startTime: performance.now(),
    totalPieces: 0,
    piecesByType: { I: 0, O: 0, T: 0, S: 0, Z: 0, J: 0, L: 0 },
    linesCleared: 0,
    linesByType: { single: 0, double: 0, triple: 0, tetris: 0 },
    tSpins: 0,
    tSpinMinis: 0,
    combos: 0,
    maxCombo: 0,
    highestLevel: 1,
    peakScore: 0
};
```

**Persistence Pattern:**

```javascript
function saveStats() {
    localStorage.setItem('tetris_session_stats', JSON.stringify(sessionStats));
}

function loadStats() {
    const stored = localStorage.getItem('tetris_session_stats');
    return stored ? JSON.parse(stored) : createDefaultStats();
}

setInterval(saveStats, 5000);
```

**Integration Point:** Update stats in existing main.js functions (spawnPiece, lockPieceToBoard, clearLines).

**Alternative Considered:** sessionStorage - REJECTED because stats lost on browser close. localStorage allows optional persistence without forcing it.

### T-Spin Detection

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| 3-Corner T Algorithm | T-spin validation | Guideline-compliant, used in Tetris DS and modern games |
| No external library | Algorithmic check | Simple logic, no dependencies needed |

**Algorithm Choice:** 3-Corner T (Tetris DS standard)

**Detection Criteria:**
1. Last maneuver must be rotation (not movement)
2. Three of four diagonal corners around T center are occupied
3. Full T-spin: Two front corners + at least one back corner occupied
4. Mini T-spin: Only one front corner + two back corners occupied
5. Exception: Kick moves 1x2 blocks = full T-spin regardless

**Why 3-Corner T:**
- "Used in Tetris DS and other SRS based games"
- Clear distinction between full T-spin and mini
- Compatible with existing SRS rotation system (wall kicks already implemented in js/pieces.js)

**Integration Point:** Add detection in lockPieceToBoard() after rotation flag tracking.

**Alternatives Considered:**
- Immobile detection (The New Tetris) - REJECTED as outdated
- 3-corner T no kick (Tetris Evolution) - REJECTED as less rewarding
- TGM3 simple rotation check - REJECTED as too permissive

### Combo Multipliers

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| JavaScript counter | Track consecutive clears | No dependencies needed |
| No external library | Simple arithmetic | Increment/reset pattern |

**State Management:**

```javascript
let comboCount = 0;
let lastClearTime = 0;

function onLinesClear(linesCleared) {
    if (linesCleared > 0) {
        comboCount++;
        lastClearTime = performance.now();
    }
}

function onPieceLock() {
    if (performance.now() - lastClearTime > 500) {
        comboCount = 0;
    }
}
```

**Integration Point:** Extend existing clearLines logic in js/main.js.

### Additional Themes

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| JavaScript Objects | Theme definitions | Already used in js/themes.js |
| CSS3 Variables | Color schemes | Optional for advanced theming |

**Current:** 3 themes (Classic, Neon, Retro) in js/themes.js
**Target:** 5+ themes for v2.0

**Integration Point:** Add theme objects to THEMES in js/themes.js. No stack changes needed.

## What NOT to Add

| Library | Why Rejected |
|---------|--------------|
| Howler.js / Tone.js | Web Audio API sufficient, violates no-dependencies constraint |
| Mousetrap / Hotkeys.js | Native KeyboardEvent.code sufficient, no framework needed |
| Store.js / localForage | Native localStorage sufficient, no abstraction needed |
| Any npm package | Project constraint: minimal dependencies |

## New Dependencies Summary

**New External Dependencies:** 0

**New Native APIs:**
- Web Audio API (AudioContext, AudioBufferSourceNode, GainNode)
- localStorage (already available, new usage for key bindings and stats)
- Performance.now() (already available, new usage for stats timing)

**Existing APIs (no changes):**
- BroadcastChannel
- requestAnimationFrame
- Canvas 2D Context
- KeyboardEvent.code (already in use, extended usage)

## Browser Compatibility

All APIs baseline since 2021-2022. Compatible with:
- Chrome 89+
- Firefox 88+
- Safari 14.1+
- Edge 89+

No polyfills needed for target browsers (modern evergreen browsers).

## Integration Checklist

- [ ] Create audio manager module (js/audio.js) using Web Audio API
- [ ] Load sound assets (OGG/MP3 files)
- [ ] Add key remapping UI in admin panel
- [ ] Extend input.js to use configurable bindings
- [ ] Add session stats tracking to game loop
- [ ] Implement 3-corner T detection in rotation/lock logic
- [ ] Add combo counter to line clear logic
- [ ] Create 2+ additional theme definitions
- [ ] Update admin panel for stats display
- [ ] Add localStorage persistence for bindings and stats

## File Structure Updates

```
js/
├── game/
│   ├── audio.js         # NEW: Web Audio API wrapper
│   ├── stats.js         # NEW: Session statistics tracking
│   └── input.js         # MODIFIED: Configurable key bindings
├── shared/
│   └── themes.js        # MODIFIED: Add 2+ themes
```

## Sources

**Web Audio API:**
- [Audio Streaming with Web Audio API 2026](https://medium.com/@coders.stop/audio-streaming-with-web-audio-api-making-sound-actually-sound-good-on-the-web-65915047736f)
- [HTML5 audio and Web Audio API are BFFs](https://developer.chrome.com/blog/html5-audio-and-the-web-audio-api-are-bffs)
- [Web Audio API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [Using the Web Audio API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Using_Web_Audio_API)
- [Case Study - HTML5 Game with Web Audio](https://web.dev/articles/webaudio-fieldrunners)
- [Can I Use - Web Audio API](https://caniuse.com/audio-api)
- [Web Audio API Browser Compatibility](https://www.lambdatest.com/web-technologies/audio-api)

**Keyboard Events:**
- [Keyboard: keydown and keyup](https://javascript.info/keyboard-events)
- [Should I use e.code or e.key](https://blog.andri.co/022-should-i-use-ecode-or-ekey-when-handling-keyboard-events/)
- [KeyboardEvent: code property - MDN](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/code)
- [Can I Use - KeyboardEvent.code](https://caniuse.com/keyboardevent-code)

**localStorage:**
- [JavaScript LocalStorage Guide](https://www.meticulous.ai/blog/localstorage-complete-guide)
- [LocalStorage, sessionStorage](https://javascript.info/localstorage)
- [Mastering localStorage in JavaScript](https://medium.com/dev-simplified/mastering-localstorage-in-javascript-74c65b93fecf)
- [Browser Storage Guide](https://dev.to/aneeqakhan/a-developers-guide-to-browser-storage-local-storage-session-storage-and-cookies-4c5f)

**T-Spin Detection:**
- [T-Spin - TetrisWiki](https://tetris.wiki/T-Spin)
- [Tetris Aside: Coding for T-Spins](https://katyscode.wordpress.com/2012/10/13/tetris-aside-coding-for-t-spins/)
- [T-Spin - Hard Drop Tetris Wiki](https://harddrop.com/wiki/T-Spin)
- [T-Spin Guide](https://winternebs.github.io/TETRIS-FAQ/tspin/)
