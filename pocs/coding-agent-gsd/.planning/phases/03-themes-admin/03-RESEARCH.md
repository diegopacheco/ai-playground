# Phase 3: Themes & Admin - Research

**Researched:** 2026-02-02
**Domain:** Real-time cross-tab synchronization with BroadcastChannel API, canvas theming patterns, admin panel architecture
**Confidence:** HIGH

## Summary

This phase implements real-time theme switching and admin panel controls for a vanilla JavaScript canvas game. The core technical challenge is synchronizing state changes between two browser contexts (game tab and admin panel tab) using the BroadcastChannel API, while dynamically applying theme changes to canvas rendering without disrupting gameplay.

The standard approach uses BroadcastChannel for same-origin, cross-tab communication with a structured message protocol (type/payload pattern). Themes are defined as configuration objects containing color palettes and piece shape definitions, with the rendering layer consuming theme data rather than hardcoded values. The admin panel opens in a separate window via window.open() and communicates bidirectionally through BroadcastChannel.

Key architectural decisions center on separating theme configuration from rendering logic, implementing a message-based synchronization protocol, and ensuring theme changes apply instantly without requiring re-initialization of game state.

**Primary recommendation:** Define themes as pure configuration objects, refactor rendering to consume theme data dynamically, use BroadcastChannel with structured message protocol for real-time sync.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| BroadcastChannel API | Native Web API | Cross-tab same-origin communication | Native browser API, widely supported since March 2022, lightweight publish-subscribe pattern |
| Canvas 2D Context | Native Web API | Theme-aware rendering | Direct control of drawing styles via fillStyle/strokeStyle properties |
| window.open() | Native Web API | Admin panel in separate window | Native method for same-origin window creation with full DOM access |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Structured Clone Algorithm | Native | Message serialization for BroadcastChannel | Automatic, handles objects, arrays, primitives |
| CSS Variables | Native CSS | Optional UI theming outside canvas | When applying theme to HTML elements surrounding canvas |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| BroadcastChannel | localStorage + storage event | More complex, requires polling or event handling, persists data unnecessarily |
| BroadcastChannel | SharedWorker | Heavier, more complex setup, overkill for simple state sync |
| BroadcastChannel | WebSocket + server | Requires server infrastructure, network latency, more complex |
| window.open() | iframe | Less separation, harder to manage as separate "admin panel" UX |

**Installation:**
```bash
No dependencies required - all native Web APIs
```

## Architecture Patterns

### Recommended Project Structure
```
js/
├── main.js           # Game loop, state management
├── render.js         # Canvas rendering (theme-aware)
├── pieces.js         # Piece definitions (theme-aware)
├── themes.js         # Theme configuration objects
├── sync.js           # BroadcastChannel message handling
├── input.js          # Existing input handling
└── board.js          # Existing board logic

admin.html            # Admin panel page
js/admin/
└── controls.js       # Admin UI controls and BroadcastChannel sender
```

### Pattern 1: Theme Configuration Object
**What:** Centralized theme definition as pure data structure
**When to use:** Always - separates configuration from logic
**Example:**
```javascript
const THEMES = {
  classic: {
    name: 'Classic',
    colors: {
      background: '#0f0f1a',
      grid: '#2a2a4a',
      sidebar: '#16162a',
      I: '#00f0f0',
      O: '#f0f000',
      T: '#a000f0',
      S: '#00f000',
      Z: '#f00000',
      J: '#0000f0',
      L: '#f0a000'
    },
    pieceShapes: 'standard'
  },
  neon: {
    name: 'Neon',
    colors: {
      background: '#000000',
      grid: '#00ff00',
      sidebar: '#001100',
      I: '#00ffff',
      O: '#ffff00',
      T: '#ff00ff',
      S: '#00ff00',
      Z: '#ff0000',
      J: '#0000ff',
      L: '#ff8800'
    },
    pieceShapes: 'standard'
  },
  retro: {
    name: 'Retro',
    colors: {
      background: '#f4e4c1',
      grid: '#8b7355',
      sidebar: '#d4c4a1',
      I: '#5f9ea0',
      O: '#daa520',
      T: '#9370db',
      S: '#3cb371',
      Z: '#cd5c5c',
      J: '#4682b4',
      L: '#d2691e'
    },
    pieceShapes: 'standard'
  }
};
```

### Pattern 2: BroadcastChannel Message Protocol
**What:** Structured message format with type and payload
**When to use:** All BroadcastChannel communication
**Example:**
```javascript
const channel = new BroadcastChannel('tetris-sync');

channel.postMessage({
  type: 'THEME_CHANGE',
  payload: { themeName: 'neon' }
});

channel.postMessage({
  type: 'SPEED_CHANGE',
  payload: { dropInterval: 800 }
});

channel.postMessage({
  type: 'STATS_REQUEST',
  payload: {}
});

channel.onmessage = (event) => {
  const { type, payload } = event.data;
  switch(type) {
    case 'THEME_CHANGE':
      applyTheme(payload.themeName);
      break;
    case 'SPEED_CHANGE':
      dropInterval = payload.dropInterval;
      break;
    case 'STATS_RESPONSE':
      updateStatsDisplay(payload);
      break;
  }
};
```

### Pattern 3: Theme-Aware Rendering
**What:** Rendering functions consume current theme instead of hardcoded colors
**When to use:** All canvas drawing operations
**Example:**
```javascript
let currentTheme = THEMES.classic;

function drawGrid() {
  ctx.fillStyle = currentTheme.colors.background;
  ctx.fillRect(0, 0, COLS * CELL_SIZE, ROWS * CELL_SIZE);

  ctx.strokeStyle = currentTheme.colors.grid;
  ctx.lineWidth = 1;

  for (let col = 0; col <= COLS; col++) {
    ctx.beginPath();
    ctx.moveTo(col * CELL_SIZE, 0);
    ctx.lineTo(col * CELL_SIZE, ROWS * CELL_SIZE);
    ctx.stroke();
  }
}

function drawPiece(pieceType, x, y, rotation) {
  const piece = PIECES[pieceType];
  const shape = piece.shapes[rotation];
  ctx.fillStyle = currentTheme.colors[pieceType];

  for (let row = 0; row < shape.length; row++) {
    for (let col = 0; col < shape[row].length; col++) {
      if (shape[row][col]) {
        ctx.fillRect(
          (x + col) * CELL_SIZE + 1,
          (y + row) * CELL_SIZE + 1,
          CELL_SIZE - 2,
          CELL_SIZE - 2
        );
      }
    }
  }
}
```

### Pattern 4: Instant Theme Application
**What:** Theme changes apply immediately in next render frame
**When to use:** When receiving THEME_CHANGE messages
**Example:**
```javascript
function applyTheme(themeName) {
  if (THEMES[themeName]) {
    currentTheme = THEMES[themeName];
  }
}

channel.onmessage = (event) => {
  if (event.data.type === 'THEME_CHANGE') {
    applyTheme(event.data.payload.themeName);
  }
};
```

### Pattern 5: Admin Panel Bidirectional Communication
**What:** Admin sends commands, game sends stats back
**When to use:** Admin panel needs live game data
**Example:**
```javascript
const channel = new BroadcastChannel('tetris-sync');

channel.postMessage({
  type: 'STATS_REQUEST',
  payload: {}
});

channel.onmessage = (event) => {
  if (event.data.type === 'STATS_RESPONSE') {
    document.getElementById('game-score').textContent = event.data.payload.score;
    document.getElementById('game-level').textContent = event.data.payload.level;
  }
};

setInterval(() => {
  channel.postMessage({ type: 'STATS_REQUEST', payload: {} });
}, 1000);
```

### Anti-Patterns to Avoid
- **Hardcoded colors in render functions:** Makes theme switching impossible without code changes
- **Storing theme in localStorage:** Adds unnecessary persistence, complicates sync logic
- **Sending unstructured messages:** Makes message handling brittle and hard to maintain
- **Forgetting to handle messages in sender tab:** Admin panel changes won't reflect in admin UI itself
- **Re-initializing game state on theme change:** Theme is purely visual, game logic should be unaffected

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-tab communication | Custom localStorage polling | BroadcastChannel API | Native API handles pub-sub, no polling needed, cleaner API |
| Message serialization | Custom JSON stringify/parse | Structured Clone Algorithm | Automatic in BroadcastChannel, handles more types than JSON |
| Window references | Custom window tracking | window.opener property | Native bidirectional reference between same-origin windows |
| Color palette management | Ad-hoc color strings | Theme configuration object | Centralized, type-safe, easier to extend with new themes |

**Key insight:** BroadcastChannel API eliminates the need for any custom cross-tab communication infrastructure. Same-origin windows have full DOM access to each other, so complex postMessage protocols are unnecessary.

## Common Pitfalls

### Pitfall 1: Self-Broadcasting Confusion
**What goes wrong:** Developer expects sending tab to receive its own BroadcastChannel messages
**Why it happens:** BroadcastChannel explicitly excludes the sender from receiving messages
**How to avoid:** Manually call the same function that handles incoming messages when making local changes
**Warning signs:** Admin panel shows stale data after user makes changes in admin UI

```javascript
function updateTheme(themeName) {
  applyTheme(themeName);
  channel.postMessage({
    type: 'THEME_CHANGE',
    payload: { themeName }
  });
}
```

### Pitfall 2: Browser Compatibility Assumption
**What goes wrong:** BroadcastChannel not supported in older browsers, Safari had delayed support
**Why it happens:** Assuming all users have modern browsers
**How to avoid:** Feature detection and graceful degradation or polyfill
**Warning signs:** Admin panel doesn't affect game in certain browsers

```javascript
if (!window.BroadcastChannel) {
  console.warn('BroadcastChannel not supported');
}
```

### Pitfall 3: Sending Non-Serializable Data
**What goes wrong:** messageerror event fires, message never arrives
**Why it happens:** Attempting to send functions, DOM nodes, or circular references
**How to avoid:** Only send plain objects, arrays, primitives
**Warning signs:** Silent failures, messageerror events in console

```javascript
channel.addEventListener('messageerror', (event) => {
  console.error('Message deserialization failed:', event);
});
```

### Pitfall 4: Memory Leaks from Unclosed Channels
**What goes wrong:** BroadcastChannel objects remain in memory after tab close
**Why it happens:** Forgetting to call channel.close()
**How to avoid:** Close channels on beforeunload
**Warning signs:** Increasing memory usage in long-running tabs

```javascript
window.addEventListener('beforeunload', () => {
  channel.close();
});
```

### Pitfall 5: Board State Mutation on Theme Change
**What goes wrong:** Switching themes clears or corrupts game board state
**Why it happens:** Confusing theme change with game reset
**How to avoid:** Theme changes should only update currentTheme variable, not game state
**Warning signs:** Pieces disappear or board clears when changing themes

### Pitfall 6: Hardcoded Color in board Array
**What goes wrong:** Board stores piece colors from old theme, doesn't update with theme change
**Why it happens:** Current implementation stores piece.color directly in board array
**How to avoid:** Store piece type in board, look up color from current theme when rendering
**Warning signs:** Locked pieces retain old theme colors after theme switch

**Current codebase issue:** board.js lockPiece() stores `piece.color` in board array. This should store piece type instead, and render.js should look up color from current theme.

## Code Examples

Verified patterns from official sources:

### BroadcastChannel Setup
```javascript
const channel = new BroadcastChannel('tetris-sync');

channel.onmessage = (event) => {
  console.log('Received:', event.data);
};

channel.postMessage({ type: 'PING', payload: {} });

window.addEventListener('beforeunload', () => {
  channel.close();
});
```

### Opening Admin Panel Window
```javascript
const adminWindow = window.open(
  'admin.html',
  'tetris-admin',
  'width=400,height=600,resizable=yes'
);

if (adminWindow) {
  adminWindow.focus();
}
```

### Canvas fillStyle Dynamic Update
```javascript
ctx.fillStyle = currentTheme.colors.background;
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.fillStyle = currentTheme.colors[pieceType];
ctx.fillRect(x, y, width, height);
```

### Theme Cycling on Level Up
```javascript
const themeOrder = ['classic', 'neon', 'retro'];
let themeIndex = 0;

function onLevelUp() {
  themeIndex = (themeIndex + 1) % themeOrder.length;
  const newTheme = themeOrder[themeIndex];
  applyTheme(newTheme);

  channel.postMessage({
    type: 'THEME_CHANGE',
    payload: { themeName: newTheme }
  });
}
```

### Admin Stats Polling
```javascript
const channel = new BroadcastChannel('tetris-sync');
const statsElements = {
  score: document.getElementById('stat-score'),
  level: document.getElementById('stat-level'),
  theme: document.getElementById('stat-theme')
};

channel.onmessage = (event) => {
  if (event.data.type === 'STATS_RESPONSE') {
    const { score, level, theme } = event.data.payload;
    statsElements.score.textContent = score;
    statsElements.level.textContent = level;
    statsElements.theme.textContent = theme;
  }
};

setInterval(() => {
  channel.postMessage({ type: 'STATS_REQUEST', payload: {} });
}, 1000);
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| localStorage + polling | BroadcastChannel API | March 2022 baseline | Cleaner API, no polling overhead, automatic pub-sub |
| Hardcoded colors in render | Theme configuration objects | Modern practice | Instant theme switching, easier to add new themes |
| iframe for admin | window.open() separate tab | Modern UX pattern | Better separation, easier to manage as distinct tool |
| postMessage for same-origin | Direct window access + BroadcastChannel | Modern practice | Simpler API for same-origin, BroadcastChannel handles broadcasting |

**Deprecated/outdated:**
- localStorage events for cross-tab sync: Replaced by BroadcastChannel API which is purpose-built for this use case
- Polling intervals to detect changes: BroadcastChannel uses event-driven model

## Open Questions

Things that couldn't be fully resolved:

1. **Piece shape variations per theme**
   - What we know: THEM-02 mentions "themes define piece shapes"
   - What's unclear: Whether this means visual styling (rounded corners, outlines) or actual different shape matrices
   - Recommendation: Start with color-only themes. If shape variation needed, store shape set reference in theme object

2. **Admin panel persistence**
   - What we know: Admin panel should control game in real-time
   - What's unclear: Should admin panel settings persist across page reloads, or are they session-only
   - Recommendation: Session-only for MVP. Persistence can be added later via localStorage if needed

3. **Board growth interval timing**
   - What we know: ADMN-05 requires adjusting board growth interval
   - What's unclear: Current codebase doesn't implement board growth feature yet
   - Recommendation: Add boardGrowthInterval to game state, expose via sync protocol even if feature not implemented

## Sources

### Primary (HIGH confidence)
- [MDN - BroadcastChannel API](https://developer.mozilla.org/en-US/docs/Web/API/BroadcastChannel) - BroadcastChannel API usage, limitations, browser support
- [MDN - Broadcast Channel API](https://developer.mozilla.org/en-US/docs/Web/API/Broadcast_Channel_API) - Architecture patterns, best practices
- [MDN - Canvas fillStyle](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/fillStyle) - Dynamic color application
- [MDN - window.open()](https://developer.mozilla.org/en-US/docs/Web/API/Window/open) - Opening admin panel window

### Secondary (MEDIUM confidence)
- [Chrome Developers - BroadcastChannel API](https://developer.chrome.com/blog/broadcastchannel) - Message bus architecture
- [javascript.info - Cross-window communication](https://javascript.info/cross-window-communication) - Same-origin window access patterns
- [DEV Community - Stop Using LocalStorage](https://dev.to/henriqueschroeder/stop-using-localstorage-discover-the-power-of-broadcastchannel-26fe) - BroadcastChannel vs localStorage comparison
- [Netguru - Frontend Design Patterns 2026](https://www.netguru.com/blog/frontend-design-patterns) - CSS variables for theming

### Tertiary (LOW confidence)
- WebSearch results on canvas theming patterns - General approaches, not specific to this use case
- WebSearch results on admin panel templates - UI patterns, not architecture guidance

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - BroadcastChannel and Canvas APIs are native, well-documented Web APIs with official MDN documentation
- Architecture: HIGH - Message protocol pattern verified from official Chrome blog and MDN docs, theme object pattern standard JavaScript practice
- Pitfalls: HIGH - Self-broadcasting behavior documented in official specs, compatibility verified via Can I Use and MDN

**Research date:** 2026-02-02
**Valid until:** 2026-03-02 (30 days - stable APIs, unlikely to change)
