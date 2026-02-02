# Stack Research: Browser Tetris with Real-time Admin

## Recommended Stack

### Core Technologies

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| Vanilla JavaScript (ES6+) | Game logic, UI | No dependencies, full control, fast |
| HTML5 Canvas | Game rendering | Native, performant for 2D games, pixel-perfect control |
| CSS3 | Admin UI, themes | Native styling, CSS variables for theming |
| BroadcastChannel API | Tab sync | Native browser API, real-time, no server needed |

### Why These Choices

**Canvas over DOM manipulation:**
- Tetris requires frequent redraws (60fps game loop)
- Canvas is optimized for this — DOM is not
- Easier collision detection with pixel/grid logic

**BroadcastChannel over alternatives:**
- `localStorage` events: Works but hacky, not designed for messaging
- `SharedWorker`: Overkill for two tabs
- `BroadcastChannel`: Purpose-built for cross-tab messaging, clean API

**Vanilla JS over frameworks:**
- Project constraint: minimal dependencies
- Game logic doesn't benefit from React/Vue reactivity
- Smaller bundle, faster load
- Full control over game loop timing

### File Structure

```
/
├── index.html          # Player game
├── admin.html          # Admin panel
├── css/
│   ├── game.css
│   ├── admin.css
│   └── themes/
│       ├── classic.css
│       ├── neon.css
│       └── retro.css
├── js/
│   ├── game/
│   │   ├── core.js     # Game loop, state
│   │   ├── pieces.js   # Tetromino definitions
│   │   ├── board.js    # Board logic, collision
│   │   ├── render.js   # Canvas drawing
│   │   └── input.js    # Keyboard handling
│   ├── admin/
│   │   └── panel.js    # Admin controls
│   ├── shared/
│   │   ├── config.js   # Default config values
│   │   ├── sync.js     # BroadcastChannel wrapper
│   │   └── themes.js   # Theme definitions
│   └── main.js         # Entry point
```

### What NOT to Use

| Technology | Why Avoid |
|------------|-----------|
| React/Vue/Angular | Overkill, adds bundle size, complicates game loop |
| Phaser/PixiJS | Heavy for simple Tetris, unnecessary abstraction |
| WebSockets | No server needed — same browser tabs |
| localStorage for sync | Not designed for messaging, race conditions |

### Browser Support

BroadcastChannel supported in all modern browsers:
- Chrome 54+
- Firefox 38+
- Edge 79+
- Safari 15.4+

No polyfill needed for modern browser target.
