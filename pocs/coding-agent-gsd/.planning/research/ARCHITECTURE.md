# Architecture Research: Browser Tetris with Admin Sync

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER                               │
│  ┌─────────────────┐              ┌─────────────────┐       │
│  │   GAME TAB      │              │   ADMIN TAB     │       │
│  │                 │              │                 │       │
│  │  ┌───────────┐  │              │  ┌───────────┐  │       │
│  │  │  Canvas   │  │              │  │  Controls │  │       │
│  │  │  Render   │  │              │  │   Panel   │  │       │
│  │  └───────────┘  │              │  └───────────┘  │       │
│  │       ↑         │              │       │         │       │
│  │  ┌───────────┐  │              │       ↓         │       │
│  │  │   Game    │  │   config     │  ┌───────────┐  │       │
│  │  │   State   │←─┼──────────────┼──│  Config   │  │       │
│  │  └───────────┘  │  (Broadcast  │  │  State    │  │       │
│  │       ↑         │   Channel)   │  └───────────┘  │       │
│  │  ┌───────────┐  │              │                 │       │
│  │  │  Config   │  │              │                 │       │
│  │  │ Receiver  │  │              │                 │       │
│  │  └───────────┘  │              │                 │       │
│  └─────────────────┘              └─────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Major Components

### 1. Game Engine (game tab)

| Component | Responsibility |
|-----------|----------------|
| GameLoop | requestAnimationFrame loop, timing, freeze cycle |
| Board | Grid state, collision detection, line clearing |
| Piece | Current piece state, rotation, movement |
| PieceFactory | Generate random pieces, next piece queue |
| Renderer | Canvas drawing, theme application |
| InputHandler | Keyboard events, action mapping |
| ConfigReceiver | Listen for admin config changes |

### 2. Admin Panel (admin tab)

| Component | Responsibility |
|-----------|----------------|
| ConfigPanel | UI controls (sliders, dropdowns, buttons) |
| ConfigState | Current config values |
| ConfigBroadcaster | Send changes to game tab |
| ThemePreview | Show theme before applying |

### 3. Shared Modules

| Component | Responsibility |
|-----------|----------------|
| SyncChannel | BroadcastChannel wrapper, message types |
| Themes | Theme definitions (colors, piece shapes) |
| Config | Default values, config schema |

## Data Flow

### Config Change Flow
```
Admin: User changes fall speed slider
  → ConfigPanel updates ConfigState
  → ConfigBroadcaster sends message via BroadcastChannel
  → Game: ConfigReceiver receives message
  → Game: GameLoop reads new config on next tick
  → Game: Piece falls at new speed
```

### Theme Change Flow
```
Admin: User selects "Neon" theme
  → ConfigBroadcaster sends {type: 'theme', value: 'neon'}
  → Game: ConfigReceiver receives
  → Game: Renderer loads theme colors/shapes
  → Game: Canvas redraws with new theme
```

### Freeze Cycle Flow
```
GameLoop tick:
  → Check freeze timer
  → If freeze period: skip piece movement, show frozen state
  → If play period: normal game logic
  → Update timer display
```

### Board Growth Flow
```
GameLoop tick:
  → Check growth timer (every 30s)
  → If growth triggered:
    → Board.expand() adds rows/columns
    → Renderer resizes canvas
    → Existing pieces stay in place (relative to bottom-left)
```

## Message Types (BroadcastChannel)

```javascript
// Admin → Game
{ type: 'config', key: 'fallSpeed', value: 500 }
{ type: 'config', key: 'pointsPerRow', value: 15 }
{ type: 'config', key: 'growthInterval', value: 45000 }
{ type: 'theme', value: 'neon' }

// Game → Admin (optional status updates)
{ type: 'status', score: 100, level: 2 }
```

## Build Order (Dependencies)

1. **Core rendering** — Canvas setup, draw grid
2. **Piece definitions** — 7 tetrominoes, rotation states
3. **Board logic** — Grid, collision, line clear
4. **Game loop** — Timing, piece spawning
5. **Input handling** — Keyboard controls
6. **Scoring/levels** — Points, level progression
7. **Themes** — Color/shape definitions, CSS variables
8. **Sync channel** — BroadcastChannel wrapper
9. **Admin UI** — Control panel
10. **Freeze cycle** — Timer, frozen state
11. **Board growth** — Dynamic resize
12. **Integration** — Connect all pieces

## State Management

**Game State (game tab):**
```javascript
{
  board: number[][],      // Grid cells
  currentPiece: Piece,    // Active piece
  nextPiece: Piece,       // Preview
  score: number,
  level: number,
  isFrozen: boolean,
  freezeTimer: number,
  growthTimer: number,
  config: Config          // From admin
}
```

**Config State (shared):**
```javascript
{
  theme: string,          // 'classic' | 'neon' | 'retro'
  fallSpeed: number,      // ms per row
  pointsPerRow: number,   // points for clearing
  growthInterval: number, // ms between growth
  freezeInterval: number, // ms between freezes
  freezeDuration: number  // ms frozen
}
```
