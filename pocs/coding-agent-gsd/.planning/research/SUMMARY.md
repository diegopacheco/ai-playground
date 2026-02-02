# Research Summary: Tetris Twist

## Stack Decision

**Vanilla JS + HTML5 Canvas + CSS3 + BroadcastChannel**

No frameworks. Canvas for game rendering (performant 60fps), BroadcastChannel for real-time tab sync (native API, no server). This fits the "minimal dependencies" constraint perfectly.

## Table Stakes Features

Must have for any Tetris:
- 7 tetrominoes with rotation
- Movement (left, right, soft drop, hard drop)
- Line clearing and scoring
- Next piece preview
- Game over detection
- Pause

## Key Differentiators

What makes this unique:
- **10s play / 10s freeze cycle** — visual indicator critical
- **Growing board** — plan for max size, scale/scroll strategy
- **Real-time admin control** — BroadcastChannel sync, debounced inputs
- **Live theme switching** — full canvas redraw on change

## Architecture Highlights

- Two HTML pages: `index.html` (game), `admin.html` (panel)
- Shared JS modules for config, themes, sync
- Game state in game tab, config broadcast from admin
- `requestAnimationFrame` game loop with delta timing

## Critical Watch-Outs

1. **Game loop timing** — use rAF + delta time, pause on tab hidden
2. **Wall kicks** — don't let rotation fail silently at edges
3. **Canvas blur** — scale by devicePixelRatio for crisp rendering
4. **Board growth** — use grid coordinates, plan max size
5. **Freeze UX** — clear visual state (timer, overlay) so it doesn't look like a bug
6. **Theme swap** — full redraw to avoid artifacts

## Build Order Recommendation

1. Canvas setup + grid rendering
2. Piece definitions + movement
3. Board state + collision + line clear
4. Game loop + scoring
5. Themes + admin panel + sync
6. Freeze cycle + board growth
7. Polish + edge cases
