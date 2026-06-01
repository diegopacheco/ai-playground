# 3D Tetris — Design Doc

## Goal
A playable 3D Tetris (Blockout style) in the browser. Pieces are 3D polycubes
that fall down a rectangular pit. The player moves and rotates the piece across
three axes. When a horizontal layer of the pit is completely filled it clears
and everything above drops down.

## Stack
- React 19 — UI and component model
- Bun 1.3 — runtime, package manager and bundler (native HTML entrypoint, no Vite)
- TypeScript 6 — types across game logic and components
- TanStack Store — single source of truth for game state, React binding via `@tanstack/react-store`
- Three.js + @react-three/fiber — WebGL 3D rendering of the pit, the settled blocks and the active piece

No other runtime libraries. The dev server is a small `server.ts` using `Bun.serve`.

## Pit and coordinates
The pit is a box `W x D x H` (width x depth x height).

- `x` in `[0, W)` — left/right
- `z` in `[0, D)` — near/far (depth)
- `y` in `[0, H)` — up; `y = 0` is the floor, pieces fall toward it

Default size: `W = 5`, `D = 5`, `H = 12`.

The board is a flat `number[]` of length `W*D*H`. Index = `x + z*W + y*W*D`.
A value of `0` means empty; any other value is a color index for a settled cell.

## Pieces
A piece is a set of integer cell offsets plus a position and a color:

```
type Piece = { cells: Vec3[]; pos: Vec3; color: number }
```

The shape set mixes flat tetrominoes (I, O, L, T, S) with genuinely 3D pieces
(tripod, branch) so rotation across all three axes matters.

### Rotation
Rotation is a 90 degree turn of every offset around one axis through the piece
origin:

- X axis: `(x, y, z) -> (x, -z, y)`
- Y axis: `(x, y, z) -> (z, y, -x)`
- Z axis: `(x, y, z) -> (-y, x, z)`

A rotation is applied only if every resulting cell is still inside the pit and
not overlapping a settled cell. No wall kicks; the move is simply rejected.

## Engine
Pure functions over the board plus the store actions:

- `spawn()` — pick a random shape, place it centered at the top. If it cannot be
  placed the game is over.
- `move(dx, dy, dz)` — shift if the target is valid.
- `rotate(axis)` — rotate if the result is valid.
- `tick()` — gravity step: move down one cell; if blocked, lock the piece, clear
  full layers, then spawn the next piece.
- `drop()` — hard drop: move down until blocked, then lock immediately.
- `lock()` — write the piece colors into the board.
- `clearLayers()` — remove every full `y` layer and shift the layers above down.

### Scoring and speed
- Cleared layers per lock score `[0, 100, 300, 600, 1000]` times the level.
- Level rises every 8 cleared layers.
- Gravity interval = `max(120, 800 - (level - 1) * 70)` ms.

## State (TanStack Store)
```
type GameState = {
  board: number[]
  piece: Piece | null
  nextShape: number
  score: number
  level: number
  cleared: number
  status: 'idle' | 'playing' | 'paused' | 'over'
}
```

The game loop is a single `requestAnimationFrame` accumulator that reads the
current level for its interval and calls `tick()`. It pauses when status is not
`playing`.

## Rendering
`@react-three/fiber` `Canvas` with a fixed angled perspective camera so controls
stay consistent:

- ambient light + one directional light
- pit drawn as a wireframe box and a floor grid
- each settled cell is a colored box with darkened edges
- the active piece is rendered slightly emissive so it stands out
- the ghost projection of where the piece will land is drawn as faint boxes

## Controls
- Arrow Left / Right — move along x
- Arrow Up / Down — move along z (depth)
- `Q` / `E` — rotate around y
- `A` / `D` — rotate around x
- `W` / `S` — rotate around z
- Space — hard drop
- `P` — pause / resume
- Enter — start / restart

## Files
```
index.html        HTML entrypoint, loads src/main.tsx
server.ts         Bun.serve dev server, bundles the HTML
src/main.tsx      React root
src/App.tsx       layout, keyboard handling, game loop
src/game/types.ts shared types
src/game/pieces.ts shape and color definitions
src/game/store.ts  TanStack Store + engine actions
src/components/Scene.tsx   Canvas, camera, lights
src/components/Pit.tsx     pit frame, settled cells, ghost
src/components/ActivePiece.tsx active piece boxes
src/components/Hud.tsx     score, level, controls overlay
start.sh / stop.sh
```

## Out of scope
- Multiplayer, persistence, sound
- Mobile touch controls
- Wall kicks and modern SRS rotation rules
