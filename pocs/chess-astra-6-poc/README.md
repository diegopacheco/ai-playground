<p align="center"><img src="public/logo.svg" width="80" alt="The Wizard’s Gambit crest" /></p>
<h1 align="center">The Wizard’s Gambit</h1>
<p align="center">A little magic. A game of minds.</p>

A playable, wizard-inspired 3D chess game for one human against a local CPU. Choose white, green, brown, or black pieces and challenge the Castle Guardian’s contrasting army in a Hogwarts-inspired castle library with tall oak bookshelves, Gothic stonework, house-colored banners, and a glowing fireplace. Built with Three.js, Bun, Vite, and TypeScript.

![The castle library with the full 3D board and match controls](printscreens/great-hall.png)

## Features

- Sculpted 3D pieces, bronze trim, soft shadows, and drifting sparks sit among 540 colored books, vaulted stonework, a library ladder, and a carved fireplace. There are no floating candles.
- Captured pieces rise and spin away as glowing fragments scatter through an expanding golden ring, with a falling magical swish and sparkle.
- On checkmate the defeated king trembles, topples onto its side, and shatters into crimson fragments while a low toll, an impact thud timed to the fall, and a descending chime play.
- Choose **white, green, brown, or black** for your army at any time. The CPU uses green against white, and white against the other colors. Your preference survives reloads, undo, and new games.
- The **fullscreen button** beside the camera and sound controls expands the board and sidebar together; use it again or press Escape to exit.
- Three CPU levels use a background worker so the board stays responsive during search.
- Checkmate displays a prominent result card over the board, identifies the winner, and offers an immediate **New game** button. It also appears when reopening a finished match and works in fullscreen and on mobile.
- Full legal chess includes castling, en passant, promotion choices, checkmate, stalemate, and automatic draws.
- Click pieces and destinations, or enter coordinate moves and standard algebraic notation with a keyboard.
- Orbit, zoom, rotate the board, or switch to an overhead camera. Near-side library walls cut away when the camera moves behind them so they do not obscure the board.
- Undo a human/CPU turn, start a fresh match, and restore the current match after a reload.
- The speaker control opens a small looping YouTube player for **Hedwig’s Theme** in the corner of the board, alongside locally synthesized move, capture, and checkmate sounds. Tap again to mute and unload the player. The video id lives in the `song` constant in `src/main.ts`; the video must allow embedding for the player to start.
- The full interface fits the browser viewport. Move history and game panels scroll internally on smaller screens, while the outer page stays fixed.
- Reduced-motion preferences keep firelight and ambient effects still and suppress capture and king-fall animations.
- A playable flat board appears when WebGL cannot initialize.

## How it Works?

1. You control the side that moves first; choose its appearance with the four piece-color swatches.
2. Select a piece to display its legal destinations, then select a golden ring.
3. Alternatively, enter `e2e4`, `Nf3`, or `O-O` in the Move field.
4. A pawn reaching the final rank opens a queen, rook, bishop, or knight picker.
5. The rules engine validates the move before the scene and chronicle change.
6. Captures trigger the defeated piece’s magical exit, including en passant captures.
7. The Guardian searches the position in a Web Worker and returns its chosen move.
8. The browser saves the complete move history, challenge level, and selected piece color after each change.
9. Checkmate shows the winner and a New game button over the board. Restart immediately, or undo to revisit the position; stalemates and draws show the same card with the reason, so a finished match never looks frozen.

## Architecture

![Hand-drawn architecture showing move validation, CPU search, rendering, captures, and storage](printscreens/architecture.png)

The [editable SVG diagram](docs/architecture.svg) embeds the Caveat handwriting font, uses pastel boxes and a wobble filter, and shows the capture path with a solid arrow. The font’s license is in [docs/Caveat-OFL.txt](docs/Caveat-OFL.txt).

This is a static browser application. Vite serves source during development and writes deployable assets to `dist/`. The CPU runs on the user’s device. There is no backend, account, database, external chess API, or paid service.

## Stack

- **Three.js 0.186** — renders the board, procedurally sculpted pieces, shadows, particles, and camera controls.
- **chess.js 1.4** — provides reliable legal move generation and game termination rules.
- **TypeScript 7.0** — checks application, engine, worker, and testing contracts.
- **Bun 1.4** — installs dependencies, executes Vite, and runs the engine tests.
- **Vite 8.3** — supplies fast development updates and the production asset build.
- **Playwright 1.63** — checks real browser interaction, fallback behavior, and responsive screenshots.
- **Native HTML, CSS, Web Audio, and Web Workers** — keep UI, sound, and CPU execution free of extra runtime libraries.

Exact resolved versions are recorded in `bun.lock`. The two runtime dependencies are Three.js and chess.js. Interface fonts load from Google Fonts, with local serif and sans-serif fallbacks; gameplay does not depend on the font request succeeding.

## How to run the app

Install Bun 1.4+ and Node.js 22.12+ with npm available for `npx playwright`. Bash, `curl`, and `lsof` are used by the operational scripts. The scripts support Bash 3.2 on macOS and modern Bash on Linux.

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/status.sh
./scripts/ui.sh
```

Open **http://127.0.0.1:5173**. The port is declared once in `scripts/ports.env`; Vite, Playwright, and operational scripts all read it. Vite uses a strict port and binds to loopback.

For a foreground development session:

```bash
bun run dev
```

Build and serve the production assets:

```bash
bun run build
bun run preview
```

Stop the development server before starting preview, since both use the configured port. The `dist/` directory can be served by a static host at its root. The current bundle is approximately 174 KB gzipped plus a 38 KB CPU worker before compression. Vite reports its normal 500 KB uncompressed chunk advisory because the main bundle includes the Three.js renderer.

## Tests

```bash
./scripts/test-all.sh
```

The complete command runs Bun tests, TypeScript validation, the production build, and Chromium browser tests. Playwright starts a temporary Vite server if one is not already running. Browser tests refresh the screenshots in `printscreens/`; failures retain a trace under `test-results/`.

Individual checks:

```bash
bun run test
bun run build
npx playwright test
PLAYWRIGHT_PREVIEW=1 npx playwright test
```

Validation on September 12, 2026:

```text
11 engine and rules tests passed
15 browser tests passed
TypeScript validation passed
Production build passed
```

Engine tests cover legal CPU replies at all levels, mate selection, terminal positions, budget fallback, search state preservation, repetition restoration, castling, en passant, underpromotion, and invalid saved moves. Browser tests cover a complete human/CPU turn, 3D board clicks, camera controls, audio toggling, save/restore, invalid input, restart confirmation, capture effects, checkmate, the guide, mobile layout, reduced motion, corrupt storage, WebGL fallback, promotion, interrupted turns, repetition draws, page overflow at eight desktop/mobile/landscape sizes, the YouTube player’s looping embed and mute behavior, capture and checkmate sound timing, stalemate results, all four piece colors, preference persistence during a match, fullscreen play with the restart dialog, live player/CPU checkmates, and restarting from the result card on mobile and in fullscreen.

## Contracts / APIs

There are no HTTP application APIs. These are the internal contracts:

| Contract | Shape and behavior |
|---|---|
| Move input | SAN such as `Nf3`, or coordinates such as `e2e4`; promotion can include a trailing `q`, `r`, `b`, or `n`. |
| CPU request | `{ history: string[], difficulty: 'apprentice' \| 'wizard' \| 'grandmaster' }` via `worker.postMessage`. |
| CPU response | `{ move: { from, to, promotion? } \| null }`, or `{ error: string }`. The controller validates the returned move again. |
| Saved match | Local storage key `wizards-gambit-v1` stores `{ history: string[], difficulty, pieceColor: 'white' | 'green' | 'brown' | 'black' }`. Older saves without a valid color default to white. |
| Search | `chooseMove(game, difficulty, budget = 1400)` returns a legal move or `null` when the match is over, preserving the supplied game state. |
| Scene update | `ChessScene.sync(game, move?)` mirrors the legal position and animates the latest move or capture. |

## Key data structures and design decisions

- **One authoritative `Chess` instance:** the scene mirrors the rules engine and never decides whether a move is legal.
- **SAN history:** persisting moves rather than only FEN preserves repetition detection and turn-by-turn undo after reload.
- **Piece appearance:** dedicated player and CPU materials keep color changes separate from the board and library materials. Cosmetic colors do not change chess sides or turn order; the fallback board and chronicle use the same palette.
- **Fullscreen:** the native Fullscreen API expands the game grid; the camera resizes with its container and the button follows fullscreen change events. Unsupported browsers keep the control disabled.
- **Square identifiers:** algebraic squares such as `e4` connect chess state, raycast targets, highlights, and 3D coordinates.
- **Dedicated CPU worker:** iterative deepening with negamax, alpha-beta pruning, material values, move ordering, and simple positional evaluation keeps search isolated from rendering.
- **Bounded difficulty:** Apprentice searches one ply, Wizard two, Grandmaster up to three within a 1.4-second soft budget. These names are flavor, not chess ratings. The time budget is checked between search operations and is not a hard real-time deadline.
- **Music and sound effects:** music streams from YouTube’s embedded player, which loops the video and is unloaded when muted. `src/audio.ts` synthesizes move, capture, and checkmate sounds with Web Audio oscillators and filtered noise, timing the checkmate impact to the king’s fall.
- **Viewport layout:** a `100dvh` application shell allocates remaining height to the game grid. Explicit minimum sizes and internal overflow keep controls reachable on short screens.
- **Short-lived effects:** defeated pieces and particles live independently of the updated board until their animation ends, then their geometry and effect materials are released.
- **Cancellation:** undo, restart, and challenge changes terminate active search; pending CPU timers are also canceled.
- **Simple deployment:** no framework or backend is needed for a single chess table. All pieces are built from geometry, without remote model assets.
- **Resilience:** invalid saved matches reset safely, storage failure is reported, WebGL failure reveals a flat board, and worker failure leaves undo/new game available.

## Printscreens

### Castle library

The opening position shows both sculpted armies, the book-lined castle library and burning fireplace, challenge selection, and an empty chronicle.

![Opening position in the castle library](printscreens/great-hall.png)

### Active match

A human move and the Guardian’s reply appear in the chronicle. The previous move’s squares are highlighted.

![An active human-versus-CPU match](printscreens/active-match.png)

### Capture enchantment

The ivory pawn captures on d5. The captured jade pawn dissolves into shards and a gold shockwave while the next turn begins. A still image records one moment of the moving effect.

![A capture with a golden ring and magical fragments](printscreens/capture.png)

### Overhead view

The overhead camera and board rotation provide a clear alternative angle for planning moves.

![Rotated overhead board](printscreens/overhead.png)

### Promotion

A pawn reaching the last rank opens a picker, including rook, bishop, and knight underpromotion.

![Pawn promotion choice](printscreens/promotion.png)

### Checkmate

A centered **Checkmate** card names the winner and offers **New game** immediately. The move input is disabled, while undo remains available. Clicking New game resets the board directly and retains your selected color and fullscreen mode.

![Checkmate result with an immediate New game button](printscreens/checkmate.png)

The player’s victory appears over the board in fullscreen:

![Player checkmate and New game button in fullscreen](printscreens/checkmate-victory.png)

The same result fits inside the mobile game panel:

<img src="printscreens/checkmate-mobile.png" width="390" alt="Mobile checkmate result with the New game button visible" />

### Spellbook

The guide explains selection, notation, special moves, persistence, and the CPU’s search depth.

<img src="printscreens/spellbook.png" width="390" alt="The How to play dialog on mobile" />

### Mobile

The 390 × 844 layout fits the board, opponent controls, chronicle, and status inside one browser viewport. Smaller panels scroll internally; the page does not scroll. Reduced-motion preferences are enabled in this capture.

<img src="printscreens/mobile.png" width="390" alt="The complete mobile chess interface" />

### Laptop viewport

The 1366 × 768 view keeps the board and move input within the window. The sidebar can scroll independently when its content needs more space.

![Castle library chess fitting a laptop viewport](printscreens/laptop.png)

### Piece colors

The four swatches recolor your army immediately while keeping the opposing pieces distinct. These views show white, green, brown, and black without changing the starting position.

| White | Green |
|---|---|
| ![White player pieces against green](printscreens/pieces-white.png) | ![Green player pieces against white](printscreens/pieces-green.png) |

| Brown | Black |
|---|---|
| ![Brown player pieces against white](printscreens/pieces-brown.png) | ![Black player pieces against white](printscreens/pieces-black.png) |

### Game fullscreen

The fullscreen button expands the chessboard and all match controls together. Here the green army has played its first move, and the CPU has replied.

![The game in fullscreen with green player pieces](printscreens/fullscreen.png)

## References

The implementation follows the official [Three.js renderer documentation](https://threejs.org/docs/pages/WebGLRenderer.html), [OrbitControls documentation](https://threejs.org/docs/pages/OrbitControls.html), [chess.js API](https://github.com/jhlywa/chess.js), and [Vite guide](https://vite.dev/guide/).

## Scripts

All scripts live in `scripts/` and run from any directory of the repository.

| Script | What it does |
|---|---|
| `./scripts/setup.sh` | Installs locked dependencies and Playwright Chromium. |
| `./scripts/start-all.sh` | Starts Vite in the background and checks HTTP readiness for up to 30 seconds. |
| `./scripts/status.sh` | Shows the frontend’s port, UP/DOWN state, and listening PID; reporting DOWN is a successful status query. |
| `./scripts/test-all.sh` | Runs engine tests, type validation, production build, and browser tests. |
| `./scripts/ui.sh` | Opens the running frontend with the platform browser opener. |
| `./scripts/stop-all.sh` | Stops only the Vite process recorded and owned by this project, waiting up to 30 seconds. |
| `./scripts/common.sh` | Resolves paths, loads the port, and provides process ownership helpers. |

Ports are declared in `scripts/ports.env`. PID files and logs are stored in `.run/`; the frontend log is `.run/logs/frontend.log`. Start and stop are safe to repeat. An occupied port belonging to another process produces an error and is never forcibly cleared. There is no SQL console because this app has no database.

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/status.sh
./scripts/ui.sh
./scripts/test-all.sh
./scripts/stop-all.sh
```
