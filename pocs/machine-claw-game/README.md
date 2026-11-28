# Lucky Claw - machine-claw-game

A 3D arcade claw machine for the browser. Insert a coin, steer the claw over a pile of plush toys,
drop it and try to carry a stuffy to the prize chute. It runs on desktop and on phones, and every
grab is decided by a real physics simulation: nothing is scripted and nothing is magnetic.

Built with Fable 5.2.

Play it: https://diegopacheco.github.io/ai-playground/games/machine-claw-game/

![Winning a stuffy](printscreens/04-won.png)

## How it works

1. A coin buys one try and starts a 20 second timer.
2. While the timer runs the player moves the claw left, right, forward and back.
3. DROP (or the timer running out) sends the claw down until it touches a toy or the floor.
4. The three fingers close, the claw lifts, travels back over the chute and opens.
5. A toy that falls through the chute is a prize. A new toy falls in from the top to replace it.
6. The fingers and toys are rigid bodies in Rapier, so a toy is lifted only by contact and friction
   and can slip out on the way, like on a real machine.

## Architecture

![Architecture](printscreens/architecture.png)

* The player uses the React HUD (or the keyboard), which sends input to the engine.
* The engine runs the frame loop: it steps the simulation, draws the scene, plays sounds and
  publishes one HUD snapshot back to React.
* The simulation is headless. It joins the pure round rules with the Rapier arena and has no
  browser dependency, which is why a full round is played inside the test suite.
* The Three.js scene only reads poses from the simulation, it never changes them.

## Features

* Coin and credits - one coin is one try, extra coins are kept as credits, up to 9.
* Real physics grab - kinematic fingers and dynamic toys, so skill and luck both matter.
* Slack sensor - the claw stops on top of a toy instead of pushing through it.
* Forty stuffies in fourteen kinds - bear, bunny, cat, panda, frog, chick, koala, pig, penguin, octopus, dino,
  unicorn, fox and whale, built in code with shiny eyes and blush cheeks. Every kind is always in the pile.
* My Stuffies - every win is saved in localStorage. The STUFFIES tile in the stats row shows the total and opens a pastel
  3D showcase with each kind on a pedestal, its name and how many were won; drag or swipe to spin them.
* Restock - every prize is replaced by a new toy falling from the top.
* Free camera - drag to orbit around the machine, scroll or pinch to zoom, VIEW jumps between front and side.
* Arcade hall - neon signs, rows of game cabinets, a prize wall, string lights and a patterned carpet.
* Web and mobile - keyboard on desktop, touch joystick and big buttons on phones, portrait or landscape.
* Arcade feel - animated LEDs, marquee, coin toss, confetti and soft synthesized sounds, with a smooth motor whir while the claw moves.

## Stack

* TypeScript 7 (strict) - typed rules, typed events, no `any`.
* Bun 1.4 - package manager and test runner.
* Vite 8 - dev server and production build.
* React 19.3 - the HUD overlay only.
* Three.js 0.187 - the 3D scene, lights, shadows and materials.
* Rapier 0.21 - rigid body physics compiled to WASM.

TanStack is in the ts stack but is not used: the game has no routes, server state, forms or tables.

## Contracts

There is no backend and no REST API. The app is one web page.

| Action | Keyboard | Touch / mouse |
|---|---|---|
| Insert coin | `C` | COIN button |
| Move claw | Arrows or `WASD` | Joystick |
| Drop claw | `Space` | DROP button |
| Switch camera | `V` | VIEW button |
| Look around | Drag with the mouse | Drag with one finger |
| Zoom | Scroll wheel | Pinch |
| Open / close My Stuffies | STUFFIES tile / `Esc` | Tap STUFFIES / tap × |
| Spin stuffies | Drag in the showcase | Swipe in the showcase |

## Key data structures and design decisions

* `GameState` and `advance(state, input, sensors, dt)` in `src/game/machine.ts` - the round is a pure
  reducer over seven phases: idle, aiming, descending, closing, lifting, returning, releasing.
* `Sensors` - what physics tells the rules: touching, at top, at home, grip closed, grip open.
* `Arena` in `src/physics/arena.ts` - the Rapier world: cabinet boxes, toy bodies and the claw
  (a hub and three fingers, each finger two capsules and a tip).
* `layout.ts` - cabinet boxes and finger segments declared once and used by physics and rendering,
  so what is drawn is what collides.
* Toys are round with one ball collider. A taller two-ball toy was measured first and the claw
  squeezed it out instead of cupping it.
* Physics runs at a fixed 120 steps per second, so the grab behaves the same on every screen.
* The pile is 2.5x the original 16 toys (`PILE_SIZE` in `src/game/simulation.ts`). It spawns on a layered grid
  with no overlaps and slightly smaller toys, so 40 bodies settle calmly instead of exploding.
* Each plush kind is merged into one geometry per material and cached, so 40 stuffies cost few draw calls.
* `src/game/collection.ts` holds the won counts by kind name, pure and tested; the showcase in
  `src/render/showcase.ts` uses one Three.js renderer on one canvas, and the machine pauses while it is open.
* More detail in `design-doc.md`.

## How to run

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/ui.sh
./scripts/stop-all.sh
```

## How to test

```bash
./scripts/test-all.sh
```

It runs the typecheck, the 27 tests (round rules, headless physics rounds, the bigger pile and the stuffies collection) and the production build.

## Printscreens

### Insert coin
![Insert coin](printscreens/01-insert-coin.png)
The machine waits for a coin. The claw is parked open over the chute and the pile has settled by physics.

### Aiming
![Aiming](printscreens/02-aiming.png)
A coin was inserted, the timer is running and the claw was moved over the pile. DROP is lit.

### Lifting
![Lifting](printscreens/03-lifting.png)
The fingers closed under a stuffy and the claw is carrying it up. Only contact and friction hold it.

### Prize
![Prize](printscreens/04-won.png)
A stuffy fell through the chute: the prize counter is 1, the prize is on the shelf under the stats and
confetti flies.

### Looking around
![Orbit view](printscreens/05-orbit-view.png)
Dragging orbits the camera around the machine and scrolling zooms, so the pile can be judged in 3D.
The arcade hall surrounds the machine.

### Mobile
![Mobile](printscreens/06-mobile.png)
Portrait phone layout: touch joystick on the left, COIN and DROP on the right, STUFFIES tile in the stats row.

### My Stuffies
![My Stuffies](printscreens/07-collection.png)
Every stuffy won so far, one pedestal per kind with its name and count. Drag to spin them.

### My Stuffies on a phone
![My Stuffies mobile](printscreens/08-collection-mobile.png)
The same showcase in portrait at 390x844, swipe to spin and tap × to go back to the machine.

## Scripts

All scripts live in `scripts/` and run from any directory of the repository.

| Script | What it does |
|---|---|
| `./scripts/setup.sh` | Installs dependencies and prepares the app |
| `./scripts/start-all.sh` | Starts every service and prints the full link of each one |
| `./scripts/status.sh` | Shows every service port as UP or DOWN |
| `./scripts/test-all.sh` | Runs every test suite |
| `./scripts/ui.sh` | Opens the UI in the browser |
| `./scripts/stop-all.sh` | Stops every service |

Ports are declared in `scripts/ports.env`.

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/status.sh
./scripts/ui.sh
./scripts/stop-all.sh
```
