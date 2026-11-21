# Design Doc - machine-claw-game

## Goal
A 3D arcade claw machine that runs in the browser on desktop and mobile. The player inserts a coin,
steers the claw over a pile of plush toys, drops it and tries to carry a toy to the prize chute.
Nothing is scripted: a toy is won only when the physics engine really carries it into the chute.

## Architecture
Four layers, each one only knows the layer below it.

1. `src/game` - pure rules. `machine.ts` is the round state machine (coin, aim, descend, close, lift,
   return, release). `simulation.ts` joins the rules with the physics arena and has no DOM or WebGL
   dependency, so the whole game runs headless in tests.
2. `src/physics` - `arena.ts` owns the Rapier world: cabinet colliders, plush toy bodies and the
   kinematic claw (hub plus three two-segment fingers).
3. `src/render` and `src/audio` - Three.js scene (cabinet, plush meshes, claw rig, camera, confetti,
   coin toss) and WebAudio sound effects. They read the simulation, they never change it.
4. `src/ui` - React HUD: credits, timer, prizes, joystick, DROP, COIN and VIEW buttons. It talks to
   `engine.ts`, which runs the frame loop and publishes one immutable HUD snapshot.

## Stack
* TypeScript 7 (strict), Bun 1.4, Vite 8, React 19.3 - the ts stack.
* Three.js 0.187 - rendering, asked for by the task.
* Rapier 0.21 (`@dimforge/rapier3d-compat`) - rigid body physics in WASM, needed for real physics.
* TanStack is part of the ts stack but is not installed: the game has no routes, no server state,
  no forms and no tables, so there is nothing for it to own.

## Public endpoints
There is no backend and no REST API. The only endpoint is the web page served by Vite on the
`frontend` port declared in `scripts/ports.env`.

Player contract:

| Action | Keyboard | Touch / mouse |
|---|---|---|
| Insert coin | `C` | COIN button |
| Move claw | Arrows or `WASD` | Joystick |
| Drop claw | `Space` | DROP button |
| Switch camera | `V` | VIEW button |

## Important design decisions
* Real grab, no magnet. The fingers are kinematic bodies with capsule colliders and the toys are
  dynamic bodies. A toy is lifted only by contact and friction, so grabs can fail like on a real machine.
* The claw stops descending when a ray cast from the hub finds a toy or the floor right below it,
  the same job the slack-cable switch does on a real machine.
* The rules are a pure reducer `advance(state, input, sensors, dt)`. Physics feeds it sensors and it
  returns the next state, so every rule is unit tested without a physics engine.
* The simulation is headless. An end to end test plays a full round in Bun and checks that the toy
  really lands in the chute.
* Finger geometry and cabinet boxes are declared once in `layout.ts` and used by both physics and
  rendering, so what is drawn is what collides.
* Physics steps with the frame time, clamped to 1/30 s, so motion stays smooth on 60 Hz and 120 Hz screens.
* Toys are round plush balls with one ball collider each. A two-ball body plus head toy was tried first
  and measured: it is taller than the claw cage, so the fingers squeezed it out instead of cupping it.
* One coin is one try. A won toy is replaced by a new one falling from the top.
* The random source is injected into the simulation so tests are deterministic.

## Pros
* Honest physics makes every try different.
* Rules and physics are tested without a browser.
* No backend, no assets to download: all models, textures and sounds are generated in code.

## Cons
* Kinematic fingers push with unlimited force, so a toy squeezed against the floor can pop away.
* Plush toys are rigid bodies that only look soft, there is no cloth or soft body simulation.
* The bundle is about 1.9 MB gzipped, most of it the Rapier WASM.
* Progress (prizes won) is not saved between page loads.
