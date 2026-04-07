# Steel Breach

Steel Breach is a small browser FPS built with plain HTML, CSS, and JavaScript. It uses a classic raycasting approach to create a fast retro corridor shooter feel with a heavier industrial presentation.

## Features

- Pure frontend stack with no package manager and no framework
- Raycasted walls and billboard enemies
- Keyboard movement, mouse look, sprint, and instant-fire combat
- Local run and stop scripts for a quick workflow

## Controls

- `W`, `A`, `S`, `D` to move and turn
- `Q`, `E` to strafe
- `Shift` to sprint
- Mouse to aim after pointer lock
- Left click to fire

## Run

```bash
./run.sh
```

Open `http://127.0.0.1:8091`.

## Stop

```bash
./stop.sh
```

## Files

- `index.html` contains the HUD and canvas shell
- `style.css` defines the visual identity and layout
- `game.js` contains the raycaster, enemy logic, combat, and rendering
- `run.sh` starts a local static server and stores its PID
- `stop.sh` stops the local server using the stored PID
