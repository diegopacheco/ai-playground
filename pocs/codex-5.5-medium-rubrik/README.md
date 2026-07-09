# Rubik's Cube

A colorful browser Rubik's Cube with 3D-looking cubies, visible layer rotations, and Scramble and Solve controls.

## Run

```bash
./start.sh
```

Open `http://127.0.0.1:8097`.

Use a different port with `PORT=8081 ./start.sh`.

## Stop

```bash
./stop.sh
```

## Files

`index.html` contains the page structure.

`styles.css` contains the 3D cube presentation and responsive layout.

`app.js` contains the cube state, scramble sequence, solve sequence, and turn animation.
