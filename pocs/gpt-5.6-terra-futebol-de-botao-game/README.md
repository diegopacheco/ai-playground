# Mesa 12

A dependency-free futebol de botão game for the browser. Play locally with two people or challenge an AI-controlled Inter powered by Claude, Codex, or Agy.

## Run

```bash
./start.sh
```

Open `http://127.0.0.1:8080`.

Use a different port when needed:

```bash
PORT=9090 ./start.sh
```

Stop the server:

```bash
./stop.sh
```

## Play

1. Select a button from the team shown in the yellow turn panel.
2. Drag backward to aim and set the shot power.
3. Release to flick the button.
4. Wait for every piece to stop before the next turn.
5. Put the yellow ball into the opposing goal.

At startup, choose **Humano × Humano** or **Humano × IA**. AI mode requires at least one authenticated local CLI:

```bash
claude -p "prompt"
codex exec "prompt"
agy --print "prompt"
```

The match lasts three minutes. Use **PAUSAR** to pause the clock and **REINICIAR** to reset the score, clock, and pieces.

## Stack

- HTML5 Canvas
- CSS
- JavaScript
- Python game server

No package installation or third-party runtime libraries are required.
