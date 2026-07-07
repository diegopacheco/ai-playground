# GESTURE GOALKEEPER

Block penalty shots with your hand. The webcam tracks your hand position and turns it into goalkeeper gloves on the screen. Move fast, read the shot, and stop as many kicks as possible before five goals get through.

## How to play

- Move your hand in front of the webcam to move the goalkeeper gloves
- Put one or both hands in the shot path to save the ball
- Each save gives 100 points, then the shooter speeds up
- Five goals against ends the match
- No webcam? Use the mouse as the fallback controller

## Run it

```bash
./start.sh
```

Open http://localhost:8000 and allow camera access.

```bash
./stop.sh
./test.sh
```

## Architecture

The browser owns the game loop, canvas rendering, camera capture, fallback mouse input, and synthesized audio. It sends small JPEG webcam frames over a WebSocket to the Python server. The server decodes each frame with OpenCV, runs MediaPipe HandLandmarker, extracts hand centers from the 21 landmarks, and sends normalized hand positions back to the browser.

The game keeps the tracking payload small:

```json
{"hands":[{"x":0.48,"y":0.31,"size":0.18}]}
```

## Stack

- Python: http.server, websockets, OpenCV, MediaPipe Tasks
- Browser: vanilla HTML, CSS, canvas, WebAudio
- Input: webcam hand tracking with mouse fallback
- Build: none

## Files

- `server.py` serves the game and tracks hands
- `web/index.html` loads the canvas app
- `web/style.css` styles the stadium shell
- `web/game.js` runs the game
- `start.sh` starts the server
- `stop.sh` stops the server
- `test.sh` verifies HTTP and WebSocket behavior
- `test_client.py` checks the tracking WebSocket pipeline
