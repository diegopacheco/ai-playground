# Jurassic Runner — Design Doc

## 1. Goal

A browser endless-runner set in a prehistoric jungle where the **player's body is the
controller**. There is no keyboard in the intended experience:

- **jump** in real life → the runner jumps
- **crouch / duck** → the runner ducks
- **lean or step left / right** → the runner switches lanes

The game has a **timer**, a **score**, and **obstacles**. At the start it **snaps a photo of
your face onto the runner**. Your **live camera is shown on the right**. If you lose, the run
**auto-restarts after 3 seconds**.

## 2. Architecture

The webcam lives in the browser; **Python + OpenCV + MediaPipe** is the perception brain. Live
frames become a small control signal in a fast loop. This mirrors the sibling project
`open-can-enduro-game`, swapping hand tracking for full-body pose tracking.

![architecture](printscreens/architecture.png)

1. The browser grabs the webcam with `getUserMedia` and draws each frame to a hidden canvas.
2. That frame is JPEG-encoded and pushed to Python over a **WebSocket**.
3. Python decodes it with **OpenCV** (`cv2.imdecode`) and runs **MediaPipe PoseLandmarker**
   (33 body landmarks).
4. From the landmarks it derives a tiny JSON control packet and sends it back.
5. The game canvas turns that packet into lane / jump / duck and renders at 60 fps.

The forward path (solid) is the capture pipeline; the dotted blue path is the control signal
coming back.

### Why no LLM in the control loop

A runner needs to react ~30–60 times a second. An LLM call takes hundreds of milliseconds to
seconds, so routing every frame through a model would make the game unplayable. Control is
therefore **pure computer vision**. An LLM still fits this design — but only as a slow, async
*watcher* (auto-pause when nobody is in frame, coaching tips), never on the per-frame path.

## 3. The control packet

`pose_server.py` returns one JSON object per frame:

| Field | Meaning |
| --- | --- |
| `present` | a body was detected this frame |
| `x` | horizontal lane signal, `0..1`, mirrored so moving right moves the runner right |
| `y` | vertical signal (shoulder height) `0..1`, used for jump / duck |
| `face` | normalized bounding box `{x,y,w,h}` around the head, for the face snapshot |

- `x` is `1 - (shoulder_center_x)` — mirrored so the on-screen runner moves the same way you do.
- `y` is the shoulder midpoint height. It is compared against a calibrated baseline.
- `face` is computed from the nose / eyes / ears landmarks (indices 0–10) with padding, so the
  client can crop a clean head shot from the raw camera frame.

## 4. Pose → game mapping

All four controls come from two numbers plus a one-time calibration:

| Control | Source | Rule |
| --- | --- | --- |
| Lane (left/right) | `x` | `lane = clamp((x − 0.5 − deadzone) · gain, −1, 1)` with `gain ≈ 7.5`, `deadzone ≈ 0.03` |
| Jump | `y` | grounded **and** `baseline − y > 0.05` → launch a jump (then physics) |
| Duck | `y` | grounded **and** `y − baseline > 0.07` → duck (squash) while held |

**Calibration**: when a run begins, a 3-second countdown averages your neutral shoulder height
into `baseline` and grabs the `face` box to snapshot your face. Jump/duck are measured relative
to that baseline so the game adapts to your height and camera position.

The horizontal `gain`/`deadzone` were tuned up after first playtest: a modest lean/step now
reaches a side lane, instead of needing to move nearly across the frame.

## 5. Game design

A pseudo-3D perspective trail with three lanes. Obstacles spawn at the horizon, grow as they
approach, and resolve at the player line (`t = 0.9`).

| Obstacle | Avoid by |
| --- | --- |
| Boulder (`rock`) | **jump** over it, or switch lane |
| Pterodactyl (`ptero`) | **duck** under it, or switch lane |
| Tree (`tree`) | **switch lane** (too tall to jump, too low to duck) |

Each obstacle maps to one of the four controls, so every gesture matters. A single obstacle
spawns at a time, guaranteeing at least one safe lane.

- **Score**: +1 each few frames of survival, +10 per obstacle cleared.
- **Timer**: elapsed survival time, shown live.
- **Difficulty**: forward speed and spawn rate ramp with elapsed time.
- **Lose & restart**: any unavoided obstacle = crash. A T-Rex lunges in, and the run
  **auto-restarts after 3 seconds** (face and baseline are kept).

### Theming

Dusk-jungle palette, smoking volcano, parallax treeline, scrolling dirt trail, side ferns and
bushes, a comic display font, and a "your face on a jungle explorer" runner. The whole site is
themed to match the game.

### Audio

A dependency-free **procedural soundtrack** built on the Web Audio API (`audio.js`): a low
detuned drone, a tribal kick/tom/shaker groove, and a pentatonic flute riff for a jungle feel,
plus a synthesized T-Rex roar on game over. It starts on the first user gesture (browser
autoplay rules) and has a `SOUND: ON/OFF` toggle. No audio files, no libraries.

## 6. Keyboard fallback

For machines without a camera (and for automated screenshots), arrow keys + space drive the
runner. Keyboard input is **momentary**: the moment you stop pressing, body control resumes —
so a stray key press can never silently disable the camera.

## 7. Stack

| Piece | Choice |
| --- | --- |
| Body tracking | MediaPipe `PoseLandmarker` (lite, float16) |
| Frame decode | OpenCV (`opencv-python`) |
| Transport | `websockets` |
| Static server | Python stdlib `http.server` |
| Game | plain HTML canvas + vanilla JS |
| Music | Web Audio API, procedural |
| Python | 3.9 |

No game engine, no frontend framework, no build step, no audio assets.

## 8. Privacy

- The webcam never leaves your machine: frames go browser → local Python over `ws://localhost`.
- The browser owns the camera, so the Python side never opens a camera device — no OS camera
  prompt for the server.
- In the README screenshots, the camera panel is deliberately pixelated for privacy; on your
  machine it shows your sharp live feed with the lane dot.

## 9. Files

```
pose_server.py       WebSocket pose-tracking + static file server
web/index.html       layout
web/style.css        jungle theme
web/game.js          canvas game loop, camera capture, pose → control, face snapshot
web/audio.js         procedural jungle soundtrack (Web Audio API)
test_client.py       sends one frame through the pipeline
requirements.txt     mediapipe, opencv-python, websockets, numpy
start.sh stop.sh test.sh
diagram.html         source of the hand-drawn architecture image
design-doc.md        this document
```
