# Air Board — Design Doc

## 1. Goal

A browser **whiteboard you draw on with your bare hands**. There is no pen and no mouse in the
intended experience — your webcam is the input device:

- **point** (index finger) → the pen is down and follows your fingertip
- **thumbs up** → cycle to the next colour
- **open hand** (five fingers, a "stop") → erase where your hand is
- **fist** → the pen is up (rest position)

The board has a **colour / size / eraser / undo / clear toolbar** (clicked with the mouse), and your
**live camera sits in a square on the right** with a dot tracking each fingertip so you can see
exactly where the pen is.
It **tracks up to two hands**, so a second person can draw on the same board at the same time. The
whole UI is a **light theme** around a light paper board.

This POC fills the gap called out in *Vision-Reactive Agents (camera + OpenCV)*: the other OpenCV
POCs run a fixed CV op and stop. Here perception (a hand) is turned into a **continuous action**
(ink on a canvas), and the same board snapshot is the natural hand-off point to a slow LLM
*watcher* loop (section 7).

## 2. Architecture

The webcam lives in the browser; **Python + OpenCV + MediaPipe** is the perception brain. Live
frames become a small control packet in a fast loop. This mirrors the sibling projects
`open-can-enduro-game` (hand → steering) and `jurassic-runner-game` (body → jump/duck), swapping
the game for a drawing surface.

![architecture](printscreens/architecture.png)

1. The browser grabs the webcam with `getUserMedia` and draws each frame to a hidden canvas.
2. That frame is JPEG-encoded and pushed to Python over a **WebSocket**.
3. Python decodes it with **OpenCV** (`cv2.imdecode`) and runs **MediaPipe HandLandmarker**
   (21 hand landmarks).
4. From the landmarks it derives a tiny JSON packet — the index-fingertip position and a gesture
   `mode` — and sends it back.
5. The board canvas turns that packet into a cursor, lays down strokes, and renders at 60 fps.

The forward path (solid) is the capture pipeline; the dotted blue path is the control packet coming
back. The browser owns the camera, so Python never opens a camera device — no OS camera prompt for
the server.

### Why no LLM in the drawing loop

A pen needs to follow your finger ~30–60 times a second. An LLM call takes hundreds of milliseconds
to seconds, so routing every fingertip sample through a model would make the pen lag uselessly.
Drawing is therefore **pure computer vision**. An LLM still fits this design — but only as a slow,
async *watcher* on the resulting image, never on the per-frame path. That watcher is section 7.

## 3. The control packet

`server.py` returns one JSON object per frame holding **0, 1, or 2 hands**:

```json
{ "hands": [ { "x": 0.42, "y": 0.31, "mode": "draw" }, { "x": 0.77, "y": 0.55, "mode": "erase" } ] }
```

| Field | Meaning |
| --- | --- |
| `hands` | array of detected hands this frame, at most two |
| `x` | fingertip horizontal position, `0..1`, mirrored so moving right moves the pen right |
| `y` | fingertip vertical position, `0..1` |
| `mode` | `"draw"`, `"erase"`, `"color"`, or `"idle"` — from which fingers are up |

- `x` is `1 − index_tip.x` so the on-screen pen moves the same way your hand does.
- `x`/`y` are landmark **8** (the index fingertip) for every mode.
- The array is unordered (MediaPipe does not guarantee a stable hand order), so the **client**
  matches hands to pens across frames (section 4.1).

Everything else — gain, smoothing, the toolbar, colour cycling, undo — lives in the browser. The
server stays a thin, stateless perception function, one two-hand detector per connection.

## 4. Gesture → pen

A finger is **up** when its tip landmark is higher on screen (smaller `y`) than its PIP joint. The
thumb counts as up only when its tip is **both** above its own MCP joint **and** above the index PIP
— i.e. it is clearly sticking up out of a closed fist, not just resting at the side. Each hand is
classified independently:

| Gesture | Fingers up | `mode` | Effect |
| --- | --- | --- | --- |
| **Point** | index only | `draw` | pen **down**, ink follows the fingertip |
| **Open hand** | index + middle + ring + pinky | `erase` | erase under the fingertip |
| **Thumbs up** | thumb only (clearly raised) | `color` | cycle to the next colour (once per gesture) |
| **Fist / other** | — | `idle` | pen **up**, rest |

```
index_up  = lm[8].y  < lm[6].y      ring_up  = lm[16].y < lm[14].y
middle_up = lm[12].y < lm[10].y     pinky_up = lm[20].y < lm[18].y
thumb_up  = lm[4].y  < lm[2].y and lm[4].y < lm[6].y
```

`erase` is checked before `draw` so a fully open hand always wins. **Pen vs eraser is decided live by
the gesture**, not by a sticky toggle: pointing always draws with the current colour, opening your
hand always erases — which is exactly "open your hand to rub it out, point to draw again". `color` is
**edge-triggered**: the colour advances once when a hand *enters* the thumbs-up pose and not again
until it leaves, so the palette steps one swatch per thumbs-up instead of racing through six colours
per second. The stricter `thumb_up` test (tip above the index PIP, not just the thumb MCP) is what
keeps a **plain fist** from being misread as a thumbs-up: a fist now reliably reads as `idle` (pen
up) and no longer churns through colours on its own.

### 4.1 Tracking two hands

The detector runs with `num_hands=2`, so the packet can carry two hands, but their order is not
stable frame to frame. The client keeps **two persistent pens** and, each frame, assigns the
incoming detections to pens by **nearest cursor** (a greedy 1-to-1 match on screen distance). This
keeps each person's stroke attached to the same pen instead of tearing between them when MediaPipe
reorders the hands. A pen with no detection this frame goes to `present: false` and commits its
current stroke. When two hands are live, each cursor is tinted (`#1`/`#2`) on the board and in the
camera square. The board has **one shared colour and size**; either person can cycle the colour with
a thumbs-up.

### Cursor mapping and smoothing

The normalized fingertip is expanded around the centre with a gain so the corners of the board are
reachable without moving your hand across the whole frame, then clamped:

```
gx = clamp(0.5 + (x − 0.5) · 1.35, 0, 1)
```

The on-screen cursor **lerps** toward that target each frame (factor `0.55`) to absorb landmark
jitter, which is what keeps lines smooth rather than shaky.

### Toolbar input

Bare hands cover the two changes you make mid-stroke: **thumbs-up cycles the colour** and an **open
hand switches to the eraser**, both without leaving the drawing surface. The remaining toolbar cells
— brush size, undo, clear — are picked by **clicking** them with the mouse, and undo / clear /
eraser / colour also have keyboard shortcuts (section 8). The board does not steer the cursor into
the top bar from a gesture, so moving a hand near the top edge never accidentally triggers a tool.

## 5. The ink model

Strokes are kept as data and painted onto an **offscreen layer canvas**, so undo and erase are
trivial and the grid shows through erased areas:

- A stroke is `{ color, size, erase, points[] }`.
- While the pen is down, each new fingertip sample appends a point and paints just the **newest
  segment** onto the layer (cheap, no full redraw).
- **Eraser** paints onto the same layer with `globalCompositeOperation = "destination-out"`, cutting
  the ink back to transparent so the background grid reappears — a real eraser, not white paint.
- **Undo** pops the last stroke and **replays** the remaining strokes onto a cleared layer.
- **Clear** empties the stroke list and clears the layer.

Each frame the visible canvas is composited as: background grid → ink layer → toolbar strip →
live cursor. Drawing is restricted to below the toolbar strip, so moving through the toolbar never
leaves stray ink.

## 6. Board layout

| Region | Contents |
| --- | --- |
| Top strip (`66 px`) | 6 colours, eraser, 3 brush sizes, undo, clear — evenly spaced cells |
| Drawing area | the rest of the canvas, a light grid |
| Right panel | HAND status (hands tracked), current tool, the **camera square**, a gesture legend, action buttons |

The camera square mirrors your video and draws a dot on each tracked fingertip — green for `draw`,
grey for `erase`, purple for `color`, and a neutral dot at rest (`idle`) — so you always know what
the board thinks each hand is doing. With two hands live the dots switch to the `#1`/`#2` tints to match the board
cursors.

## 7. The vision-reactive agent (slow watcher)

This is where the project earns the "vision-reactive agent" label without breaking the fast loop.
The drawing loop is pure CV; the **agent loop is separate and slow**:

- Every few seconds (or on a gesture / button), the browser exports a board snapshot — exactly the
  PNG that `SAVE PNG` produces — and hands it to a vision-capable LLM.
- The LLM acts on the *image*, not on individual frames: *"what did I draw?"*, *"clean this sketch
  into a neat diagram"*, *"label these boxes"*, *"is this flowchart missing a step?"*.
- Its answer comes back as an annotation, a caption, or a redrawn layer — never as per-frame pen
  control.

The architecture already isolates this cleanly: the same snapshot used for `SAVE PNG` is the agent's
input, and the agent runs off the render path so a multi-second model call can never stall the pen.
The current build ships the fast CV board; the watcher is the documented next step and needs a
provider key, which is why it is not wired by default.

## 8. Mouse fallback

For machines without a camera (and for automated screenshots), the mouse drives the same
primitives: press in the drawing area to draw, release to commit a stroke, click a toolbar cell to
pick a tool. Keyboard shortcuts: `z` undo, `c` clear, `e` toggle eraser, `x` cycle colour, `f`
full-screen. The mouse and each hand are just stroke *holders* sharing one set of stroke functions
and the shared colour/size, so no path is a special case — the hands resume control on the next
frame after the mouse goes idle.

## 9. Stack

| Piece | Choice |
| --- | --- |
| Hand tracking | MediaPipe `HandLandmarker` (float16) |
| Frame decode | OpenCV (`opencv-python`) |
| Transport | `websockets` |
| Static server | Python stdlib `http.server` |
| Board | plain HTML canvas + vanilla JS |
| Python | 3.9 |

No drawing library, no frontend framework, no build step.

## 10. Privacy

- The webcam never leaves your machine: frames go browser → local Python over `ws://localhost`.
- Because the browser owns the camera, the Python side never opens a camera device — no OS camera
  prompt for the server.
- The README screenshots were captured with the camera disabled (the square shows "waiting for
  camera"), so no real face appears.

## 11. Files

```
server.py            WebSocket hand-tracking + static file server
web/index.html       layout (board + camera square + side panel)
web/style.css        dashboard theme around a light board
web/board.js         canvas board, camera capture, gesture → cursor/draw, toolbar, ink layer
test_client.py       sends one frame through the pipeline
requirements.txt     mediapipe, opencv-python, websockets, numpy
start.sh stop.sh test.sh
diagram.html         source of the hand-drawn architecture image
design-doc.md        this document
```
