# Air Lango — Design Doc

## 1. Goal

A browser **2-D boxing game you fight with your bare hands**. There is no keyboard in the intended
experience — your webcam is the input device. Two boxers stand face to face in a light, bright ring
and trade blows:

- **fist thrust toward the camera** → that glove throws a **punch**
- **open palm(s) raised** → **block** (guard up)
- **hand down / neutral** → idle, guard down

A match is **3 rounds, no clock**. Each fighter has a **life bar worth 4 clean hits**; landing the
**4th unblocked punch** is a **KO** and wins the round. **Best of three** — the first fighter to win
**2 rounds** wins the match.

Two ways to play, chosen on the start screen:

| Mode | Who controls each fighter |
| --- | --- |
| **1P vs CPU** | You drive **one** boxer with **both** hands (left hand = left glove, right hand = right glove). The opponent is computer-controlled and ramps up each round. |
| **P1 vs P2** | Two people share one camera. The frame is split **left / right**: your left half drives Player 1, the right half drives Player 2. Each person uses **one** hand. |

The whole UI is a **light theme** around a bright gym ring, with the **live camera on screen** and a
tracking dot on each hand tinted by what it is doing (punch / block / idle). A **full-screen button**
sits in the header.

This is the boxing entry in the same family as `open-can-enduro-game` (hand → steering),
`jurassic-runner-game` (body → jump/duck) and `dashboard-cv` / Air Board (hand → drawing): turn
camera perception into a **continuous action** in a fast loop, with no LLM on the per-frame path.

## 2. Architecture

The webcam lives in the browser; **Python + OpenCV + MediaPipe** is the perception brain. Live
frames become a small control packet in a fast loop. This mirrors the sibling projects, swapping the
game for a fight.

1. The browser grabs the webcam with `getUserMedia` and draws each frame to a hidden canvas.
2. That frame is JPEG-encoded and pushed to Python over a **WebSocket**.
3. Python decodes it with **OpenCV** (`cv2.imdecode`) and runs **MediaPipe HandLandmarker** with
   `num_hands=2` (21 landmarks per hand).
4. For each hand it derives a tiny JSON record — palm position, which side of the frame it is on, and
   raw hand-shape flags (fist / open / palm scale) — and sends back an array of up to two hands.
5. The fight canvas turns those records into punches and blocks and renders at 60 fps.

The forward path is the capture pipeline; the return path is the control packet. The browser owns the
camera, so Python never opens a camera device — no OS camera prompt for the server. The static server
sends `Cache-Control: no-store` so the browser always loads the latest game files.

### Why no LLM in the fight loop

A boxer needs to react ~30–60 times a second; a punch that lands 400 ms late is a punch you already
ate. An LLM call takes hundreds of milliseconds to seconds, so routing every frame through a model
would make the fight unplayable. Combat is therefore **pure computer vision**. An LLM still fits this
design — but only as a slow, async *watcher* (auto-pause when nobody is in frame, a post-match
"coach" that reviews the round), never on the per-frame path.

## 3. The control packet

`server.py` returns one JSON object per frame holding **0, 1, or 2 hands**:

```json
{ "hands": [
  { "x": 0.30, "y": 0.42, "side": "L", "fist": true,  "open": false, "scale": 0.19 },
  { "x": 0.74, "y": 0.55, "side": "R", "fist": false, "open": true,  "scale": 0.12 }
] }
```

| Field | Meaning |
| --- | --- |
| `hands` | array of detected hands this frame, at most two |
| `x` | palm-center horizontal position, `0..1`, **mirrored** (`1 − x`) so moving right moves your glove right |
| `y` | palm-center vertical position, `0..1` (smaller = higher / raised) |
| `side` | `"L"` if `x < 0.5`, else `"R"` — which half of the frame the hand is in |
| `fist` | true when **no fingers are up** (a closed fist) |
| `open` | true when **three or more** of index/middle/ring/pinky are up (an open palm; tolerant of one jittery finger so a guard does not flicker off) |
| `scale` | apparent hand size, a **forward-depth proxy** — grows as the hand moves toward the camera |

The server stays a **thin, stateless perception function**, one two-hand detector per connection. It
reports raw per-frame shape; the **client** owns smoothing, the rolling thrust baseline, and all
edge-triggering (section 4). This is the same split the whiteboard uses for its colour gesture.

- `x` is `1 − palm.x` so the on-screen glove moves the same way your hand does.
- `x`/`y`/`scale` use the **palm center** (mean of wrist `0`, index-MCP `5`, pinky-MCP `17`) so they
  stay stable while the fingers curl and uncurl.
- `scale` is the palm width — the distance between landmarks `5` and `17`, normalized — which
  **increases as the hand approaches the lens**. That is the cheap, robust z-signal for a thrust
  (MediaPipe's raw `z` is too noisy to gate a punch on).

## 4. Gesture → fight mapping

A finger is **up** when its tip is higher (smaller `y`) than its PIP joint. From that the server sets
`fist` (nothing up) and `open` (**three or more** of the four fingers up — a deliberately forgiving
threshold so one mis-tracked finger never drops your guard mid-punch). The **client** turns the
per-frame flags into the three combat states:

| State | Condition (client) | Effect |
| --- | --- | --- |
| **Punch** | `fist` **and** `scale` spikes above this hand's rolling baseline (a forward thrust) | fire **one** punch on that glove (**whoosh** SFX), then lock until the hand pulls back |
| **Block** | `open` **and** the hand is raised (`y` above the guard line) | guard up — incoming punches on that fighter are absorbed (a **clink** + gold spark on the glove) |
| **Idle** | anything else (hand down, half-open, resting fist) | guard down, no punch |

### Thrust detection and edge-triggering

`scale` alone is not a punch — a hand can simply be close to the camera. Each hand keeps a **rolling
baseline** of its recent `scale`; a punch fires only when `scale` jumps past `baseline · 1.4` **and**
the hand is a `fist`. (The 1.4 ratio is set high on purpose: a smaller value let landmark jitter fire
phantom punches when the hand was held still.) The punch is **edge-triggered with a cooldown**: it fires once on the forward
thrust and will not fire again until `scale` falls back toward baseline (the hand is drawn back),
so one jab = one punch, not a burst. This is the same edge-trigger discipline the whiteboard uses to
step one colour per thumbs-up.

A `fist` and an `open` palm are mutually exclusive, so **a single hand cannot punch and block at the
same time** — that is the core tension. In **1P vs CPU** you have two hands, so you can guard with one
and jab with the other; in **P1 vs P2** each player has only one hand and must choose offense or
defense each moment.

### 4.1 Frame-split hand mapping (both modes)

The detector returns up to two hands in **no stable order**, so the client assigns them by `side`
(which half of the frame each palm is in):

- **P1 vs P2**: the **L** hand drives Player 1, the **R** hand drives Player 2. One hand per person,
  one fighter per side. A vertical divider is drawn in the camera panel so players can see the split.
- **1P vs CPU**: **both** of your hands feed the **one** player fighter — the **L** hand is its left
  glove, the **R** hand its right glove. The CPU drives the other fighter (section 5). Raising either
  open palm puts the player's guard up; raising both is a full guard.

If two hands land on the same side in a frame, the nearer-to-center one keeps the slot and the other
is ignored that frame — cheap and stable, no cross-frame tracking needed because the sides are fixed.

### Smoothing

Each glove's on-screen position **lerps** toward its target each frame to absorb landmark jitter, the
same trick the siblings use to keep motion smooth rather than shaky. Position is mostly cosmetic here
(the fighters are stationary, section 5) — what matters is the punch/block **state**, which is gated
by the thrust + edge-trigger above rather than by raw position.

The **guard is held for a few frames** after the block gesture last registered (`guardHold`). The
detector only sends ~16 packets a second, so without this a single dropped frame could leave you open
exactly as a punch lands; the short hold makes defense feel reliable rather than glitchy.

## 5. Game design

Both fighters stand at **fixed facing positions** in the ring (Punch-Out–style side view) and do not
walk around — all input is **hands only**, so there is no body-movement requirement. Every
interaction is **punch vs block timing**:

- A glove's punch always **reaches** the opponent (no aiming, no range to manage).
- If the defender's **guard is up** when the punch arrives → **blocked**: no life lost, a guard
  *clink*, a small knockback.
- If the guard is **down** → **clean hit**: the defender loses **one of four** life segments, flashes,
  and enters a short **hit-stun** (briefly cannot punch).

### Rounds, life, and winning

| Rule | Value |
| --- | --- |
| Life bar | **4 segments** — each clean hit removes one |
| KO | the **4th** clean hit empties the bar → round over |
| Round end | **only** by KO — **there is no clock** |
| Match | **best of 3** — first fighter to **2** round wins takes the match (so 2 or 3 rounds total) |

Round flow: a **"ROUND n"** title card → a **bell** and **"FIGHT!"** → the fighters trade until one
bar empties → a **KO** animation (the loser drops) → a round-result card and **round pips** update →
the next round, or the **match result** if someone has 2 wins. Both bars refill at the start of each
round. After the **match result** card the next **match starts automatically** (a fresh best-of-3) —
or press **R** to rematch immediately; the fight never dead-ends on a static screen.

### Your face on your fighter

At the **start of a match** a still is grabbed from the webcam and drawn on each boxer's **head**, so
you fight as yourself. In **1P vs CPU** the center of the frame becomes the player's face (the CPU
keeps a plain drawn head). In **P1 vs P2** the frame is split: the **left** half is Player 1's face,
the **right** half is Player 2's — the same L/R split that drives the gloves. With no camera the
boxers keep their plain drawn heads. The snapshot stays local; it is only ever drawn to the canvas.

### CPU opponent (1P vs CPU)

A small state machine: **approach → telegraph → strike → recover → guard**. The CPU **telegraphs**
each punch with a visible wind-up so you can raise your guard in time, and it **reads your guard**:
if your palms are up it feints or waits; when your guard drops it strikes. It also **blocks** some of
your punches. Difficulty **ramps per round**:

| Round | CPU behavior |
| --- | --- |
| 1 | slow punches, long telegraphs, rarely blocks — room to learn the gestures |
| 2 | faster punches, shorter telegraphs, blocks more often |
| 3 | fastest, feints before striking, quick guard — punish only real openings |

The CPU never wins by anything but landing 4 clean punches, same rules as the player.

### Modes on the start screen

`1P vs CPU` or `P1 vs P2`, chosen before the bell. The choice only changes **who owns which side's
hand** (section 4.1) and whether the right-side fighter is driven by a hand or by the CPU AI — the
ring, life bars, rounds, and KO rules are identical.

## 6. Layout & theming

| Region | Contents |
| --- | --- |
| Header | mode label, **round number**, **best-of-3 pips**, **FULLSCREEN** button |
| Top bar | two **life bars** (4 segments each), P1 left, P2 / CPU right, with names |
| Ring | the two boxers face to face on a bright canvas, light gym backdrop, ropes and turnbuckles; each boxer wears the matching player's **snapshotted face** on its head, raises a two-glove **guard**, and throws a gold **block spark** when it absorbs a punch |
| Camera panel | the mirrored live feed with a **tracking dot per hand** — **red** = punch, **blue** = block, **grey** = idle — and a center divider line marking the L/R split |

Light theme throughout: bright ring apron, pastel crowd, bold comic display font for **"FIGHT!"**,
**"KO!"**, and the round cards. The camera panel stays visible in full-screen so players can keep
seeing what the game thinks each hand is doing, and **full-screen keeps the same light background**
(the apron and `::backdrop` are forced light — there is no dark mode anywhere).

## 7. Audio

A dependency-free **procedural** layer on the Web Audio API (no audio files): a **bell** at the start
and end of each round, a **whoosh** on a thrown punch, a **thud** on a clean hit, a metallic **clink**
on a block, and a short crowd swell on a KO. It starts on the first user gesture (browser autoplay
rules) and has a `SOUND: ON/OFF` toggle — the same approach as `jurassic-runner-game`. The round bell
is intrinsic to "3 rounds", so it is part of the core build, not an extra.

## 8. Keyboard fallback

For machines without a camera (and for automated screenshots), keys drive the same primitives,
**momentary** so hand control resumes the instant you stop pressing:

| | Left punch | Right punch | Block |
| --- | --- | --- | --- |
| **Player 1** | `F` | `G` | `Space` |
| **Player 2** | `J` | `K` | `Enter` |

`R` restarts the match. In 1P vs CPU only the Player 1 keys are live (the CPU drives the other side).
A key press is a stroke *holder* sharing the same punch/block functions as a hand, so no path is a
special case.

## 9. Stack

| Piece | Choice |
| --- | --- |
| Hand tracking | MediaPipe `HandLandmarker` (float16), `num_hands=2` |
| Frame decode | OpenCV (`opencv-python`) |
| Transport | `websockets` |
| Static server | Python stdlib `http.server` (with `no-store` headers) |
| Game | plain HTML canvas + vanilla JS |
| Music & SFX | Web Audio API, procedural |
| Python | 3.9 |

No game engine, no frontend framework, no build step, no audio assets.

## 10. Privacy

- The webcam never leaves your machine: frames go browser → local Python over `ws://localhost`.
- Because the browser owns the camera, the Python side never opens a camera device — no OS camera
  prompt for the server.
- Any README screenshots will be captured with the camera disabled (the camera panel shows a
  placeholder), so no real face appears.

## 11. Files (planned)

```
server.py            WebSocket hand-tracking (num_hands=2) + static file server (no-store)
web/index.html       layout (ring + life bars + camera panel + mode/start screen)
web/style.css        light gym/ring theme
web/fight.js         canvas fight loop, camera capture, gesture → punch/block, hand-side mapping,
                     life bars, rounds, KO, CPU AI, keyboard fallback
web/audio.js         procedural bell / whoosh / thud / clink / crowd (Web Audio API)
test_client.py       sends one frame through the pipeline
requirements.txt     mediapipe, opencv-python, websockets, numpy
start.sh stop.sh test.sh
diagram.html         source of the hand-drawn architecture image
design-doc.md        this document
```
