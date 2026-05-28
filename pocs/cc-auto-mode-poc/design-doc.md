# Design Doc: AR Point & Tell

> Built with **Claude Code Auto Mode** — the agent runs the implementation loop unattended (write → run → fix) while only stopping for decisions that matter.
> Reference: https://www.anthropic.com/engineering/claude-code-auto-mode

## Problem

You point your phone at a thing — a sofa, a hat, a plant — and want to know *what it is*: "a sofa from IKEA", "an SF Giants cap". Today that means opening a search app, describing it in words, and guessing. We want a single gesture: point the camera, tap, get a label.

## Solution

A **React Native + Expo** mobile app:

1. Live camera view (point at the object).
2. Tap a shutter button → capture a frame.
3. Send the frame to the **OpenAI vision API** with a prompt asking "what is this object?".
4. Display the answer over the captured frame ("This looks like an IKEA-style fabric sofa").

The camera preview is the "AR" surface: the result label is overlaid back onto the captured image so it reads as an annotation on the thing you pointed at.

## Why "Auto Mode"

This POC is a vehicle for Claude Code **Auto Mode**. The app is small and has tight, observable success criteria (camera opens, capture works, API returns a label, label renders), so the agent can loop independently — building, running on the Expo dev client, reading errors, and fixing — without prompting the user on every step. The user is only pulled in for genuine decisions (API key handling, model choice, UX trade-offs).

## Architecture

```
┌──────────────────────────────────────────┐
│              Expo / React Native App       │
│                                            │
│  ┌────────────┐   capture   ┌───────────┐  │
│  │ CameraView │────────────►│ Captured  │  │
│  │ (expo-     │             │ frame     │  │
│  │  camera)   │             │ (base64)  │  │
│  └────────────┘             └─────┬─────┘  │
│        ▲                          │        │
│        │ retake                   ▼        │
│  ┌────────────┐            ┌────────────┐  │
│  │  Result     │◄──────────│ OpenAI     │  │
│  │  overlay    │   label   │ client     │  │
│  └────────────┘            └─────┬──────┘  │
└──────────────────────────────────┼─────────┘
                                    │ HTTPS
                                    ▼
                       ┌────────────────────────┐
                       │ OpenAI Vision API       │
                       │ (gpt-4o / chat          │
                       │  completions, image in) │
                       └────────────────────────┘
```

## App Flow (states)

```
   ┌──────────┐  permission granted   ┌──────────┐
   │  Camera   │──────────────────────►│  Ready    │
   │ permission│                       │ (preview) │
   └──────────┘                       └─────┬─────┘
                                            │ tap shutter
                                            ▼
                                      ┌──────────┐
                                      │ Captured  │
                                      │ (freeze   │
                                      │  frame)   │
                                      └─────┬─────┘
                                            │ auto-send
                                            ▼
                                      ┌──────────┐  success  ┌──────────┐
                                      │ Analyzing │──────────►│ Result    │
                                      │ (spinner) │           │ (label    │
                                      └─────┬─────┘           │  overlay) │
                                            │ error           └─────┬─────┘
                                            ▼                       │ retake
                                      ┌──────────┐                  │
                                      │  Error    │◄─────────────────┘
                                      │ (retry)   │
                                      └──────────┘
```

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | React Native via **Expo** (managed workflow) | Fastest path to a camera app on a real device; Expo Go / dev client for live reload |
| Camera | **expo-camera** | First-party, handles permissions + capture, returns base64 |
| HTTP | built-in `fetch` | No extra dependency for one POST |
| AI | **OpenAI vision** (`gpt-4o`, chat completions with an image part) | Strong zero-shot object/brand recognition from a single image |
| Language | TypeScript | Type safety on the API response shape |

No state library, no navigation library — a single screen with local React state.

## Screens / Components

Single screen, state-driven:

- **`App.tsx`** — owns the state machine (`ready | captured | analyzing | result | error`).
- **`CameraScreen`** — `expo-camera` preview + shutter button. Requests permission on mount.
- **`ResultOverlay`** — shows the captured frame with the label rendered on top, plus a "Retake" button.
- **`openai.ts`** — wraps the OpenAI call: takes base64 image, returns `{ label: string }`.

## OpenAI Request Contract

### Request (chat completions, image input)

```json
{
  "model": "gpt-4o",
  "max_tokens": 100,
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "What is this object? Answer in one short phrase, include brand if recognizable (e.g. 'IKEA fabric sofa', 'SF Giants baseball cap')." },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,<BASE64>" } }
      ]
    }
  ]
}
```

### Response (relevant part)

```json
{
  "choices": [
    { "message": { "content": "An SF Giants baseball cap." } }
  ]
}
```

The app reads `choices[0].message.content`, trims it, and renders it as the label.

## API Key Handling

For a POC, the key lives in an Expo env var (`EXPO_PUBLIC_OPENAI_API_KEY`) loaded from a `.env` file and read via `process.env`. The app calls OpenAI directly over HTTPS.

**Caveat (documented, not solved here):** a key shipped in a mobile bundle is extractable. This is acceptable for a local POC on a personal device. The future hardening path is a thin backend proxy that holds the key and the app talks to the proxy — noted in *Future Work*, out of scope for the POC.

## Components (deliverable)

```
cc-auto-mode-poc/
├── design-doc.md
├── README.md
├── app/
│   ├── App.tsx
│   ├── src/
│   │   ├── CameraScreen.tsx
│   │   ├── ResultOverlay.tsx
│   │   └── openai.ts
│   ├── app.json
│   ├── package.json
│   ├── tsconfig.json
│   └── .env.example
├── start.sh        <-- expo start
└── test.sh         <-- shows the OpenAI call working against a sample image
```

## Edge Cases and Gaps

### Addressed

1. **Permission denied.** If camera permission is refused, show a message with a button to retry / open settings.
2. **Network / API failure.** Errors land in the `error` state with a "Try again" button; the captured frame is kept so the user can retry without re-aiming.
3. **Empty / unrecognizable image.** The prompt asks for a best-effort phrase; if the model is unsure it still returns a guess, which the UI shows verbatim.
4. **Large image payload.** Capture at reduced quality/resolution (`quality` option on `takePictureAsync`) to keep the base64 payload and latency reasonable.

### Known Limitations

1. **Not true continuous AR.** It is point → snapshot → label, not live frame-by-frame overlay. Real-time tracking is out of scope.
2. **Single object assumption.** The prompt asks for *the* object; a cluttered scene yields one label, not multiple.
3. **Key in bundle.** As above — fine for a POC, not for distribution.
4. **Latency.** A round trip to the vision API is a few seconds; the `analyzing` spinner covers it but it is not instant.
5. **Cost.** Each capture is a billed vision API call. No caching.
6. **iOS/Android device needed.** Camera does not work in a plain web/simulator without a physical device or dev client.

## Future Work

- Backend proxy to hide the API key.
- Continuous (live) labeling instead of snapshot.
- On-tap region selection so the user can point at one object in a busy scene.
- Local result history.

## Testing Strategy

- **`test.sh`** — runs the `openai.ts` call path against a bundled sample image (a known object) and asserts a non-empty label comes back, proving the API integration works without needing the phone.
- **Manual on-device** — `start.sh` launches Expo; verify on a real phone:
  - permission prompt appears and grants
  - shutter captures and freezes the frame
  - spinner shows during the call
  - a sensible label renders over the image
  - "Retake" returns to live preview
  - airplane-mode capture surfaces the error state with retry
