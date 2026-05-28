# AR Point & Tell

Point your phone's camera at an object, tap the shutter, and the app tells you what it is
("IKEA fabric sofa", "SF Giants baseball cap") using the OpenAI vision API.

Built with **Claude Code Auto Mode** — the agent ran the build → run → fix loop unattended,
stopping only for real decisions. See https://www.anthropic.com/engineering/claude-code-auto-mode

See [design-doc.md](./design-doc.md) for the full design.

## Stack

- React Native + Expo (SDK 54)
- expo-camera for the live preview and capture
- OpenAI vision (`gpt-4o`) for object identification
- TypeScript, no state/navigation libraries

## How it works

1. Live camera preview (point at the object).
2. Tap the shutter → the frame is captured as base64.
3. The frame is sent to the OpenAI vision API: "what is this object?".
4. The returned label is overlaid on the captured image. Tap "Retake" to go again.

State machine: `ready → analyzing → result / error`.

## Setup

```
cd app
cp env.example .env
```

Edit `app/.env` and set your key:

```
EXPO_PUBLIC_OPENAI_API_KEY=sk-...
```

## Run

From the POC root:

```
./start.sh
```

This installs dependencies (first run), seeds `.env` from `env.example` if missing, and starts
the Expo dev server **in the foreground** so the QR code shows in your terminal.

### Get it on your phone

1. Install **Expo Go** (App Store / Play Store).
2. Make sure your phone and this computer are on the **same Wi-Fi**.
3. **iOS:** open the Camera app, point it at the QR code, tap the banner to open in Expo Go.
   **Android:** open Expo Go → "Scan QR code".
4. The app loads over the network. Edits reload live.

Press `Ctrl+C` in the terminal to stop (or run `./stop.sh` from another terminal to kill a stray
server). If your phone can't reach the computer (different networks, VPN, firewall), run
`cd app && npx expo start --tunnel` instead.

## Test

Verifies the OpenAI integration without needing the phone (sends a sample image and asserts a
label comes back):

```
./test.sh
```

Requires `EXPO_PUBLIC_OPENAI_API_KEY` in `app/.env` or the environment.

## Notes

- The API key ships in the app bundle, which is fine for a local POC but not for distribution.
  The hardening path is a backend proxy that holds the key — see *Future Work* in the design doc.
- Each capture is a billed OpenAI vision call.
