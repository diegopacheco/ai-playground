# macOS Apps and Chrome Extensions

A catalog of the macOS apps and Chrome extensions built across the `pocs/` folder with Swift, Electron, and Chrome Manifest V3.

Website: [macos-chrome-apps/index.html](https://diegopacheco.github.io/ai-playground/macos-chrome-apps/)

## macOS Apps

### 1. `cc-token-bar`

A native menu bar app for local Claude Code usage metrics.

What it does:

- Shows token usage, estimated cost, cache efficiency, session totals, tool activity, and model distribution.
- Reads local Claude Code data and refreshes through FSEvents.
- Uses Swift, SwiftUI, AppKit, CoreServices, and system Charts with no third-party libraries.

[Open `pocs/cc-token-bar`](https://github.com/diegopacheco/ai-playground/tree/main/pocs/cc-token-bar)

### 2. `fable-5-game-stand`

An Electron edition of Game Stand for macOS.

What it does:

- Organizes a finished-game collection across PlayStation, Nintendo Switch, and Steam.
- Starts the local Python and Flask service on a free port and displays it in a sandboxed Electron window.
- Installs a native `Game Stand.app` bundle with its own icon.

[Open `pocs/fable-5-game-stand`](https://github.com/diegopacheco/ai-playground/tree/main/pocs/fable-5-game-stand)

### 3. `tv-time-like`

The Reelmark private watch journal packaged as a macOS Electron app.

What it does:

- Tracks movies, shows, seasons, episodes, runtime, completion, and viewing patterns.
- Packages the React interface and Bun API while sharing the repository SQLite database.
- Builds, signs, and installs `Reelmark.app` with a native icon.

[Open `pocs/tv-time-like`](https://github.com/diegopacheco/ai-playground/tree/main/pocs/tv-time-like)

## Chrome Extensions

### 4. `chrome-ext-render-html-md`

GitHub Render turns HTML and Markdown source files into readable documents while browsing GitHub.

What it does:

- Renders repository and raw GitHub files in an in-page or full-page reader.
- Runs HTML inside an isolated Manifest V3 sandbox.
- Uses plain HTML, CSS, and JavaScript with no runtime libraries or analytics.

[Open `pocs/chrome-ext-render-html-md`](https://github.com/diegopacheco/ai-playground/tree/main/pocs/chrome-ext-render-html-md)

### 5. `flowprint`

A browser journey recorder that generates Playwright tests.

What it does:

- Records navigation, clicks, form activity, submissions, Fetch calls, and XMLHttpRequests.
- Produces Playwright TypeScript with durable locators and route assertions.
- Redacts passwords and stores the active session locally in Chrome.

[Open `pocs/flowprint`](https://github.com/diegopacheco/ai-playground/tree/main/pocs/flowprint)

### 6. `localhost-radar`

A Chrome DevTools operations panel for the local development surface.

What it does:

- Inspects Podman containers, listening services, and localhost browser traffic.
- Uses Chrome native messaging to reach a constrained Python host.
- Opens services, copies shell commands, and controls eligible local processes with validation.

[Open `pocs/localhost-radar`](https://github.com/diegopacheco/ai-playground/tree/main/pocs/localhost-radar)

## Notes

- The macOS catalog includes native Swift and Electron apps that install or run as macOS applications.
- The Chrome catalog includes manifests with `manifest_version: 3`; ordinary web-app manifests are excluded.
- All projects remain in `pocs/`; this directory provides the catalog and landing page.
