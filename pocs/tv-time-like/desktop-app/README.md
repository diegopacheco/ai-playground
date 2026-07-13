# Reelmark Desktop

Reelmark Desktop is the macOS Electron edition of the existing watch journal. It renders the same React app, starts the same Bun API when needed, and reads and writes the same `data/reelmark.db` collection.

## Run

Requirements: macOS and Bun 1.3 or newer.

```bash
./install.sh
./run-desktop-app.sh
```

`install.sh` builds the React interface, generates the native Reelmark icon set, packages the Electron runtime, signs the local bundle, and installs `Reelmark.app` in `/Applications`. It keeps the existing repository database as the shared source of truth.

## Test

```bash
./test.sh
./uninstall.sh
```

Closing the app stops the API only when the desktop process started it. An API that was already running remains untouched. Bun is located automatically when the app is opened from Finder.
