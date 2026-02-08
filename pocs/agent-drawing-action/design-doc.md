# Design Doc: Agent Drawing Action

## Overview

A full-stack app where the user draws on an HTML5 canvas and an LLM agent guesses what the drawing is in 1-2 words. The app has two screens: a drawing canvas with a "Guess Name" button, and a history page showing all past drawings with their LLM predictions.

## Problem Statement

Testing multimodal LLM vision capabilities by having agents interpret freehand user drawings. The user draws something, the agent sees the image and responds with a short guess.

## Goals

- Canvas drawing screen with pen, colors, sizes, eraser, undo/redo, clear
- "Guess Name" button captures the canvas as PNG and sends to backend
- Backend spawns an LLM agent (CLI subprocess) that analyzes the image
- LLM responds with 1-2 words guessing what the drawing represents
- History screen shows all past drawings alongside the LLM prediction
- Multiple agent engines supported (Claude, Codex, Gemini, Copilot)

## Non-Goals

- Image generation (diffusion models)
- Multi-agent orchestration
- Authentication

## Architecture

```
+------------------+        +------------------------+        +------------------+
|                  |  POST  |                        |        |                  |
|  React Frontend  +------->+  Actix-Web Backend     +------->+  LLM Agent CLI   |
|  (TanStack)      |  JSON  |  (Rust + Tokio)        | spawn  |  (subprocess)    |
|                  |<-------+                        |<-------+                  |
+------------------+  JSON  +------------------------+  stdout+------------------+
       |                           |
       |                           v
       |                    +------+-------+
       |                    |   SQLite     |
       |                    |   drawings   |
       |                    +--------------+
       |
       v
  Two Screens:
  1. /         -> Drawing Canvas + "Guess Name" button
  2. /history  -> Table of past drawings + LLM guesses
```

## Tech Stack

| Component  | Technology                                      |
|------------|-------------------------------------------------|
| Frontend   | React 19, TanStack Router + Query, Vite, Bun, TS|
| Backend    | Rust, Tokio, Actix-Web                           |
| Database   | SQLite (via sqlx)                                |
| Agents     | CLI subprocesses (claude, codex, gemini, copilot)|
| Images     | HTML5 Canvas -> PNG base64                       |

## Screens

### Screen 1: Drawing (route: `/`)

- Full canvas area for freehand drawing
- Toolbar: Pen, Rectangle, Eraser tools
- Color palette (9 preset colors)
- Brush size selector (2px, 4px, 8px, 12px, 20px)
- Undo / Redo / Clear buttons
- Engine selector dropdown (Claude Opus, Claude Sonnet, Codex, Gemini, Copilot)
- **"Guess Name"** button: captures canvas as PNG, POSTs to backend, shows loading spinner, displays the LLM guess result inline

### Screen 2: History (route: `/history`)

- Table listing all past guesses
- Columns: thumbnail of drawing, LLM guess, engine used, timestamp, status
- Drawing image rendered inline (small preview)
- Delete button per entry
- Link back to drawing screen

## Flow

1. User draws on canvas
2. User selects an engine and clicks "Guess Name"
3. Frontend captures `canvas.toDataURL("image/png")` -> base64
4. Frontend POSTs `{ engine, image }` to `/api/guesses`
5. Backend decodes base64, saves PNG to `output/{id}/drawing.png`
6. Backend spawns LLM CLI subprocess with a prompt:
   "Look at this drawing. What is it? Answer with 1 or 2 words only."
7. LLM analyzes image, returns short text
8. Backend parses response, extracts 1-2 word answer
9. Backend saves result to SQLite, returns JSON to frontend
10. Frontend displays the guess to the user
11. History page queries `/api/guesses` to show all past results

## API Endpoints

| Method | Path                | Description                           |
|--------|---------------------|---------------------------------------|
| GET    | `/api/engines`      | List available LLM engines            |
| POST   | `/api/guesses`      | Submit drawing for guessing           |
| GET    | `/api/guesses`      | List all past guesses (history)       |
| GET    | `/api/guesses/{id}` | Get single guess                      |
| DELETE | `/api/guesses/{id}` | Delete a guess + its image            |
| GET    | `/output/{id}/*`    | Serve saved drawing PNGs              |

### POST /api/guesses request
```json
{
  "engine": "claude/opus",
  "image": "data:image/png;base64,iVBOR..."
}
```

### POST /api/guesses response
```json
{
  "id": "uuid",
  "guess": "Cat",
  "engine": "claude/opus",
  "status": "done"
}
```

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS guesses (
    id TEXT PRIMARY KEY,
    engine TEXT NOT NULL,
    guess TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT
)
```

Statuses: `pending`, `done`, `error`

## Agents

Agents are CLI tools spawned as subprocesses via `tokio::process::Command`. Each agent gets a prompt that includes the absolute path to the saved drawing PNG.

### Agent Prompt Template
```
You are an image analyst. I have a drawing saved at: {image_path}
Look at this drawing carefully. What is it? Answer with 1 or 2 words ONLY. Nothing else.
```

### Agent CLI Commands

| Engine             | Command                                                              |
|--------------------|----------------------------------------------------------------------|
| claude/opus        | `claude -p "<prompt>" --model opus --dangerously-skip-permissions`   |
| claude/sonnet      | `claude -p "<prompt>" --model sonnet --dangerously-skip-permissions` |
| codex/gpt-5-2-codex| `codex exec --full-auto --model gpt-5.2-codex "<prompt>"`           |
| gemini/gemini-3-0  | `gemini -y -p "<prompt>"`                                           |
| copilot/sonnet     | `copilot --allow-all --model claude-sonnet-4.5 -p "<prompt>"`       |
| copilot/opus       | `copilot --allow-all --model claude-opus-4.6 -p "<prompt>"`         |

### Agent Runner

- Spawns CLI as async subprocess with `tokio::process::Command`
- Captures stdout
- 480 second timeout
- Parses response, extracts first 1-2 words
- Returns result or error

## File Structure

```
agent-drawing-action/
  design-doc.md
  run.sh
  stop.sh
  frontend/
    package.json
    tsconfig.json
    vite.config.ts
    index.html
    src/
      main.tsx
      App.tsx
      router.tsx
      api/
        client.ts
      pages/
        DrawingPage.tsx
        HistoryPage.tsx
      components/
        DrawingCanvas.tsx
        EngineSelector.tsx
        GuessResult.tsx
  backend/
    Cargo.toml
    src/
      main.rs
      lib.rs
      agents/
        mod.rs
        runner.rs
        claude.rs
        codex.rs
        gemini.rs
        copilot.rs
      routes/
        mod.rs
        guesses.rs
        engines.rs
      persistence/
        mod.rs
        db.rs
        models.rs
    output/
      {guess_id}/
        drawing.png
```

## Shell Scripts

### run.sh
- Builds the Rust backend with `cargo build --release`
- Starts the backend on port 3001
- Runs `bun install` in frontend
- Starts Vite dev server on port 5173
- Saves PIDs to `.backend.pid` and `.frontend.pid`

### stop.sh
- Reads PID files and kills processes
- Cleans up PID files
- Kills any remaining orphan processes

## Risks and Mitigations

| Risk                              | Mitigation                                      |
|-----------------------------------|--------------------------------------------------|
| LLM returns verbose answer        | Parse response, take first 2 words only          |
| LLM agent CLI not installed       | Return clear error message to frontend           |
| Large canvas images               | Limit canvas resolution                          |
| Agent timeout                     | 480s timeout, kill subprocess on timeout         |

## Success Criteria

- User can draw on canvas and click "Guess Name"
- LLM agent returns a 1-2 word guess of the drawing
- Guess displays on screen immediately
- History page shows all past drawings with guesses
- Multiple engines selectable
- `run.sh` starts everything, `stop.sh` stops everything
