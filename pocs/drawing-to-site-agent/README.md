# Drawing to Site Agent

Draw a UI sketch on a canvas (or drag an image) and an AI agent turns it into a working website.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | React 19, TanStack Router + Query, Vite, Bun |
| Backend | Rust, Axum 0.8, Tokio, SQLite (sqlx) |
| LLM | CLI subprocess (claude, codex, gemini, copilot) |
| Streaming | SSE via tokio::sync::broadcast |

## Agent Engines

| Engine | CLI | Model |
|--------|-----|-------|
| claude/opus | `claude` | opus |
| claude/sonnet | `claude` | sonnet |
| codex/gpt-5-2-codex | `codex` | gpt-5.2-codex |
| gemini/gemini-3-0 | `gemini` | gemini-3-0 |
| copilot/sonnet | `copilot` | claude-sonnet-4.5 |
| copilot/opus | `copilot` | claude-opus-4.6 |

## How to Run

```bash
./run.sh
```

Opens on http://localhost:5173 (frontend) with backend on http://localhost:3000.

```bash
./stop.sh
```

## How it Works

1. Pick an agent engine and name your project
2. Draw your UI on the canvas (or drag & drop an image)
3. Click Build - the drawing is captured as PNG and sent to the backend
4. The backend spawns the selected LLM CLI with a prompt to generate HTML/CSS/JS
5. Progress is streamed back via SSE (animated progress bar with elapsed timer)
6. The generated site is previewed in an iframe
7. All builds are stored in SQLite and visible in the History page

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/engines` | List agent engines |
| POST | `/api/projects` | Create project and start build |
| GET | `/api/projects` | List all projects |
| GET | `/api/projects/{id}` | Get project details |
| DELETE | `/api/projects/{id}` | Delete project |
| GET | `/api/projects/{id}/stream` | SSE build progress |
| GET | `/output/{id}/index.html` | Serve generated site |

## Project Structure

```
drawing-to-site-agent/
  run.sh / stop.sh
  frontend/
    src/
      pages/      SetupPage, CanvasPage, ProgressPage, PreviewPage, HistoryPage
      components/ EngineSelector, DrawingCanvas, BuildProgress, SitePreview
      api/        client.ts
  backend/
    src/
      agents/     runner.rs, claude.rs, codex.rs, gemini.rs, copilot.rs
      build/      engine.rs
      routes/     projects.rs, engines.rs
      persistence/ db.rs, models.rs
      sse/        broadcaster.rs
    output/       generated sites per project
```

## Requirements

- Rust 1.75+
- Bun
- At least one LLM CLI installed: `claude`, `codex`, `gemini`, or `copilot`
