# Drawing to Site Agent - Design Doc

## Overview

A full-stack web application where the user draws a UI on a canvas and an AI agent turns that drawing into a working website. The user picks an agent engine (Claude, Codex, Gemini, Copilot), draws on a canvas, and the backend captures the canvas as an image, sends it to the selected LLM via CLI subprocess, and generates HTML/CSS/JS. The generated site is previewed in an iframe and stored in history.

## Problem

Turning UI sketches into working websites is manual and slow. This app lets anyone draw a rough sketch and get a working site back in seconds, powered by multiple LLM engines.

## Architecture

```
+------------------------------------------------------------------+
|                 FRONTEND (React 19 + TanStack + Vite + Bun)      |
|                                                                  |
|  +------------+  +------------+  +-----------+  +-------------+  |
|  | 1. Engine  |  | 2. Canvas  |  | 4.Progress|  | 5. Preview  |  |
|  |   Select   |->|   Draw     |->|    Bar    |->|   (iframe)  |  |
|  +------------+  +------------+  +-----------+  +-------------+  |
|                                                                  |
|  +--------------------------------------------------------------+|
|  | 6. History - list previous builds, click to reload           ||
|  +--------------------------------------------------------------+|
+------------------------------|-----------------------------------+
                               | REST + SSE
                               v
+------------------------------------------------------------------+
|                  BACKEND (Rust + Axum 0.8 + Tokio)               |
|                                                                  |
|  +------------+  +-------------+  +------------+  +-----------+  |
|  |  Routes    |->| Agent Runner|->| CLI Spawn  |->| LLM CLI   |  |
|  |  (axum)    |  | (tokio proc)|  | (subprocess)|  | (claude,  |  |
|  +------------+  +-------------+  +------------+  | gemini,   |  |
|                                                   | copilot,  |  |
|  +------------+  +-------------+                  | codex)    |  |
|  |  SSE       |  | Persistence |                  +-----------+  |
|  | Broadcast  |  | (SQLite)    |                                 |
|  +------------+  +-------------+                                 |
|                                                                  |
|  +--------------------------------------------------------------+|
|  | Static File Server - serves generated sites from output/     ||
|  +--------------------------------------------------------------+|
+------------------------------------------------------------------+
```

## UI Screens / Flow

### Screen 1 - Project Setup
- Dropdown to select agent engine:
  - `claude/opus`
  - `claude/sonnet`
  - `codex/gpt-5-3-codex`
  - `gemini/gemini-3-0`
  - `copilot/sonnet`
  - `copilot/opus`
- Text input for project name
- "Next" button to proceed to canvas

### Screen 2 - Canvas Drawing
- Full-screen HTML5 Canvas with drawing tools:
  - Freehand pen (multiple colors, sizes)
  - Rectangle tool
  - Text tool
  - Eraser
  - Undo/Redo
  - Clear canvas
- "Build" button to send drawing to the agent

### Screen 3 - Canvas Capture & Send
- On "Build" click, the frontend:
  - Captures the canvas as a PNG via `canvas.toDataURL("image/png")`
  - Sends the base64 image + selected engine + project name to backend via POST
- This is automatic, not a separate visible screen

### Screen 4 - Progress Bar
- After sending, a progress view appears
- SSE stream from backend pushes status updates:
  - `analyzing_drawing` - agent is reading the canvas
  - `generating_code` - agent is writing HTML/CSS/JS
  - `saving_files` - writing output to disk
  - `done` - build complete
- Progress bar animates through each step

### Screen 5 - Preview
- Once done, the generated site is loaded in an iframe
- The iframe points to a static file route served by the backend: `/output/{project_id}/index.html`
- Buttons: "Back to Canvas" (edit and rebuild), "New Project", "View History"

### Screen 6 - History
- Lists all previous builds with: project name, engine used, timestamp
- Click a row to load that build's preview in the iframe
- Delete button to remove a build

## Tech Stack

| Layer         | Technology                                     |
|---------------|-----------------------------------------------|
| Frontend      | React 19, TanStack Router + Query, Vite, Bun |
| Backend       | Rust, Axum 0.8, Tokio                        |
| LLM Calls     | CLI subprocess via `tokio::process::Command`  |
| Database      | SQLite via sqlx 0.8                           |
| Streaming     | Axum SSE + `tokio::sync::broadcast`          |
| Static Files  | tower-http static file serving               |
| CORS          | tower-http CorsLayer                         |

## Cargo.toml Dependencies

Matches the agent-debate-club pattern exactly:

```toml
[package]
name = "drawing-to-site-backend"
version = "0.1.0"
edition = "2024"

[dependencies]
tokio = { version = "1", features = ["full", "process"] }
axum = { version = "0.8", features = ["macros"] }
axum-extra = { version = "0.10", features = ["typed-header"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite"] }
tower-http = { version = "0.6", features = ["cors", "fs"] }
tokio-stream = { version = "0.1", features = ["sync"] }
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
futures = "0.3"
async-stream = "0.3"
base64 = "0.22"
```

## LLM Agent Engines

Each engine maps to a CLI tool installed on the system. The backend spawns a subprocess with the appropriate flags. Same pattern as agent-debate-club: each agent has a `build_command(prompt) -> (String, Vec<String>)` function.

| Engine              | CLI Command | Flags                                                  |
|---------------------|-------------|--------------------------------------------------------|
| claude/opus         | `claude`    | `-p <prompt> --model opus --dangerously-skip-permissions` |
| claude/sonnet       | `claude`    | `-p <prompt> --model sonnet --dangerously-skip-permissions` |
| codex/gpt-5-3-codex | `codex`     | `exec --full-auto --model gpt-5.3-codex <prompt>`     |
| gemini/gemini-3-0   | `gemini`    | `-y -p <prompt>`                                       |
| copilot/sonnet      | `copilot`   | `--allow-all --model claude-sonnet-4 -p <prompt>`     |
| copilot/opus        | `copilot`   | `--allow-all --model claude-opus-4 -p <prompt>`       |

## Backend Rust Code Structure

### AppState (lib.rs)

Same pattern as agent-debate-club:

```rust
#[derive(Clone)]
pub struct AppState {
    pub pool: Pool<Sqlite>,
    pub broadcaster: Arc<Broadcaster>,
}
```

### main.rs - Axum Server

```rust
use axum::{routing::{get, post, delete}, Router};
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    let pool = db::init_db().await;
    let broadcaster = Broadcaster::new();
    let state = AppState { pool, broadcaster };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/engines", get(engines::get_engines))
        .route("/api/projects", post(projects::create_project))
        .route("/api/projects", get(projects::get_projects))
        .route("/api/projects/{id}", get(projects::get_project))
        .route("/api/projects/{id}", delete(projects::delete_project))
        .route("/api/projects/{id}/stream", get(projects::project_stream))
        .nest_service("/output", tower_http::services::ServeDir::new("output"))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

### agents/mod.rs

```rust
pub mod runner;
pub mod claude;
pub mod gemini;
pub mod copilot;
pub mod codex;

use runner::AgentRunner;

pub fn get_runner(engine: &str) -> AgentRunner {
    AgentRunner::new(engine)
}

pub fn get_available_engines() -> Vec<(String, String)> {
    vec![
        ("claude/opus".to_string(), "Claude Opus".to_string()),
        ("claude/sonnet".to_string(), "Claude Sonnet".to_string()),
        ("codex/gpt-5-3-codex".to_string(), "Codex GPT-5.3".to_string()),
        ("gemini/gemini-3-0".to_string(), "Gemini 3.0".to_string()),
        ("copilot/sonnet".to_string(), "Copilot Sonnet".to_string()),
        ("copilot/opus".to_string(), "Copilot Opus".to_string()),
    ]
}
```

### agents/runner.rs

Same `AgentRunner` struct as agent-debate-club. Spawns CLI subprocess, 120s timeout, captures stdout:

```rust
pub struct AgentRunner {
    pub engine: String,
}

impl AgentRunner {
    pub fn new(engine: &str) -> Self {
        Self { engine: engine.to_lowercase() }
    }

    pub async fn run(&self, prompt: &str) -> Result<String, String> {
        let (cmd, args) = self.build_command(prompt);

        let mut child = Command::new(&cmd)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn {}: {}", self.engine, e))?;

        let result = timeout(Duration::from_secs(120), async {
            let mut stdout = child.stdout.take().unwrap();
            let mut output = String::new();
            stdout.read_to_string(&mut output).await.map_err(|e| e.to_string())?;
            child.wait().await.map_err(|e| e.to_string())?;
            Ok::<String, String>(output)
        }).await;

        match result {
            Ok(Ok(output)) => {
                let trimmed = output.trim().to_string();
                if trimmed.is_empty() {
                    Err(format!("Engine {} returned empty response", self.engine))
                } else {
                    Ok(trimmed)
                }
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                let _ = child.kill().await;
                Err(format!("Engine {} timed out after 120s", self.engine))
            }
        }
    }

    fn build_command(&self, prompt: &str) -> (String, Vec<String>) {
        match self.engine.as_str() {
            "claude/opus" => claude::build_command(prompt, "opus"),
            "claude/sonnet" => claude::build_command(prompt, "sonnet"),
            "codex/gpt-5-3-codex" => codex::build_command(prompt),
            "gemini/gemini-3-0" => gemini::build_command(prompt),
            "copilot/sonnet" => copilot::build_command(prompt, "claude-sonnet-4"),
            "copilot/opus" => copilot::build_command(prompt, "claude-opus-4"),
            _ => ("echo".to_string(), vec!["Unknown engine".to_string()]),
        }
    }
}
```

### agents/claude.rs

```rust
pub fn build_command(prompt: &str, model: &str) -> (String, Vec<String>) {
    (
        "claude".to_string(),
        vec![
            "-p".to_string(),
            prompt.to_string(),
            "--model".to_string(),
            model.to_string(),
            "--dangerously-skip-permissions".to_string(),
        ],
    )
}
```

### agents/codex.rs

```rust
pub fn build_command(prompt: &str) -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "exec".to_string(),
            "--full-auto".to_string(),
            "--model".to_string(),
            "gpt-5.3-codex".to_string(),
            prompt.to_string(),
        ],
    )
}
```

### agents/gemini.rs

```rust
pub fn build_command(prompt: &str) -> (String, Vec<String>) {
    (
        "gemini".to_string(),
        vec![
            "-y".to_string(),
            "-p".to_string(),
            prompt.to_string(),
        ],
    )
}
```

### agents/copilot.rs

```rust
pub fn build_command(prompt: &str, model: &str) -> (String, Vec<String>) {
    (
        "copilot".to_string(),
        vec![
            "--allow-all".to_string(),
            "--model".to_string(),
            model.to_string(),
            "-p".to_string(),
            prompt.to_string(),
        ],
    )
}
```

### sse/broadcaster.rs

Same broadcast pattern as agent-debate-club using `tokio::sync::broadcast`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuildEvent {
    StatusUpdate { step: String, progress: u8 },
    BuildComplete { project_id: String },
    Error { message: String },
}

pub struct Broadcaster {
    channels: RwLock<HashMap<String, broadcast::Sender<BuildEvent>>>,
}

impl Broadcaster {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            channels: RwLock::new(HashMap::new()),
        })
    }

    pub async fn create_channel(&self, project_id: &str) -> broadcast::Receiver<BuildEvent> {
        let mut channels = self.channels.write().await;
        let (tx, rx) = broadcast::channel(100);
        channels.insert(project_id.to_string(), tx);
        rx
    }

    pub async fn subscribe(&self, project_id: &str) -> Option<broadcast::Receiver<BuildEvent>> {
        let channels = self.channels.read().await;
        channels.get(project_id).map(|tx| tx.subscribe())
    }

    pub async fn broadcast(&self, project_id: &str, event: BuildEvent) {
        let channels = self.channels.read().await;
        if let Some(tx) = channels.get(project_id) {
            let _ = tx.send(event);
        }
    }

    pub async fn remove_channel(&self, project_id: &str) {
        let mut channels = self.channels.write().await;
        channels.remove(project_id);
    }
}
```

### routes/projects.rs - SSE Stream

Same Axum SSE pattern as agent-debate-club:

```rust
pub async fn project_stream(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.broadcaster.subscribe(&id).await;

    let stream = async_stream::stream! {
        if let Some(rx) = rx {
            let mut stream = BroadcastStream::new(rx);
            loop {
                use tokio_stream::StreamExt;
                match stream.next().await {
                    Some(Ok(event)) => {
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        let event_type = match &event {
                            BuildEvent::StatusUpdate { .. } => "status",
                            BuildEvent::BuildComplete { .. } => "build_complete",
                            BuildEvent::Error { .. } => "error",
                        };
                        yield Ok(Event::default().event(event_type).data(data));
                    }
                    Some(Err(_)) => continue,
                    None => break,
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

### routes/projects.rs - Create Project

Same `tokio::spawn` pattern as agent-debate-club debate creation:

```rust
pub async fn create_project(
    State(state): State<AppState>,
    Json(req): Json<CreateProjectRequest>,
) -> Json<CreateProjectResponse> {
    let id = Uuid::new_v4().to_string();

    let project = ProjectRecord {
        id: id.clone(),
        name: req.name.clone(),
        engine: req.engine.clone(),
        status: "pending".to_string(),
        created_at: Utc::now().to_rfc3339(),
        completed_at: None,
        error: None,
    };

    let _ = db::create_project(&state.pool, &project).await;
    let _ = state.broadcaster.create_channel(&id).await;

    let pool = state.pool.clone();
    let broadcaster = state.broadcaster.clone();
    let project_id = id.clone();
    let image_data = req.image.clone();
    let engine = req.engine.clone();

    tokio::spawn(async move {
        let builder = BuildEngine::new(pool, broadcaster);
        builder.run_build(project_id, engine, image_data).await;
    });

    Json(CreateProjectResponse { id })
}
```

### build/engine.rs - Build Orchestration

Same pattern as agent-debate-club DebateEngine:

```rust
pub struct BuildEngine {
    pool: Pool<Sqlite>,
    broadcaster: Arc<Broadcaster>,
}

impl BuildEngine {
    pub fn new(pool: Pool<Sqlite>, broadcaster: Arc<Broadcaster>) -> Self {
        Self { pool, broadcaster }
    }

    pub async fn run_build(&self, project_id: String, engine: String, image_data: String) {
        self.broadcaster.broadcast(&project_id, BuildEvent::StatusUpdate {
            step: "analyzing_drawing".to_string(), progress: 25,
        }).await;

        let output_dir = format!("output/{}", project_id);
        std::fs::create_dir_all(&output_dir).unwrap();

        let image_bytes = base64::decode(&image_data).unwrap();
        let image_path = format!("{}/drawing.png", output_dir);
        std::fs::write(&image_path, &image_bytes).unwrap();

        let prompt = format!(
            r#"You are a web developer. I have a drawing of a website UI.
The drawing is saved at: {}

Look at the drawing and generate a complete, working website that matches it.

Requirements:
- Generate exactly 3 files: index.html, style.css, script.js
- The HTML must be semantic and responsive
- The CSS must be clean, modern, and match the layout in the drawing
- Add interactivity with vanilla JS where appropriate
- Do not use any frameworks or libraries
- Write all files to: {}

Output the files now."#,
            image_path, output_dir
        );

        self.broadcaster.broadcast(&project_id, BuildEvent::StatusUpdate {
            step: "generating_code".to_string(), progress: 50,
        }).await;

        let runner = get_runner(&engine);
        match runner.run(&prompt).await {
            Ok(response) => {
                self.broadcaster.broadcast(&project_id, BuildEvent::StatusUpdate {
                    step: "saving_files".to_string(), progress: 75,
                }).await;

                parse_and_save_files(&response, &output_dir);

                db::update_project_status(&self.pool, &project_id, "done", None).await;

                self.broadcaster.broadcast(&project_id, BuildEvent::StatusUpdate {
                    step: "done".to_string(), progress: 100,
                }).await;

                self.broadcaster.broadcast(&project_id, BuildEvent::BuildComplete {
                    project_id: project_id.clone(),
                }).await;
            }
            Err(e) => {
                db::update_project_status(&self.pool, &project_id, "error", Some(&e)).await;

                self.broadcaster.broadcast(&project_id, BuildEvent::Error {
                    message: e,
                }).await;
            }
        }

        self.broadcaster.remove_channel(&project_id).await;
    }
}
```

### persistence/models.rs

```rust
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ProjectRecord {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub status: String,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateProjectRequest {
    pub name: String,
    pub engine: String,
    pub image: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateProjectResponse {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineInfo {
    pub id: String,
    pub name: String,
}
```

### persistence/db.rs

Same sqlx pattern as agent-debate-club:

```rust
pub async fn init_db() -> Pool<Sqlite> {
    let pool = SqlitePool::connect("sqlite:projects.db?mode=rwc").await.unwrap();

    sqlx::query(r#"
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            engine TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
        )
    "#).execute(&pool).await.unwrap();

    pool
}

pub async fn create_project(pool: &Pool<Sqlite>, project: &ProjectRecord) -> Result<(), sqlx::Error> { ... }
pub async fn update_project_status(pool: &Pool<Sqlite>, id: &str, status: &str, error: Option<&str>) -> Result<(), sqlx::Error> { ... }
pub async fn get_project(pool: &Pool<Sqlite>, id: &str) -> Result<Option<ProjectRecord>, sqlx::Error> { ... }
pub async fn get_all_projects(pool: &Pool<Sqlite>) -> Result<Vec<ProjectRecord>, sqlx::Error> { ... }
pub async fn delete_project(pool: &Pool<Sqlite>, id: &str) -> Result<(), sqlx::Error> { ... }
```

## Backend API

| Method | Path                          | Description                              |
|--------|-------------------------------|------------------------------------------|
| GET    | `/api/engines`                | List available agent engines             |
| POST   | `/api/projects`               | Create project, send drawing, start build|
| GET    | `/api/projects`               | List all projects (history)              |
| GET    | `/api/projects/{id}`          | Get project details                      |
| DELETE | `/api/projects/{id}`          | Delete project and its files             |
| GET    | `/api/projects/{id}/stream`   | SSE stream of build progress             |
| GET    | `/output/{id}/index.html`     | Serve generated site (static via tower)  |
| GET    | `/output/{id}/{file}`         | Serve any generated file (css, js, etc)  |

### POST /api/projects - Request Body

```json
{
  "name": "my-landing-page",
  "engine": "claude/opus",
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

### SSE Events (GET /api/projects/{id}/stream)

```
event: status
data: {"type":"status_update","step":"analyzing_drawing","progress":25}

event: status
data: {"type":"status_update","step":"generating_code","progress":50}

event: status
data: {"type":"status_update","step":"saving_files","progress":75}

event: status
data: {"type":"status_update","step":"done","progress":100}

event: build_complete
data: {"type":"build_complete","project_id":"uuid-here"}

event: error
data: {"type":"error","message":"Engine timed out after 120s"}
```

## File Structure

```
drawing-to-site-agent/
├── design-doc.md
├── run.sh
├── stop.sh
├── frontend/
│   ├── package.json
│   ├── bun.lock
│   ├── vite.config.ts
│   ├── index.html
│   ├── tsconfig.json
│   ├── run.sh
│   ├── stop.sh
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── router.tsx
│       ├── api/
│       │   └── client.ts
│       ├── pages/
│       │   ├── SetupPage.tsx
│       │   ├── CanvasPage.tsx
│       │   ├── ProgressPage.tsx
│       │   ├── PreviewPage.tsx
│       │   └── HistoryPage.tsx
│       └── components/
│           ├── EngineSelector.tsx
│           ├── DrawingCanvas.tsx
│           ├── BuildProgress.tsx
│           └── SitePreview.tsx
├── backend/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   ├── lib.rs
│   │   ├── agents/
│   │   │   ├── mod.rs
│   │   │   ├── runner.rs
│   │   │   ├── claude.rs
│   │   │   ├── gemini.rs
│   │   │   ├── copilot.rs
│   │   │   └── codex.rs
│   │   ├── build/
│   │   │   ├── mod.rs
│   │   │   └── engine.rs
│   │   ├── routes/
│   │   │   ├── mod.rs
│   │   │   ├── projects.rs
│   │   │   └── engines.rs
│   │   ├── persistence/
│   │   │   ├── mod.rs
│   │   │   ├── db.rs
│   │   │   └── models.rs
│   │   └── sse/
│   │       ├── mod.rs
│   │       └── broadcaster.rs
│   └── output/
│       └── {project_id}/
│           ├── drawing.png
│           ├── index.html
│           ├── style.css
│           └── script.js
```

## Scripts

### run.sh (root)
- Starts backend (`cargo run` in backend/) on port 3000
- Starts frontend (`bun run dev` in frontend/) on port 5173
- Runs both in background, stores PIDs

### stop.sh (root)
- Kills both processes by stored PID

### frontend/run.sh
- `bun install && bun run dev`

### frontend/stop.sh
- Kills the vite dev server

## Frontend Details

### TanStack Router
- `/` - SetupPage (engine select + project name)
- `/canvas/:projectId` - CanvasPage (drawing)
- `/progress/:projectId` - ProgressPage (build progress bar)
- `/preview/:projectId` - PreviewPage (iframe with result)
- `/history` - HistoryPage (list of past builds)

### TanStack Query
- `useEngines()` - GET /api/engines
- `useCreateProject()` - POST /api/projects (mutation)
- `useProjects()` - GET /api/projects
- `useProject(id)` - GET /api/projects/{id}
- `useDeleteProject(id)` - DELETE /api/projects/{id}

### Canvas Implementation
- HTML5 Canvas element
- Mouse/touch event handlers for drawing
- Tools: pen, rectangle, text, eraser
- Color picker and stroke size
- `canvas.toDataURL("image/png")` to capture as base64

## Constraints

- Generated sites must be self-contained (no external CDN dependencies)
- No frontend frameworks in generated output (vanilla HTML/CSS/JS only)
- CLI tools (claude, gemini, copilot, codex) must be installed on the host
- Single agent call per build (prompt includes full context)
- 120-second timeout for agent execution
- Backend uses Axum 0.8 (not Actix) to match agent-debate-club pattern
- All async via Tokio, no blocking calls
