# Multi-Agent System - Pixel Office Control Panel

A pixel-art virtual office where AI agents live, work, and execute tasks. Spawn agents into your office, assign them work, and watch them sit at their desks typing away. Built with a canvas-based pixel engine using sprites from [pixel-agents](https://github.com/pablodelucca/pixel-agents).

## Architecture

```
+-------------------------------------------------------------+
|               Frontend (React 19 + TypeScript)               |
|  +--------------------+  +-------------------------------+   |
|  |   Pixel Office     |  |     Control Panel             |   |
|  |   Canvas Engine    |  |  - Spawn Form                 |   |
|  |                    |  |  - Output Viewer (SSE)         |   |
|  |  - Sprites         |  |  - Chat Panel (interactive)    |   |
|  |  - A* Pathfinding  |  |  - Agent List + Clear All      |   |
|  |  - Animations      |  |                                |   |
|  +--------------------+  +-------------------------------+   |
|         Vite + Bun + TanStack Query + TypeScript              |
+------------------------------+-------------------------------+
                               | REST + SSE
+------------------------------+-------------------------------+
|                   Backend (Rust + Tokio)                      |
|  +----------+  +-----------+  +----------+  +--------+      |
|  |  Axum    |  |    SSE    |  |  Agent   |  | SQLite |      |
|  |  Routes  |  | Broadcast |  |  Runner  |  |   DB   |      |
|  +----------+  +-----------+  +----------+  +--------+      |
|                        |                                      |
|            +-----------+-----------+                          |
|            |   CLI Subprocess      |                          |
|            |  claude / gemini /    |                          |
|            |  copilot / codex      |                          |
|            +-----------------------+                          |
+--------------------------------------------------------------+
```

## Tech Stack

| Layer    | Technology                                      |
|----------|------------------------------------------------|
| Frontend | React 19, TypeScript, Vite 6, Bun, TanStack Query |
| Backend  | Rust 1.85+, Edition 2024, Tokio, Axum, SQLx    |
| Database | SQLite                                          |
| Realtime | Server-Sent Events (SSE)                        |
| Agents   | CLI subprocess (claude, gemini, copilot, codex) |
| Sprites  | pixel-agents by pablodelucca                    |

## Features

- **Canvas Pixel Office**: 20x14 tile grid with programmatic wood floors, carpet, rugs, and walls rendered on HTML Canvas
- **Animated Characters**: 6 character sprite sheets with walk cycles, sitting/typing animations, and directional movement
- **A* Pathfinding**: Agents walk from the entrance doorway to their assigned desk
- **Agent Spawning**: Pick an agent type (Claude/Gemini/Copilot/Codex), name it, give it a task
- **Real-time SSE Streaming**: Watch agent status changes and output in real time
- **Single Click = Output Logs**: Click an agent to see its task output (loads from DB for completed agents)
- **Double Click = Chat Panel**: Double-click for interactive chat with conversation history
- **Chat Thinking Animation**: Agent character types at desk while processing chat messages
- **Status Indicators**: Color-coded dots showing spawning/thinking/working/done/error
- **Speech Bubbles**: Agents show status in pixel-art speech bubbles
- **PC Screen Animations**: Monitors light up and flicker when agents are working
- **Unique Desk Assignment**: Each agent gets a different desk (6 desks available)
- **Clear All**: Button to wipe all agents and start fresh
- **SQLite Persistence**: All agents and messages stored for history

## How to Run

### Prerequisites
- Rust 1.85+ (edition 2024)
- Bun (for frontend)
- At least one CLI agent installed (claude, gemini, copilot, or codex)

### Start Everything
```bash
./run.sh
```

### Start Individually
```bash
cd backend && ./run.sh
cd frontend && ./run.sh
```

### Stop
```bash
./stop.sh
```

### Access
- Frontend: http://localhost:5173
- Backend API: http://localhost:3001

## API Endpoints

| Method | Path                      | Description               |
|--------|--------------------------|---------------------------|
| POST   | /api/agents/spawn        | Spawn a new agent         |
| GET    | /api/agents              | List all agents           |
| GET    | /api/agents/{id}         | Get agent with messages   |
| GET    | /api/agents/{id}/stream  | SSE stream for agent      |
| POST   | /api/agents/{id}/chat    | Send chat message         |
| DELETE | /api/agents/{id}         | Stop an agent             |
| DELETE | /api/agents/clear        | Clear all agents          |
| GET    | /api/agent-types         | List available types      |

## Agent Types

| Agent   | CLI     | Model        | Color   |
|---------|---------|--------------|---------|
| Claude  | claude  | opus         | Amber   |
| Gemini  | gemini  | gemini-3.0   | Blue    |
| Copilot | copilot | claude-sonnet-4 | Indigo |
| Codex   | codex   | gpt-5.4      | Green   |

## Project Structure

```
pixel-office-agents-fun/
+-- backend/
|   +-- Cargo.toml
|   +-- src/
|   |   +-- main.rs
|   |   +-- lib.rs
|   |   +-- routes/          (Axum handlers: spawn, stream, chat, clear)
|   |   +-- agents/          (CLI runners: claude, gemini, copilot, codex)
|   |   +-- sse/             (SSE broadcaster)
|   |   +-- persistence/     (SQLite layer)
|   +-- run.sh
|   +-- stop.sh
+-- frontend/
|   +-- package.json
|   +-- src/
|   |   +-- App.tsx
|   |   +-- canvas/          (Pixel engine: office, pathfinding, sprites)
|   |   +-- components/      (React UI: spawn form, output, chat, office)
|   |   +-- hooks/           (SSE + TanStack Query)
|   |   +-- api/             (HTTP client)
|   |   +-- types/
|   +-- public/assets/       (Sprites from pixel-agents)
|   +-- run.sh
|   +-- stop.sh
+-- run.sh
+-- stop.sh
+-- design-doc.md
+-- README.md
```

## Credits

- Pixel art sprites from [pixel-agents](https://github.com/pablodelucca/pixel-agents) by Pablo De Lucca
- Inspired by the agent-debate-club architecture pattern
