# Local Agent Orama - Implementation Plan

## Overview
Build an AI agent orchestration system with a React frontend and Rust backend that runs multiple AI coding assistants (Claude Code, Codex, Copilot CLI, Gemini CLI) in parallel, each with their own git worktree.

## Architecture

```
local-agent-orama/
├── frontend/           # TanStack Router + Vite + React 19 + bun
├── backend/            # Rust + Tokio + Actix-web
├── workspaces/         # Git worktrees for each agent per project
├── run.sh              # Build and run both
└── stop.sh             # Stop both services
```

## Frontend (frontend/)

**Tech Stack**: TanStack Router + Vite + React 19 + bun + Tailwind CSS

**Structure**:
```
frontend/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── routes/
│   │   └── index.tsx       # Main page with prompt input
│   ├── components/
│   │   ├── PromptInput.tsx # Centered prompt input
│   │   ├── AgentCard.tsx   # Card with status and green check when done
│   │   ├── FileBrowser.tsx # File tree for each implementation
│   │   ├── CodeViewer.tsx  # Display file contents
│   │   └── AgentTabs.tsx   # Tabs to switch between 4 implementations
│   └── api/
│       └── client.ts       # API calls to backend
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── package.json
└── tsconfig.json
```

**UI Layout**:
- Centered prompt input field at top
- Submit button
- 4 agent cards in a row (Claude Code, Codex, Copilot CLI, Gemini CLI)
  - Each card shows: agent name, default model name, status indicator
  - Green checkmark with green border when done
  - Spinner/loading indicator when running
  - Gray when pending
- Tab panel below to switch between 4 implementations
- File browser tree on left side of selected tab
- Code viewer on right side with syntax highlighting

## Backend (backend/)

**Tech Stack**: Rust + Tokio + Actix-web

**Structure**:
```
backend/
├── src/
│   ├── main.rs
│   ├── routes/
│   │   └── mod.rs          # API routes
│   ├── agents/
│   │   ├── mod.rs
│   │   ├── claude.rs       # Claude Code runner
│   │   ├── codex.rs        # Codex CLI runner
│   │   ├── copilot.rs      # Copilot CLI runner
│   │   └── gemini.rs       # Gemini CLI runner
│   ├── worktree/
│   │   └── mod.rs          # Git worktree management
│   ├── files/
│   │   └── mod.rs          # File listing and content serving
│   └── models/
│       └── mod.rs          # Request/Response types
├── Cargo.toml
└── Cargo.lock
```

**API Endpoints**:
- `POST /api/run` - Start agents with prompt
  - Request: `{ "prompt": "string", "project_name": "string" }`
  - Response: `{ "session_id": "uuid" }`
- `GET /api/status/{session_id}` - Get status of all agents
  - Response: `{ "agents": [{ "name": "claude", "status": "running|done|error|timeout", "worktree": "path" }] }`
- `GET /api/files/{session_id}/{agent_name}` - List files in agent's worktree
  - Response: `{ "files": [{ "path": "src/main.rs", "is_dir": false }] }`
- `GET /api/file/{session_id}/{agent_name}?path=src/main.rs` - Get file contents
  - Response: `{ "content": "file contents...", "language": "rust" }`

**Agent Execution Flow**:
1. Receive prompt and project name
2. Create base directory: `./workspaces/{project_name}/`
3. Initialize base git repo with initial commit
4. For each agent (claude, codex, copilot, gemini):
   - Create git worktree: `./workspaces/{project_name}/{agent_name}/`
   - Spawn CLI process with prompt in YOLO mode (no questions asked)
   - Track status (running/done/error/timeout)
   - 5 minute timeout per agent
5. Return status updates via polling

## CLI Agents (Local Machine)

All agents are CLI tools installed locally on the machine (no APIs used):

| Agent | CLI Command | Default Model |
|-------|-------------|---------------|
| Claude Code | `claude -p "prompt" --dangerously-skip-permissions` | claude-sonnet-4-20250514 |
| Codex | `codex --full-auto -q "prompt"` | o4-mini |
| Gemini | `gemini --yolo -p "prompt"` | gemini-2.5-pro |
| Copilot CLI | `copilot-cli --yolo "prompt"` | gpt-4o |

Each agent runs in YOLO mode (no questions asked) with a 5 minute timeout.

## Ports
- Frontend: 5173 (Vite default)
- Backend: 8080

## Usage

```bash
./run.sh    # Build and start both services
./stop.sh   # Stop both services
```

Open http://localhost:5173 and:
1. Enter a project name
2. Enter a prompt
3. Click "Run All Agents"
4. Watch the 4 agent cards show status
5. Browse files and view code when agents complete
