# Connect Four: Agent vs Agent - Design Document

## Overview

A Connect Four game where two AI agents compete against each other. The game features real-time visualization via SSE, match history persistence, and support for multiple AI agent backends (Claude, Gemini, Copilot, Codex).

## Game Rules

- 7 columns x 6 rows grid
- Players alternate dropping pieces (X and O)
- First to connect 4 pieces horizontally, vertically, or diagonally wins
- Draw if board fills with no winner

## Text Representation

```
. . . . . . .
. . . . . . .
. . O . . . .
. . X O . . .
. X O X . . .
X O X O X . .
```

- `.` = empty cell
- `X` = Agent A piece
- `O` = Agent B piece
- Columns numbered 0-6 (left to right)
- Row 0 is bottom, Row 5 is top

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│              React 19 + Tailwind + TanStack + Vite           │
├─────────────────────────────────────────────────────────────┤
│  AgentSelector  │  GameBoard (SSE)  │  MatchHistory          │
└────────┬────────┴────────┬──────────┴────────┬──────────────┘
         │                 │                   │
         │    HTTP/SSE     │                   │
         ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                         Backend                              │
│                  Rust + Tokio + Axum                         │
├─────────────────────────────────────────────────────────────┤
│  Routes  │  Game Engine  │  Agent Runner  │  Persistence     │
└────┬─────┴───────┬───────┴───────┬────────┴────────┬────────┘
     │             │               │                 │
     │             │               ▼                 ▼
     │             │    ┌──────────────────┐  ┌───────────┐
     │             │    │  CLI Subprocesses │  │  SQLite   │
     │             │    │  claude/gemini/   │  │           │
     │             │    │  copilot/codex    │  │           │
     │             │    └──────────────────┘  └───────────┘
     │             │
     └─────────────┘
```

---

## Backend (Rust)

### Project Structure

```
backend/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── routes/
│   │   ├── mod.rs
│   │   ├── game.rs
│   │   ├── agents.rs
│   │   └── history.rs
│   ├── game/
│   │   ├── mod.rs
│   │   ├── board.rs
│   │   ├── engine.rs
│   │   └── state.rs
│   ├── agents/
│   │   ├── mod.rs
│   │   ├── runner.rs
│   │   ├── claude.rs
│   │   ├── gemini.rs
│   │   ├── copilot.rs
│   │   └── codex.rs
│   ├── persistence/
│   │   ├── mod.rs
│   │   ├── db.rs
│   │   └── models.rs
│   └── sse/
│       ├── mod.rs
│       └── broadcaster.rs
```

### Dependencies (Cargo.toml)

```toml
[package]
name = "connect-four-backend"
version = "0.1.0"
edition = "2024"
rust-version = "1.93"

[dependencies]
tokio = { version = "1", features = ["full", "process"] }
axum = { version = "0.8", features = ["macros"] }
axum-extra = { version = "0.10", features = ["typed-header"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite"] }
tower-http = { version = "0.6", features = ["cors"] }
tokio-stream = "0.1"
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/agents` | List available agents |
| POST | `/api/game/start` | Start new game with selected agents |
| GET | `/api/game/:id/stream` | SSE stream for game events |
| GET | `/api/history` | Get match history |
| GET | `/api/history/:id` | Get specific match details |

### Game Flow

1. Frontend calls `POST /api/game/start` with `{ agent_a: "claude", agent_b: "gemini" }`
2. Backend creates game session, stores in SQLite, returns `game_id`
3. Frontend connects to `GET /api/game/:id/stream` (SSE)
4. Backend game loop:
   - Send current board state via SSE
   - Call Agent A CLI subprocess with board state prompt
   - Parse agent response for column number (0-6)
   - Update board, check win/draw
   - Send updated state via SSE
   - If game continues, call Agent B
   - Repeat until win/draw
5. Store final result in SQLite
6. Send game-over event via SSE

### Agent CLI Commands

```rust
pub enum AgentType {
    Claude,
    Gemini,
    Copilot,
    Codex,
}

impl AgentType {
    pub fn build_command(&self, prompt: &str) -> Command {
        match self {
            AgentType::Claude => {
                let mut cmd = Command::new("claude");
                cmd.args(["-p", prompt, "--model", "opus-4-5", "--dangerously-skip-permissions"]);
                cmd
            }
            AgentType::Gemini => {
                let mut cmd = Command::new("gemini");
                cmd.args(["-y", prompt]);
                cmd
            }
            AgentType::Copilot => {
                let mut cmd = Command::new("copilot");
                cmd.args(["--allow-all", "--model", "claude-sonnet-4", "-p", prompt]);
                cmd
            }
            AgentType::Codex => {
                let mut cmd = Command::new("codex");
                cmd.args(["exec", "--full-auto", "--model", "gpt-5.2", prompt]);
                cmd
            }
        }
    }
}
```

### Agent Prompt Template

```
You are playing Connect Four. You are player {X or O}.

Current board state:
. . . . . . .
. . . . . . .
. . O . . . .
. . X O . . .
. X O X . . .
X O X O X . .

Columns are numbered 0-6 from left to right.
Choose a column to drop your piece.
Respond with ONLY a single digit (0-6) representing your chosen column.
```

### Agent Response Parser

```rust
pub fn parse_agent_response(output: &str) -> Result<u8, ParseError> {
    output
        .chars()
        .find(|c| c.is_ascii_digit())
        .and_then(|c| c.to_digit(10))
        .filter(|&n| n <= 6)
        .map(|n| n as u8)
        .ok_or(ParseError::InvalidColumn)
}
```

### SSE Event Types

```rust
pub enum GameEvent {
    BoardUpdate {
        board: [[char; 7]; 6],
        current_player: String,
        last_move: Option<(u8, u8)>,
    },
    AgentThinking {
        agent: String,
    },
    AgentMoved {
        agent: String,
        column: u8,
    },
    GameOver {
        winner: Option<String>,
        winning_cells: Option<Vec<(u8, u8)>>,
        duration_ms: u64,
    },
    Error {
        message: String,
    },
}
```

### SQLite Schema

```sql
CREATE TABLE matches (
    id TEXT PRIMARY KEY,
    agent_a TEXT NOT NULL,
    agent_b TEXT NOT NULL,
    winner TEXT,
    is_draw INTEGER NOT NULL DEFAULT 0,
    moves TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER
);

CREATE INDEX idx_matches_started_at ON matches(started_at DESC);
```

### Board Module

```rust
pub struct Board {
    cells: [[Cell; 7]; 6],
}

pub enum Cell {
    Empty,
    X,
    O,
}

impl Board {
    pub fn new() -> Self;
    pub fn drop_piece(&mut self, column: u8, player: Cell) -> Result<u8, DropError>;
    pub fn check_winner(&self) -> Option<Cell>;
    pub fn is_full(&self) -> bool;
    pub fn to_text(&self) -> String;
    pub fn is_column_valid(&self, column: u8) -> bool;
}
```

---

## Frontend (React)

### Project Structure

```
frontend/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── AgentSelector.tsx
│   │   ├── GameBoard.tsx
│   │   ├── GameCell.tsx
│   │   ├── MatchHistory.tsx
│   │   └── MatchDetail.tsx
│   ├── hooks/
│   │   ├── useGameStream.ts
│   │   └── useMatchHistory.ts
│   ├── api/
│   │   └── client.ts
│   └── types/
│       └── index.ts
```

### Dependencies (package.json)

```json
{
  "name": "connect-four-frontend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "@tanstack/react-query": "^5.0.0",
    "@tanstack/react-router": "^1.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "autoprefixer": "^10.0.0",
    "postcss": "^8.0.0",
    "tailwindcss": "^3.0.0",
    "typescript": "^5.0.0",
    "vite": "^6.0.0"
  }
}
```

### UI Views

#### 1. Agent Selection View

```tsx
interface AgentSelectorProps {
  onStart: (agentA: string, agentB: string) => void;
}

function AgentSelector({ onStart }: AgentSelectorProps) {
  const [agentA, setAgentA] = useState<string>("");
  const [agentB, setAgentB] = useState<string>("");
  const { data: agents } = useQuery({ queryKey: ["agents"], queryFn: fetchAgents });

  return (
    <div className="flex flex-col gap-4 p-8">
      <h1 className="text-2xl font-bold">Connect Four: Agent vs Agent</h1>
      <select value={agentA} onChange={(e) => setAgentA(e.target.value)}>
        <option value="">Select Agent A</option>
        {agents?.map((a) => <option key={a} value={a}>{a}</option>)}
      </select>
      <select value={agentB} onChange={(e) => setAgentB(e.target.value)}>
        <option value="">Select Agent B</option>
        {agents?.map((a) => <option key={a} value={a}>{a}</option>)}
      </select>
      <button
        onClick={() => onStart(agentA, agentB)}
        disabled={!agentA || !agentB}
        className="bg-blue-600 text-white px-4 py-2 rounded"
      >
        Start Game
      </button>
    </div>
  );
}
```

#### 2. Game Board View (SSE)

```tsx
function GameBoard({ gameId }: { gameId: string }) {
  const { board, currentPlayer, status, winner } = useGameStream(gameId);

  return (
    <div className="flex flex-col items-center gap-4 p-8">
      <div className="text-xl">
        {status === "playing" && `${currentPlayer} is thinking...`}
        {status === "finished" && (winner ? `${winner} wins!` : "Draw!")}
      </div>
      <div className="grid grid-cols-7 gap-1 bg-blue-800 p-2 rounded">
        {board.map((row, r) =>
          row.map((cell, c) => (
            <GameCell key={`${r}-${c}`} value={cell} />
          ))
        )}
      </div>
    </div>
  );
}

function GameCell({ value }: { value: string }) {
  const bg = value === "X" ? "bg-red-500" : value === "O" ? "bg-yellow-500" : "bg-white";
  return <div className={`w-12 h-12 rounded-full ${bg}`} />;
}
```

#### 3. Match History View

```tsx
function MatchHistory() {
  const { data: matches } = useQuery({ queryKey: ["history"], queryFn: fetchHistory });

  return (
    <div className="p-8">
      <h2 className="text-xl font-bold mb-4">Match History</h2>
      <table className="w-full">
        <thead>
          <tr>
            <th>Agent A</th>
            <th>Agent B</th>
            <th>Winner</th>
            <th>Duration</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          {matches?.map((m) => (
            <tr key={m.id}>
              <td>{m.agent_a}</td>
              <td>{m.agent_b}</td>
              <td>{m.is_draw ? "Draw" : m.winner}</td>
              <td>{m.duration_ms}ms</td>
              <td>{m.started_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

### SSE Hook

```tsx
function useGameStream(gameId: string) {
  const [board, setBoard] = useState<string[][]>([]);
  const [currentPlayer, setCurrentPlayer] = useState<string>("");
  const [status, setStatus] = useState<"playing" | "finished">("playing");
  const [winner, setWinner] = useState<string | null>(null);

  useEffect(() => {
    const eventSource = new EventSource(`/api/game/${gameId}/stream`);

    eventSource.addEventListener("board_update", (e) => {
      const data = JSON.parse(e.data);
      setBoard(data.board);
      setCurrentPlayer(data.current_player);
    });

    eventSource.addEventListener("game_over", (e) => {
      const data = JSON.parse(e.data);
      setStatus("finished");
      setWinner(data.winner);
    });

    return () => eventSource.close();
  }, [gameId]);

  return { board, currentPlayer, status, winner };
}
```

---

## Scripts

### run.sh

```bash
#!/bin/bash

cd backend && cargo build --release &
BACKEND_PID=$!

cd frontend && bun install && bun run build &
FRONTEND_PID=$!

wait $BACKEND_PID
wait $FRONTEND_PID

cd backend && ./target/release/connect-four-backend &
BACKEND_SERVER=$!

cd frontend && bun run preview --port 3000 &
FRONTEND_SERVER=$!

echo "Backend running on http://localhost:8080"
echo "Frontend running on http://localhost:3000"

wait
```

### stop.sh

```bash
#!/bin/bash

pkill -f connect-four-backend
pkill -f "vite preview"
```

---

## Configuration

### Backend Config

```rust
pub struct Config {
    pub database_url: String,
    pub server_port: u16,
    pub agent_timeout_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database_url: "sqlite:connect_four.db".into(),
            server_port: 8080,
            agent_timeout_secs: 60,
        }
    }
}
```

### Frontend Config

```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";
```

---

## Error Handling

### Agent Errors

- Timeout: If agent does not respond within 60 seconds, forfeit the game
- Invalid response: If agent returns invalid column, retry once, then forfeit
- Process failure: If CLI command fails, forfeit the game

### Game Errors

- Column full: Reject move, ask agent again
- Invalid column number: Reject move, ask agent again

---

## File Summary

### Backend Files

| File | Purpose |
|------|---------|
| `main.rs` | Entry point, server setup |
| `lib.rs` | Re-exports all modules |
| `routes/mod.rs` | Route module exports |
| `routes/game.rs` | Game start and SSE endpoints |
| `routes/agents.rs` | List agents endpoint |
| `routes/history.rs` | Match history endpoints |
| `game/mod.rs` | Game module exports |
| `game/board.rs` | Board struct and logic |
| `game/engine.rs` | Game loop orchestration |
| `game/state.rs` | Game state management |
| `agents/mod.rs` | Agent module exports |
| `agents/runner.rs` | Subprocess execution |
| `agents/claude.rs` | Claude-specific config |
| `agents/gemini.rs` | Gemini-specific config |
| `agents/copilot.rs` | Copilot-specific config |
| `agents/codex.rs` | Codex-specific config |
| `persistence/mod.rs` | Persistence module exports |
| `persistence/db.rs` | SQLite connection pool |
| `persistence/models.rs` | Database models |
| `sse/mod.rs` | SSE module exports |
| `sse/broadcaster.rs` | SSE event broadcasting |

### Frontend Files

| File | Purpose |
|------|---------|
| `main.tsx` | React entry point |
| `App.tsx` | Main app with routing |
| `components/AgentSelector.tsx` | Agent selection UI |
| `components/GameBoard.tsx` | Game board display |
| `components/GameCell.tsx` | Individual cell component |
| `components/MatchHistory.tsx` | History list view |
| `components/MatchDetail.tsx` | Single match detail |
| `hooks/useGameStream.ts` | SSE subscription hook |
| `hooks/useMatchHistory.ts` | History data hook |
| `api/client.ts` | API client functions |
| `types/index.ts` | TypeScript types |

---

## Ports

| Service | Port |
|---------|------|
| Backend | 8080 |
| Frontend | 3000 |

---

## Sequence Diagram

```
Frontend              Backend                Agent A            Agent B
    │                    │                      │                   │
    │  POST /game/start  │                      │                   │
    │───────────────────>│                      │                   │
    │   { game_id }      │                      │                   │
    │<───────────────────│                      │                   │
    │                    │                      │                   │
    │  GET /game/:id/stream (SSE)               │                   │
    │───────────────────>│                      │                   │
    │                    │                      │                   │
    │  event: board_update                      │                   │
    │<───────────────────│                      │                   │
    │                    │                      │                   │
    │                    │  claude -p "..."     │                   │
    │                    │─────────────────────>│                   │
    │                    │      "3"             │                   │
    │                    │<─────────────────────│                   │
    │                    │                      │                   │
    │  event: agent_moved                       │                   │
    │<───────────────────│                      │                   │
    │                    │                      │                   │
    │  event: board_update                      │                   │
    │<───────────────────│                      │                   │
    │                    │                      │                   │
    │                    │  gemini -y "..."                         │
    │                    │─────────────────────────────────────────>│
    │                    │      "4"                                 │
    │                    │<─────────────────────────────────────────│
    │                    │                      │                   │
    │  event: agent_moved                       │                   │
    │<───────────────────│                      │                   │
    │                    │                      │                   │
    │  event: board_update                      │                   │
    │<───────────────────│                      │                   │
    │                    │                      │                   │
    │        ...         │                      │                   │
    │                    │                      │                   │
    │  event: game_over  │                      │                   │
    │<───────────────────│                      │                   │
    │                    │                      │                   │
```
