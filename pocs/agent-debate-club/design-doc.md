# Agent Debate Club - Design Document

## Overview

Agent Debate Club is a real-time debate platform where two AI agents (A and B) argue on a user-defined topic while a third agent (C) acts as judge. The system uses SSE for live updates and SQLite for persistence.

## Technology Stack

### Backend
- Rust 2024 edition (1.93)
- Axum (web framework)
- Tokio (async runtime)
- SQLite (persistence via rusqlite)
- SSE (Server-Sent Events for real-time updates)

### Frontend
- React 19
- Bun (runtime/package manager)
- Vite (build tool)
- TanStack Query (data fetching)
- Tailwind CSS (styling)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React 19)                      │
├─────────────────────────────────────────────────────────────────┤
│  ThemeSetup  │  DebateView (SSE)  │  HistoryView                │
└──────────────┴───────────────────┴──────────────────────────────┘
                              │
                              │ HTTP + SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (Rust/Axum)                         │
├─────────────────────────────────────────────────────────────────┤
│  Routes  │  DebateEngine  │  Broadcaster  │  Database           │
└──────────┴────────────────┴───────────────┴─────────────────────┘
                              │
                              │ CLI Subprocess
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Runners                               │
├─────────────────────────────────────────────────────────────────┤
│  Claude  │  Gemini  │  Copilot  │  Codex                        │
└──────────┴──────────┴───────────┴───────────────────────────────┘
```

## UI Screens

### 1. Theme Setup Screen

**Purpose**: Configure debate parameters

**Components**:
- Text input: Debate theme/topic
- Dropdown A: Select Agent A (claude, gemini, copilot, codex)
- Dropdown B: Select Agent B (claude, gemini, copilot, codex)
- Agent C (Judge): Fixed as claude opus (most capable for judging)
- Number input: Debate duration (default: 60 seconds)
- Start button: Initiates debate

**Validation**:
- Theme must not be empty
- Agent A and B can be same or different
- Duration: 30-300 seconds range

### 2. Debate Action Screen

**Purpose**: Live debate visualization

**Components**:
- Topic header: Shows current debate theme
- Timer: Countdown showing remaining time
- Chat widget: Scrollable message list
  - Each message shows:
    - Agent name (A or B)
    - Message content
    - Label: "ATTACK" or "DEFENSE" (determined by agent response)
    - Timestamp
- Status indicator: Shows which agent is thinking

**Flow**:
1. Agent A speaks first (prompted to attack the topic or defend a position)
2. Agent B responds (can attack Agent A's argument or defend)
3. Alternates until timer expires
4. Chat history sent to Judge (Agent C)
5. Judge announces winner with reasoning

**SSE Events**:
- `AgentThinking { agent: String }`
- `AgentMessage { agent: String, content: String, stance: String }`
- `DebateOver { winner: String, reason: String, duration_ms: u64 }`
- `Error { message: String }`

### 3. History Screen

**Purpose**: View past debates

**Components**:
- List of debate cards showing:
  - Topic
  - Agent A vs Agent B
  - Winner
  - Date/time
  - Duration
- Click to expand: Full debate transcript and judge reasoning

## Database Schema

```sql
CREATE TABLE debates (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    agent_a TEXT NOT NULL,
    agent_b TEXT NOT NULL,
    agent_judge TEXT NOT NULL,
    winner TEXT,
    judge_reason TEXT,
    duration_seconds INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    content TEXT NOT NULL,
    stance TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (debate_id) REFERENCES debates(id)
);
```

## Backend Structure

```
backend/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── routes/
│   │   ├── mod.rs
│   │   ├── debate.rs
│   │   └── history.rs
│   ├── agents/
│   │   ├── mod.rs
│   │   ├── runner.rs
│   │   ├── claude.rs
│   │   ├── gemini.rs
│   │   ├── copilot.rs
│   │   └── codex.rs
│   ├── debate/
│   │   ├── mod.rs
│   │   ├── engine.rs
│   │   └── state.rs
│   ├── sse/
│   │   ├── mod.rs
│   │   └── broadcaster.rs
│   └── db/
│       ├── mod.rs
│       └── repository.rs
```

## Frontend Structure

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
│   │   ├── ThemeSetup.tsx
│   │   ├── DebateView.tsx
│   │   ├── ChatWidget.tsx
│   │   ├── Timer.tsx
│   │   ├── HistoryList.tsx
│   │   └── DebateCard.tsx
│   ├── hooks/
│   │   ├── useDebateSSE.ts
│   │   └── useDebates.ts
│   ├── api/
│   │   └── debates.ts
│   └── types/
│       └── index.ts
```

## API Endpoints

### POST /api/debates
Create and start a new debate.

**Request**:
```json
{
  "topic": "Is AI beneficial for humanity?",
  "agent_a": "claude",
  "agent_b": "gemini",
  "duration_seconds": 60
}
```

**Response**:
```json
{
  "id": "uuid-here",
  "topic": "Is AI beneficial for humanity?",
  "agent_a": "claude",
  "agent_b": "gemini",
  "agent_judge": "claude",
  "duration_seconds": 60,
  "started_at": "2024-01-15T10:30:00Z"
}
```

### GET /api/debates/:id/stream
SSE endpoint for real-time debate updates.

**Events**:
```
event: agent_thinking
data: {"agent": "A"}

event: agent_message
data: {"agent": "A", "content": "...", "stance": "ATTACK"}

event: debate_over
data: {"winner": "A", "reason": "...", "duration_ms": 60000}
```

### GET /api/debates
List all debates (history).

**Response**:
```json
[
  {
    "id": "uuid",
    "topic": "...",
    "agent_a": "claude",
    "agent_b": "gemini",
    "winner": "A",
    "judge_reason": "...",
    "duration_seconds": 60,
    "started_at": "...",
    "ended_at": "..."
  }
]
```

### GET /api/debates/:id
Get single debate with messages.

**Response**:
```json
{
  "id": "uuid",
  "topic": "...",
  "agent_a": "claude",
  "agent_b": "gemini",
  "winner": "A",
  "judge_reason": "...",
  "messages": [
    {
      "agent": "A",
      "content": "...",
      "stance": "ATTACK",
      "created_at": "..."
    }
  ]
}
```

## Agent Prompts

### Debater Prompt (Agent A/B)
```
You are Agent {A/B} in a debate about: "{topic}"

Previous messages:
{chat_history}

You must respond with your argument. Start your response with either [ATTACK] or [DEFENSE] to indicate your stance.
- ATTACK: You are challenging the opposing view or making an aggressive argument
- DEFENSE: You are defending your position or responding to criticism

Keep your response concise (2-3 sentences). Be persuasive and logical.
```

### Judge Prompt (Agent C)
```
You are the judge of a debate about: "{topic}"

Agent A ({agent_a_type}):
Agent B ({agent_b_type}):

Full debate transcript:
{full_transcript}

Analyze the debate and declare a winner. Consider:
1. Strength of arguments
2. Use of logic and evidence
3. Effective rebuttals
4. Overall persuasiveness

Respond in this format:
WINNER: A or B
REASON: Your explanation (2-3 sentences)
```

## Debate Engine Flow

1. **Start**: Create debate record in SQLite
2. **Loop** (while time remaining):
   - Broadcast `AgentThinking` for current agent
   - Build prompt with chat history
   - Execute agent CLI subprocess
   - Parse response for stance (ATTACK/DEFENSE)
   - Save message to SQLite
   - Broadcast `AgentMessage`
   - Switch to other agent
3. **Judge**:
   - Build judge prompt with full transcript
   - Execute judge agent
   - Parse winner and reason
   - Update debate record
   - Broadcast `DebateOver`

## Configuration

### Backend (environment)
```
DATABASE_URL=debates.db
HOST=0.0.0.0
PORT=3000
AGENT_TIMEOUT_SECS=30
```

### Frontend (.env)
```
VITE_API_URL=http://localhost:3000
```

## Error Handling

- Agent timeout: Skip turn, note in transcript
- Agent CLI error: Retry once, then skip turn
- Invalid stance format: Default to ATTACK
- SSE disconnect: Frontend auto-reconnects
- Database error: Return 500 with error message

## Security Considerations

- Input sanitization for debate topics
- Rate limiting on debate creation
- CORS configuration for frontend origin
- No user authentication (single-user local app)
