# Terminator Mosquito Game - Design Doc

## Overview

A real-time simulation game where AI agents control a Terminator and Mosquitos on a grid battlefield. The Terminator hunts mosquitos and eggs while mosquitos breed, lay eggs, and multiply. The user picks which AI agent/model controls the Terminator and which controls the Mosquitos. The simulation runs in cycles with a highly visual grid UI, stats panel, timer, and sound effects.

## Tech Stack

| Layer    | Technology                                      |
|----------|------------------------------------------------|
| Frontend | Remix (React 19), TypeScript, Tailwind CSS 4   |
| Backend  | Java 25, Spring Boot 4.0.2, Spring Data JDBC   |
| Database | SQLite (via spring-data-jdbc, no Hibernate)     |
| Comms    | Server-Sent Events (SSE) for real-time updates  |
| Agents   | CLI-based AI agents (claude, gemini, copilot, codex) |
| Audio    | Web Audio API / HTML5 Audio for sound effects   |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Remix Frontend (:5173)                 │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │ Setup Tab│  │ Simulation Tab│  │ Stats Panel      │  │
│  │ (agents) │  │ (grid + game) │  │ (clock/counts)   │  │
│  └──────────┘  └───────────────┘  └──────────────────┘  │
│         │              ▲                                 │
│         │ POST         │ SSE                             │
└─────────┼──────────────┼────────────────────────────────┘
          │              │
          ▼              │
┌─────────────────────────────────────────────────────────┐
│             Spring Boot Backend (:8080)                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ REST API │  │ Game Engine  │  │ Agent Runner      │  │
│  │ Controllers│ │ (simulation) │  │ (CLI executor)    │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
│         │              │                    │            │
│         ▼              ▼                    ▼            │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ SQLite   │  │ SSE Emitter  │  │ claude/gemini/    │  │
│  │ (JDBC)   │  │ (broadcast)  │  │ copilot/codex CLI │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Game Rules

### Grid
- 20x20 grid (400 cells)
- Each cell can contain: nothing, Terminator, Mosquito(s), Egg(s)
- Terminator is visually distinct (red robot icon)
- Mosquitos are small flying icons
- Eggs are small oval icons

### Terminator (1 unit)
- Moves 1 block every 3rd cycle
- Can move: UP, DOWN, LEFT, RIGHT only (4 directions)
- Kills all mosquitos and eggs on the cell it enters
- AI agent decides direction each move cycle
- Cannot move outside grid boundaries

### Mosquitos (start with 6)
- Move 1 block per cycle in any direction (UP, DOWN, LEFT, RIGHT, and diagonals = 8 directions)
- AI agent decides direction for each mosquito each cycle
- When 2 mosquitos are on the same cell, they "date" and produce 1 egg on that cell
- Mosquitos die after 14 cycles (lifespan counter)
- Mosquitos cannot move outside grid boundaries

### Eggs
- Laid when 2 mosquitos occupy the same cell (dating)
- After 3 cycles on the grid, the egg hatches into a new mosquito
- Eggs do not move
- Terminator destroys eggs by stepping on them

### Cycle Timing
- 1 cycle = ~700ms sleep + LLM response time
- Both agent CLI calls run in parallel via a 2-thread ExecutorService
- Agent CLI process timeout: 10 seconds
- Future.get timeout: 12 seconds
- If LLM times out or returns invalid JSON, entities move randomly (logged)
- Eggs laid every time 2 mosquitos share a cell
- Game runs continuously until user stops it or win/lose condition is met

### Win/Lose Conditions
- **Terminator wins**: All mosquitos AND all eggs are destroyed
- **Mosquitos win**: Mosquito population reaches 50 (overwhelming the Terminator)
- **Draw**: Game runs for 200 cycles without either condition met

## Sound Effects

| Event              | Sound                                          |
|--------------------|-------------------------------------------------|
| Terminator kills   | "Hasta la vista, baby" (Terminator voice clip)  |
| Mosquitos date     | Romantic/kissing sound effect                   |
| Egg hatches        | Cracking/popping sound effect                   |
| Terminator moves   | Heavy footstep / mechanical step sound          |
| Game start         | "I'll be back" (Terminator voice clip)          |
| Terminator wins    | Victory fanfare + "Terminated" voice            |
| Mosquitos win      | Buzzing swarm crescendo                         |

Sounds are short MP3 files served from `/public/sounds/` and played via the Web Audio API. A mute toggle is available in the UI.

## AI Agent Integration

### Available Agents

| Agent   | Models                                              |
|---------|-----------------------------------------------------|
| claude  | opus, sonnet, haiku                                 |
| gemini  | gemini-3.1-pro, gemini-3-flash, gemini-2.5-pro     |
| copilot | claude-sonnet-4.6, claude-sonnet-4.5, gemini-3-pro  |
| codex   | gpt-5.4, gpt-5.4-mini, gpt-5.3-codex              |

### Agent Decision Making
Each cycle, the backend calls the agent CLI with a prompt describing:
- Current grid state (positions of terminator, all mosquitos, all eggs)
- Entity being controlled (terminator or mosquito group)
- Valid moves

The terminator agent returns:
```json
{"direction": "UP|DOWN|LEFT|RIGHT"}
```

The mosquito agent returns:
```json
{"moves": [{"id": "m_1", "direction": "UP"}, {"id": "m_2", "direction": "DOWN_LEFT"}]}
```

Fallback: if agent times out (10s process / 12s future) or returns invalid JSON, entities move randomly. All agent calls and fallbacks are logged with timing.

### Agent CLI Commands
- **claude**: `claude -p "<prompt>" --model <model> --dangerously-skip-permissions`
- **gemini**: `gemini -y -p "<prompt>"`
- **copilot**: `copilot --allow-all --model <model> -p "<prompt>"`
- **codex**: `codex exec --full-auto -m <model> "<prompt>"`

## Frontend UI

### Tab 1: Setup
- Title: "Terminator Mosquito Game"
- Two sections side by side:
  - **Left**: "Choose Terminator Agent" - select agent + model from dropdown
  - **Right**: "Choose Mosquito Agent" - select agent + model from dropdown
- Grid size selector (default 20x20)
- "Start Simulation" button (disabled until both agents selected)
- Dark theme with red/green accent colors

### Tab 2: Simulation
Layout: Game grid on the LEFT (70% width), Stats panel on the RIGHT (30% width)

**Game Grid (Left)**:
- 20x20 visual grid with cell borders
- Terminator: red robot emoji/icon with glow effect
- Mosquitos: small mosquito emoji with age indicator (fades as they age)
- Eggs: small egg emoji with timer indicator
- Kill animations: explosion effect when terminator kills
- Dating animation: heart effect when mosquitos date
- Hatch animation: crack effect when egg hatches
- Smooth CSS transitions for movement

**Stats Panel (Right)**:
- Clock: elapsed time counting up (MM:SS format)
- Cycle counter
- Terminator agent name + model
- Mosquito agent name + model
- Live counts:
  - Terminator: 1
  - Mosquitos alive: N
  - Eggs on grid: N
  - Total kills: N
  - Total hatched: N
  - Total dates: N
- Event log: scrolling list of recent events
- Mute/unmute sound toggle
- "Stop Simulation" button

### Tab 3: History
- Table of past simulations
- Columns: Date, Terminator Agent, Mosquito Agent, Winner, Duration, Max Mosquitos, Total Kills

## Backend API Endpoints

| Method | Path                        | Description                    |
|--------|-----------------------------|--------------------------------|
| GET    | /api/agents                 | List available agents + models |
| POST   | /api/games                  | Create and start a new game    |
| GET    | /api/games                  | List all games (history)       |
| GET    | /api/games/{id}             | Get game details               |
| GET    | /api/games/{id}/stream      | SSE stream for real-time updates |
| POST   | /api/games/{id}/stop        | Stop a running game            |

## SSE Events

| Event             | Data                                                       |
|-------------------|------------------------------------------------------------|
| game_start        | {grid_size, terminator_pos, mosquito_positions, agent_info} |
| cycle_update      | {cycle, terminator, mosquitos[], eggs[], kills[], dates[], hatches[], deaths[], stats} |
| game_over         | {winner, cycles, total_kills, total_hatched, total_dates, max_mosquitos, alive_mosquitos, active_eggs} |

## Database Schema (SQLite + Spring Data JDBC)

```sql
CREATE TABLE games (
    id TEXT PRIMARY KEY,
    terminator_agent TEXT NOT NULL,
    terminator_model TEXT NOT NULL,
    mosquito_agent TEXT NOT NULL,
    mosquito_model TEXT NOT NULL,
    grid_size INTEGER NOT NULL DEFAULT 20,
    winner TEXT,
    total_cycles INTEGER DEFAULT 0,
    max_mosquitos INTEGER DEFAULT 0,
    total_kills INTEGER DEFAULT 0,
    total_hatched INTEGER DEFAULT 0,
    total_dates INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE game_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(id)
);
```

## Project Structure

```
agent-terminator-game/
├── backend/
│   ├── src/main/java/com/game/terminator/
│   │   ├── TerminatorGameApplication.java
│   │   ├── config/
│   │   │   ├── CorsConfig.java
│   │   │   ├── SqliteConfig.java
│   │   │   └── SqliteDialect.java
│   │   ├── controller/
│   │   │   ├── AgentController.java
│   │   │   └── GameController.java
│   │   ├── engine/
│   │   │   ├── GameEngine.java
│   │   │   ├── Direction.java
│   │   │   ├── Position.java
│   │   │   ├── Mosquito.java
│   │   │   └── Egg.java
│   │   ├── agent/
│   │   │   ├── AgentRegistry.java
│   │   │   └── AgentRunner.java
│   │   ├── model/
│   │   │   ├── Game.java
│   │   │   └── GameEvent.java
│   │   ├── repository/
│   │   │   ├── GameRepository.java
│   │   │   └── GameEventRepository.java
│   │   └── sse/
│   │       └── SseBroadcaster.java
│   ├── src/main/resources/
│   │   ├── application.properties
│   │   └── schema.sql
│   └── pom.xml
├── frontend/
│   ├── app/
│   │   ├── root.tsx
│   │   ├── app.css
│   │   ├── routes/
│   │   │   └── _index.tsx
│   │   ├── components/
│   │   │   ├── SetupPanel.tsx
│   │   │   ├── GameGrid.tsx
│   │   │   ├── StatsPanel.tsx
│   │   │   └── HistoryTable.tsx
│   │   ├── hooks/
│   │   │   ├── useGameSSE.ts
│   │   │   └── useSound.ts
│   │   ├── api/
│   │   │   └── game.ts
│   │   └── types/
│   │       └── index.ts
│   ├── public/
│   │   ├── favicon.ico
│   │   └── sounds/
│   │       ├── hasta-la-vista.mp3
│   │       ├── ill-be-back.mp3
│   │       ├── terminated.mp3
│   │       ├── mosquito-date.mp3
│   │       ├── egg-hatch.mp3
│   │       ├── footstep.mp3
│   │       ├── victory.mp3
│   │       └── swarm.mp3
│   ├── package.json
│   ├── env.d.ts
│   ├── vite.config.ts
│   └── tsconfig.json
├── run.sh
├── stop.sh
├── test.sh
└── design-doc.md
```

## run.sh / stop.sh

**run.sh**: builds the Java backend with Maven, starts it on port 8080, installs frontend deps with bun, starts Remix dev server on port 5173, saves PIDs to `/tmp/terminator-game-*.pid`.

**stop.sh**: kills both processes by PID, falls back to pkill and lsof port kill, cleans up PID files.

## Game Engine Flow

```
1. User selects agents on Setup tab -> POST /api/games
2. Backend creates game record in SQLite
3. Backend starts GameEngine in a new thread
4. GameEngine loop (every ~700ms + LLM time):
   a. Increment cycle counter
   b. Build grid state prompts for both agents
   c. Submit both agent CLI calls in parallel (2-thread pool)
   d. Wait for Terminator response (every 3rd cycle only, 12s timeout)
   e. Wait for Mosquito response (every cycle, 12s timeout)
   f. On timeout/error: fall back to random moves (logged)
   g. Apply moves (with boundary checks)
   h. Check terminator cell -> kill mosquitos/eggs on that cell
   i. Check mosquito collisions -> if 2+ mosquitos on same cell, create egg
   j. Check egg timers -> hatch eggs older than 3 cycles
   k. Check mosquito age -> kill mosquitos older than 14 cycles
   l. Broadcast cycle_update via SSE
   m. Check win/lose conditions
5. On game over -> broadcast game_over, update DB, shutdown executor
```

## Visual Design

- Dark background (#0a0a0a) with grid lines (#1a1a1a)
- Terminator: Red glow (#ef4444), robot/skull icon
- Mosquitos: Green (#22c55e), small flying bug icon with wing animation
- Eggs: Yellow (#eab308), small oval with pulse animation
- Kill effect: Red explosion burst (CSS animation)
- Date effect: Pink hearts floating up (CSS animation)
- Hatch effect: Yellow crack and pop (CSS animation)
- Stats panel: Dark card with neon-styled counters
- Font: monospace for counters, sans-serif for labels
