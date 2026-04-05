# Design Document: Pixel Office - Agent HQ

## Overview

Pixel Office is a web application that visualizes AI coding agents as pixel-art characters working in a virtual office. Users can spawn agents, assign tasks, and watch them work in real-time through an animated canvas interface.

The system combines the visual concept from [pixel-agents](https://github.com/pablodelucca/pixel-agents) (a VS Code extension) with the agent orchestration pattern from [agent-debate-club](https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-debate-club), creating a standalone web app.

## Goals

1. Provide a visual, gamified interface for managing AI agents
2. Support spawning multiple agent types (Claude, Gemini, Copilot, Codex)
3. Real-time feedback through animated pixel characters and SSE streaming
4. Persistent storage of agent tasks and outputs
5. Interactive: click agents for logs, double-click for chat

## System Design

### Frontend Architecture

```
React 19 App
├── App.tsx (layout + state management)
├── Canvas Layer
│   ├── PixelOfficeEngine (game loop: update -> render)
│   ├── Sprite Loader (PNG sprite sheets)
│   ├── A* Pathfinding (grid navigation)
│   └── Character Animator (frame cycling)
├── UI Layer
│   ├── AgentSpawnForm (name, type, task)
│   ├── AgentOutputPanel (SSE-driven log viewer)
│   ���── AgentChatPanel (interactive messaging)
│   └── Agent List (status overview)
└── Data Layer
    ├── TanStack Query (server state)
    └── useAgentSSE hook (EventSource)
```

**Canvas Engine Details:**
- 20x14 tile grid, each tile 16x16px scaled 3x (960x672 display)
- Floor tiles define walkable areas (warm wood for workspace, blue for lounge)
- Walls at grid edges, furniture placed at fixed positions
- 6 desk slots with PCs and chairs for agent seating
- Characters: 16x32px sprites, 7 frames per animation row, 3 directions
- A* pathfinding on walkable grid for character movement
- 60fps game loop via requestAnimationFrame

**Interaction Model:**
- Single click on character: opens output panel (SSE log viewer)
- Double click on character: opens chat panel
- Click detection via canvas coordinate mapping to tile grid

### Backend Architecture

```
Axum Web Server (port 3001)
├── Routes
│   ├── POST /api/agents/spawn     -> creates agent + starts async task
│   ├���─ GET  /api/agents/{id}/stream -> SSE event stream
│   ├── GET  /api/agents           -> list all agents
│   ├── GET  /api/agents/{id}      -> agent detail + messages
│   ├��─ DELETE /api/agents/{id}    -> stop agent
│   └── GET  /api/agent-types      -> available agent types
├── Agent Runner
│   ├── Subprocess spawner (tokio::process::Command)
│   ├── 120s timeout per execution
│   └── CLI builders (claude, gemini, copilot, codex)
├── SSE Broadcaster
│   ├── Per-agent broadcast channels (tokio::sync::broadcast)
│   ├── 100-message buffer per channel
│   └── Event types: agent_status, agent_message, agent_done, agent_error
└── Persistence
    ├── SQLite via sqlx
    ├── agents table (id, name, type, task, status, desk_index, timestamps)
    └── messages table (id, agent_id, content, role, timestamp)
```

**Agent Lifecycle:**
1. User submits spawn request (name, type, task)
2. Backend assigns desk index, creates DB record, creates SSE channel
3. Async tokio task starts:
   - Status: spawning -> thinking -> working
   - Runs CLI agent as subprocess with task as prompt
   - Captures stdout, saves to DB, broadcasts via SSE
   - Status: working -> done (or error)
4. Frontend receives SSE events, updates canvas and UI

### Data Flow

```
User Action          Frontend                  Backend                 CLI Agent
    │                    │                        │                       │
    ├─ Spawn Agent ─────>│                        │                       │
    │                    ├── POST /agents/spawn ──>│                       │
    │                    │                        ├── Insert DB            │
    │                    │                        ├── Create SSE channel   │
    │                    │<── { agent record } ────┤                       │
    │                    │                        ├── tokio::spawn         │
    │                    ├── EventSource connect ─>│                       │
    │                    │                        ├── Broadcast: thinking  │
    │                    │<── SSE: thinking ───────┤                       │
    │  (char walks       │                        ├── Broadcast: working   │
    │   to desk)         │<── SSE: working ───────┤                       │
    │                    │                        ├── Command::new(cli) ──>│
    │  (char types       │                        │                       ├── AI runs
    │   at desk)         │                        │                       │
    │                    │                        │<── stdout ─���───────────┤
    │                    │                        ├── Save message to DB   │
    │                    │<── SSE: agent_message ──┤                       │
    │  (output panel     │                        ├── Broadcast: done      │
    │   shows result)    │<── SSE: agent_done ────┤                       │
    │  (char goes idle)  │                        │                       │
```

### Office Layout

```
    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
 0  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W
 1  W  .  B  .  B  .  WB WB .  LP W  .  .  .  .  SP .  CK CT .  W
 2  W  .  PC .  PC .  .  .  PC .  W  .  .  .  .  .  .  .  .  W
 3  W  .  D1 .  D2 .  .  .  D3 .  W  .  SF SF SF .  .  .  .  W
 4  W  .  C1 .  C2 .  .  .  C3 .  W  .  .  .  .  .  .  .  .  W
 5  W  .  .  .  .  .  .  .  .  .  _  .  .  CT .  .  .  .  .  W
 6  W  PL .  .  .  .  .  .  .  .  _  .  .  .  .  .  .  .  .  W
 7  W  .  .  .  .  .  .  .  .  .  W  .  .  .  .  .  .  .  .  W
 8  W  .  C4 .  C5 .  .  .  C6 .  W  .  .  .  .  .  .  .  .  W
 9  W  .  D4 .  D5 .  .  .  D6 .  W  .  .  .  .  .  .  .  .  W
10  W  .  PC .  PC .  .  .  PC .  W  .  .  .  .  .  .  .  .  W
11  W  .  .  .  .  .  .  .  .  BN W  LP .  .  .  .  .  .  .  W
12  W  .  .  .  .  .  .  .  .  .  W  .  .  .  .  .  .  .  PL W
13  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W

Legend: W=Wall, D=Desk, C=Chair, PC=Computer, PL=Plant, LP=Large Plant,
        B=Bookshelf, WB=Whiteboard, SF=Sofa, CT=Coffee Table/Cactus,
        SP=Small Painting, CK=Clock, BN=Bin, _=Doorway
```

### Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Frontend framework | React 19 | Latest stable, concurrent features |
| Build tool | Vite 6 + Bun | Fast HMR, native TS support |
| Server state | TanStack Query | Automatic refetching, cache management |
| Rendering | HTML Canvas 2D | Pixel-perfect sprite rendering, no DOM overhead |
| Pathfinding | A* | Optimal path on grid, simple to implement |
| Backend runtime | Tokio | Async subprocess management, SSE broadcasting |
| Web framework | Axum 0.8 | Ergonomic, tower-based, SSE support |
| Database | SQLite (sqlx) | Zero config, embedded, sufficient for single-node |
| Realtime | SSE | Simpler than WebSockets, one-directional is enough |
| Agent execution | CLI subprocess | Reuses existing CLI tools, no API key management |

### Sprite System

Character sprites from [pixel-agents](https://github.com/pablodelucca/pixel-agents):
- 6 character skins (char_0.png through char_5.png)
- Each sprite sheet: 7 columns x 3 rows
- Frame size: 16x32 pixels
- Row 0: facing down, Row 1: facing up, Row 2: facing right
- Columns 0-3: walk cycle, Columns 4-5: typing animation, Column 6: idle

Furniture sprites:
- Desks, PCs (with on/off states), chairs, plants, bookshelves, whiteboard, sofa, etc.
- PC has 3 animated "on" frames for screen flicker effect

### Future Possibilities

- WebSocket support for bidirectional agent chat
- Multiple office rooms/floors
- Agent collaboration (two agents working on same task)
- Custom sprite upload for personalized characters
- Sound effects and background music
- Agent task queue (assign multiple tasks in sequence)
- Office customization (drag-and-drop furniture placement)
