# Design Document: Multi-Agent System - Pixel Office Control Panel

## Overview

A web application that visualizes AI coding agents as pixel-art characters working in a virtual office. Users can spawn agents, assign tasks, chat interactively, and watch them work in real-time through an animated canvas interface.

The system combines the visual concept from [pixel-agents](https://github.com/pablodelucca/pixel-agents) (a VS Code extension) with the agent orchestration pattern from [agent-debate-club](https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-debate-club), creating a standalone web app.

## Goals

1. Provide a visual, gamified interface for managing AI agents
2. Support spawning multiple agent types (Claude, Gemini, Copilot, Codex)
3. Real-time feedback through animated pixel characters and SSE streaming
4. Interactive chat with agents, with conversation history persistence
5. Single click for logs, double click for chat

## System Design

### Frontend Architecture

```
React 19 App
+-- App.tsx (layout, state, ref to canvas engine)
+-- Canvas Layer
|   +-- PixelOfficeEngine (game loop: update -> render)
|   +-- Programmatic floor rendering (wood, carpet, rug, walls)
|   +-- Sprite Loader (PNG character sprite sheets)
|   +-- A* Pathfinding (grid navigation)
|   +-- Character Animator (frame cycling, typing, walking, sitting)
|   +-- Speech Bubbles + Status Indicators
+-- UI Layer
|   +-- AgentSpawnForm (name, type, task)
|   +-- AgentOutputPanel (SSE for active, REST for completed)
|   +-- AgentChatPanel (interactive messaging with thinking state)
|   +-- Agent List (status overview, clear all)
+-- Data Layer
    +-- TanStack Query (server state, 3s polling)
    +-- useAgentSSE hook (EventSource for active agents)
```

**Canvas Engine Details:**
- 20x14 tile grid, each tile 16x16px scaled 3x (960x672 display)
- Programmatic floor rendering: warm hardwood planks (left room), blue-gray carpet (right room), dark red rug (accent areas), purple-indigo walls with molding
- 6 desk slots with PCs and chairs for agent seating
- Each agent assigned a unique desk via DB-level tracking
- Characters: 16x32px sprites, 7 frames per animation row, 3 directions (down, up, right)
- A* pathfinding on walkable grid for character movement from entrance to desk
- Completed agents snap directly to desk (no walk animation needed)
- 60fps game loop via requestAnimationFrame
- Engine created once (stable useEffect), click handler via ref to avoid re-creation

**Interaction Model:**
- Single click on character or agent list row: opens output panel
- Double click on character or agent list row: opens chat panel
- Chat panel shows "Thinking..." and triggers typing animation on canvas
- Click detection via canvas coordinate mapping to tile grid

### Backend Architecture

```
Axum Web Server (port 3001)
+-- Routes
|   +-- POST   /api/agents/spawn       -> creates agent + starts async task
|   +-- GET    /api/agents/{id}/stream  -> SSE event stream
|   +-- POST   /api/agents/{id}/chat    -> interactive chat (builds prompt with history)
|   +-- GET    /api/agents              -> list all agents
|   +-- GET    /api/agents/{id}         -> agent detail + messages
|   +-- DELETE /api/agents/{id}         -> stop agent
|   +-- DELETE /api/agents/clear        -> clear all agents + messages
|   +-- GET    /api/agent-types         -> available agent types
+-- Agent Runner
|   +-- Subprocess spawner (tokio::process::Command)
|   +-- 120s timeout per execution
|   +-- CLI builders:
|       +-- claude: claude -p <prompt> --model opus --dangerously-skip-permissions
|       +-- gemini: gemini -y -p <prompt>
|       +-- copilot: copilot -p <prompt>
|       +-- codex: codex exec -c model="gpt-5.4" <prompt>
+-- SSE Broadcaster
|   +-- Per-agent broadcast channels (tokio::sync::broadcast)
|   +-- 100-message buffer per channel
|   +-- Event types: agent_status, agent_message, agent_done, agent_error
+-- Persistence
    +-- SQLite via sqlx
    +-- agents table (id, name, type, task, status, desk_index, timestamps)
    +-- messages table (id, agent_id, content, role, timestamp)
    +-- Unique desk assignment: queries all used desks, picks first free 0-5
```

**Agent Lifecycle:**
1. User submits spawn request (name, type, task)
2. Backend assigns unique desk index, creates DB record, creates SSE channel
3. Async tokio task starts:
   - Status: spawning -> thinking -> working
   - Runs CLI agent as subprocess with task as prompt
   - Captures stdout, saves to DB, broadcasts via SSE
   - Status: working -> done (or error)
4. Frontend receives SSE events, updates canvas and UI
5. Completed agents: output loaded from REST API, character shown at desk

**Chat Lifecycle:**
1. User double-clicks agent, opens chat panel
2. Chat panel loads conversation history from REST API
3. User sends message -> POST /api/agents/{id}/chat
4. Backend builds prompt with full conversation history
5. Runs agent CLI with conversation context
6. Returns response, saves both user + assistant messages to DB
7. Frontend shows "Thinking..." bubble, agent types at desk during processing

### Data Flow

```
User Action          Frontend                  Backend                 CLI Agent
    |                    |                        |                       |
    +-- Spawn Agent ---->|                        |                       |
    |                    +-- POST /agents/spawn ->|                       |
    |                    |                        +-- Insert DB            |
    |                    |                        +-- Create SSE channel   |
    |                    |<-- { agent record } ---|                       |
    |                    |                        +-- tokio::spawn         |
    |                    +-- EventSource connect >|                       |
    |                    |                        +-- Broadcast: thinking  |
    |  (char walks       |<-- SSE: thinking ------|                       |
    |   to desk)         |                        +-- Broadcast: working   |
    |                    |<-- SSE: working -------|                       |
    |  (char types       |                        +-- Command::new(cli) ->|
    |   at desk)         |                        |                       +-- AI runs
    |                    |                        |<-- stdout ------------|
    |                    |                        +-- Save message to DB   |
    |                    |<-- SSE: agent_message --|                       |
    |  (output panel     |                        +-- Broadcast: done      |
    |   shows result)    |<-- SSE: agent_done ----|                       |
    |  (char sits idle)  |                        |                       |
    |                    |                        |                       |
    +-- Chat Message --->|                        |                       |
    |  (char types)      +-- POST /chat --------->|                       |
    |                    |                        +-- Build prompt w/hist  |
    |                    |                        +-- Command::new(cli) ->|
    |                    |                        |                       +-- AI runs
    |                    |                        |<-- stdout ------------|
    |                    |<-- { response } -------|                       |
    |  (char stops)      |                        |                       |
```

### Office Layout

```
    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
 0  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W
 1  W  .  PC .  .  PC .  .  PC .  W  .  .  .  .  .  .  .  CT W
 2  W  .  D1 .  .  D2 .  .  D3 .  W  .  BK .  .  SP .  CK .  W
 3  W  .  C1 .  .  C2 .  .  C3 .  W  .  .  .  .  .  .  .  .  W
 4  W  PL .  .  .  .  .  .  .  PL W  .  .  RG RG RG RG .  .  W
 5  W  .  .  .  RG RG RG .  .  .  W  .  .  SF SF SF .  .  .  W
 6  W  LP .  .  RG RG RG .  .  .  _  .  .  CT .  .  .  .  .  W
 7  W  .  .  .  RG RG RG .  BN .  _  .  .  .  .  .  .  .  .  W
 8  W  .  .  .  .  .  .  .  .  .  W  .  .  .  .  .  .  .  .  W
 9  W  .  C4 .  .  C5 .  .  C6 .  W  .  .  .  .  .  .  .  .  W
10  W  .  D4 .  .  D5 .  .  D6 .  W  LP .  .  .  .  .  .  .  W
11  W  .  PC .  .  PC .  .  PC .  W  .  .  .  .  .  .  .  PL W
12  W  .  .  .  .  .  .  .  .  .  W  .  .  .  .  .  .  .  CT W
13  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W  W

Legend: W=Wall, D=Desk, C=Chair, PC=Computer, PL=Plant, LP=Large Plant,
        BK=Bookshelf, WB=Whiteboard, SF=Sofa, CT=Coffee Table/Cactus,
        SP=Small Painting, CK=Clock, BN=Bin, RG=Rug, _=Doorway

Floor tiles: Left room = warm hardwood, Right room = blue-gray carpet,
             RG = dark red accent rug
```

### Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Frontend framework | React 19 | Latest stable, concurrent features |
| Build tool | Vite 6 + Bun | Fast HMR, native TS support |
| Server state | TanStack Query | Automatic refetching, cache management |
| Rendering | HTML Canvas 2D | Pixel-perfect sprite rendering, no DOM overhead |
| Floor rendering | Programmatic canvas | Sprite tiles looked bad when scaled; programmatic gives full control |
| Pathfinding | A* | Optimal path on grid, simple to implement |
| Backend runtime | Tokio | Async subprocess management, SSE broadcasting |
| Web framework | Axum 0.8 | Ergonomic, tower-based, SSE support |
| Database | SQLite (sqlx) | Zero config, embedded, sufficient for single-node |
| Realtime | SSE | Simpler than WebSockets, one-directional is enough |
| Agent execution | CLI subprocess | Reuses existing CLI tools, no API key management |

### Sprite System

Character sprites from [pixel-agents](https://github.com/pablodelucca/pixel-agents):
- 6 character skins (char_0.png through char_5.png)
- Each sprite sheet: 7 columns x 3 rows (112x96 px)
- Frame size: 16x32 pixels
- Row 0: facing down, Row 1: facing up, Row 2: facing right
- Columns 0-3: walk cycle, Columns 4-5: typing animation, Column 6: idle

Furniture sprites:
- Desks, PCs (with on/off states), chairs, plants, bookshelves, whiteboard, sofa, etc.
- PC has 3 animated "on" frames for screen flicker effect

### Key Implementation Details

**Engine Stability**: The canvas engine is created once with an empty useEffect dependency array. The click handler uses a ref pattern to always have the latest agents list without causing engine re-creation.

**Desk Assignment**: Each agent gets a unique desk. The backend queries all existing desk_index values and picks the first free slot (0-5).

**Completed Agent Display**: Agents already in done/error/stopped status skip the walk animation and snap directly to their assigned desk chair.

**Chat Animation**: When a user sends a chat message, the canvas engine is notified via a ref (useImperativeHandle) to show typing animation on the agent's character.

**Output Panel**: Active agents stream output via SSE. Completed agents load stored messages from the REST API.

### Future Possibilities

- WebSocket support for bidirectional real-time agent chat
- Multiple office rooms/floors
- Agent collaboration (two agents working on same task)
- Custom sprite upload for personalized characters
- Sound effects and background music
- Agent task queue (assign multiple tasks in sequence)
- Office customization (drag-and-drop furniture placement)
