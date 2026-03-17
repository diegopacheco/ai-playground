# Dungeon Master AI

Text-based RPG where an LLM acts as the Dungeon Master, generating the world and responding to player actions in real-time.

## Architecture

```
Frontend (React 19 + Tailwind)        Backend (Rust + Axum)
┌─────────────────────────┐           ┌──────────────────────────┐
│  GameSetup              │           │  POST /api/games          │
│  GamePlay               │──HTTP──>  │  POST /api/games/:id/action│
│  GameHistory            │           │  GET  /api/games          │
│  CharacterPanel         │<──SSE───  │  GET  /api/games/:id      │
│                         │           │  GET  /api/games/:id/stream│
└─────────────────────────┘           └──────────┬───────────────┘
                                                 │
                                      ┌──────────┴───────────────┐
                                      │  Claude CLI (sonnet)      │
                                      │  SQLite (persistence)     │
                                      └──────────────────────────┘
```

## Stack

| Layer    | Technology                          |
|----------|-------------------------------------|
| Frontend | React 19, TypeScript, Tailwind CSS, Vite |
| Backend  | Rust, Axum 0.8, Tokio, SQLx, SQLite |
| LLM      | Claude CLI (sonnet model)           |
| Streaming| Server-Sent Events (SSE)            |

## Features

- Real-time DM narration streamed via SSE
- Character stats tracking (HP, XP, Gold, Level, Inventory, Location)
- Multiple settings: Medieval Fantasy, Sci-Fi, Cyberpunk, Pirate, and more
- Game history with resume capability
- LLM-driven world generation and action responses

## Prerequisites

- Rust 1.75+
- Bun
- Claude CLI installed and configured (`claude` command available)

## How to Run

```bash
./run.sh
```

- Backend: http://localhost:3000
- Frontend: http://localhost:5173

## How to Stop

```bash
./stop.sh
```

## How it Works

1. Player creates a game by choosing a character name and setting
2. The backend calls Claude CLI to generate an opening scene
3. Player types actions (e.g., "I open the chest", "I attack the goblin")
4. Claude responds as the DM with narrative and updated character stats
5. Everything streams in real-time through SSE to the React frontend
6. Game state persists in SQLite so adventures can be resumed

## Project Structure

```
ai-rpg/
├── backend/
│   └── src/
│       ├── main.rs              # Axum server (port 3000)
│       ├── routes/game_routes.rs # API endpoints
│       ├── game/engine.rs       # DM game logic + prompt engineering
│       ├── agents/claude.rs     # Claude CLI wrapper
│       ├── persistence/db.rs    # SQLite schema + queries
│       └── sse/broadcaster.rs   # SSE event broadcasting
├── frontend/
│   └── src/
│       ├── App.tsx              # Main app with routing
│       ├── components/
│       │   ├── GameSetup.tsx    # Character creation screen
│       │   ├── GamePlay.tsx     # Main game screen
│       │   ├── CharacterPanel.tsx # Stats sidebar
│       │   └── GameHistory.tsx  # Past adventures list
│       ├── hooks/useGameSSE.ts  # SSE event listener
│       └── api/games.ts        # API client
├── run.sh
└── stop.sh
```
