# Tetris Game - Design Document

## Overview

A full-stack Tetris game with a React 19 frontend and Rust/Actix backend. The game supports 10 difficulty levels, multiple visual themes, configurable timers, and a persistent leaderboard.

## Architecture

```
┌─────────────────────┐     HTTP/JSON      ┌──────────────────────┐
│   React Frontend    │ ◄────────────────► │   Rust Backend       │
│   (Vite + Bun)      │                    │   (Actix + Tokio)    │
│                     │                    │                      │
│ - Game Engine       │                    │ - Score Storage      │
│ - Canvas Renderer   │                    │ - Config Management  │
│ - TanStack Router   │                    │ - CORS Support       │
│ - Config UI         │                    │                      │
└─────────────────────┘                    └──────────────────────┘
```

## Frontend Stack

- **React 19** with TypeScript
- **Vite 8** for bundling and dev server
- **Bun** as the JavaScript runtime and package manager
- **TanStack Router** for client-side routing
- **HTML5 Canvas** for game rendering

## Backend Stack

- **Rust 1.94+** (edition 2024)
- **Actix-web 4** as the HTTP framework
- **Tokio** as the async runtime
- **Serde** for JSON serialization
- **In-memory storage** with Mutex-protected Vec

## Game Mechanics

### Levels (1-10)

Each level increases the drop speed of pieces. Players advance one level for every 10 lines cleared.

| Level | Drop Speed (ms) |
|-------|-----------------|
| 1     | 800             |
| 2     | 720             |
| 3     | 630             |
| 4     | 540             |
| 5     | 450             |
| 6     | 370             |
| 7     | 290             |
| 8     | 220             |
| 9     | 160             |
| 10    | 100             |

### Scoring

- 1 line:  100 x (level + 1)
- 2 lines: 300 x (level + 1)
- 3 lines: 500 x (level + 1)
- 4 lines: 800 x (level + 1)

Difficulty multipliers: Easy (0.7x), Medium (1.0x), Hard (1.5x)

### Pieces

Standard 7 Tetrominos: I, O, T, L, J, S, Z

### Controls

- Arrow Left/Right: Move piece horizontally
- Arrow Down: Soft drop
- Arrow Up: Rotate piece
- Space: Hard drop (instant placement)

## Configuration Options

### Difficulty
- **Easy**: Slower speed, 0.7x score multiplier
- **Medium**: Standard speed, 1.0x score multiplier
- **Hard**: Faster speed, 1.5x score multiplier

### Themes
- **Classic**: Dark blue with vibrant piece colors
- **Neon**: Black background with glowing neon colors
- **Pastel**: Light warm background with soft colors
- **Dark**: GitHub-style dark theme
- **Ocean**: Deep blue ocean-inspired palette

### Timer
- Enable/disable game timer
- Configurable duration (1-30 minutes)
- Game ends when timer reaches zero

## API Endpoints

| Method | Path          | Description           |
|--------|---------------|-----------------------|
| GET    | /health       | Health check          |
| GET    | /api/scores   | Get leaderboard       |
| POST   | /api/scores   | Submit a new score    |
| GET    | /api/config   | Get current config    |
| PUT    | /api/config   | Update config         |

## Data Models

### ScoreEntry
```json
{
  "id": "uuid",
  "player_name": "string",
  "score": 12000,
  "level": 5,
  "lines_cleared": 42,
  "difficulty": "medium",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### GameConfig
```json
{
  "difficulty": "medium",
  "theme": "classic",
  "timer_enabled": false,
  "timer_minutes": 5
}
```
