# Memory Game - Design Document

## Architecture Overview

A card-matching memory game with a Rust backend (Axum), React frontend (TypeScript + Vite + Tailwind), and SQLite database. Players flip cards to find matching pairs. The game tracks scores, moves, and maintains a leaderboard.

### System Components
- **Backend**: Rust (Axum) REST API on port 3000, serving game state, scores, and leaderboard
- **Frontend**: React 19 (TypeScript, Vite, Tailwind CSS, TanStack React Query) SPA on port 5173
- **Database**: SQLite via SQLx with foreign key enforcement and transaction safety

## Backend API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/players | Register a player name |
| GET | /api/players/{id}/stats | Get player statistics |
| POST | /api/games | Create a new game session |
| GET | /api/games/{id} | Get game state |
| POST | /api/games/{id}/flip | Flip a card |
| GET | /api/leaderboard | Get top 10 scores |

### Request/Response Shapes

**POST /api/players**
- Request: `{ "name": string }`
- Success (201): `{ "id": number, "name": string }`
- Conflict (409): `{ "error": "Player name already exists" }`

**GET /api/players/{id}/stats**
- Success (200): `{ "id": number, "name": string, "games_played": number, "games_won": number, "best_score": number, "average_moves": number }`
- Not Found (404): `{ "error": "Player not found" }`

**POST /api/games**
- Request: `{ "player_id": number }`
- Success (201): `GameResponse` (board cards have `value: null` since all face-down)

**GET /api/games/{id}**
- Success (200): `GameResponse`
- Not Found (404): `{ "error": "Game not found" }`

**POST /api/games/{id}/flip**
- Request: `{ "position": number }`
- Success (200): `{ "game": GameResponse, "matched": boolean | null }`
- Bad Request (400): `{ "error": "Game already completed" | "Invalid position" | "Card already matched" | "Card already flipped" }`
- Not Found (404): `{ "error": "Game not found" }`

**GameResponse shape:**
```
{
  "id": number,
  "player_id": number,
  "board": [{ "position": number, "value": number | null, "flipped": boolean, "matched": boolean }],
  "moves": number,
  "matches_found": number,
  "total_pairs": number,
  "status": "in_progress" | "completed",
  "score": number
}
```

**GET /api/leaderboard**
- Success (200): `[{ "player_name": string, "score": number, "moves": number }]`
- Returns top 10 completed games ordered by score descending

### Game Logic (Backend)
- Game board: 4x4 grid (16 cards, 8 pairs with values 1-8)
- Card values are hidden (null) in responses unless the card is flipped or matched
- Each flip request validates: game not completed, valid position, card not already matched, card not already flipped
- If 2 unmatched cards are already flipped when a new flip arrives, they are reset to face-down before flipping the new card
- A move is counted when 2 cards are flipped face-up
- When 2 flipped cards match: both are marked as matched (and flipped is reset to false), matches_found increments
- When all 8 pairs found: status set to "completed", completed_at timestamp recorded
- Score formula: `max(1000 - (moves * 10), 100)` (minimum score is 100)
- The flip endpoint uses a database transaction for atomicity and enables foreign keys via PRAGMA within the transaction
- Each flip is recorded in the flips table for audit

### CORS Configuration
- Allows origin: `http://localhost:5173`
- Allowed methods: GET, POST
- Allowed headers: content-type

## Frontend Components

| Component | Props | Description |
|-----------|-------|-------------|
| App | - | Root component, manages page state (home/game/leaderboard), wraps in QueryClientProvider |
| PlayerForm | `onGameStart: (gameId, playerId) => void` | Name entry form, creates player and game via mutation, then calls onGameStart |
| GameBoard | `gameId: number, onPlayAgain: () => void` | 4x4 grid of cards, manages flip state, shows ScoreBoard and GameOver overlay |
| Card | `card: CardData, index: number, onClick: () => void, disabled: boolean` | Individual card with 3D flip animation, color-coded by value (8 colors), shows "?" when face-down |
| ScoreBoard | `moves: number, matchesFound: number, totalPairs: number, isComplete: boolean` | Displays moves, matches (found/total), and elapsed time with a client-side timer |
| GameOver | `score: number, moves: number, onPlayAgain: () => void` | Modal overlay showing final score, moves, and play again button |
| Leaderboard | `onBack: () => void` | Fetches and displays top scores in a ranked table with player name, score, and moves |

### Frontend State Management
- TanStack React Query for server state (game data, leaderboard)
- Local React state for page navigation (`home | game | leaderboard`), current gameId, and playerId
- GameBoard uses local `flipping` state to prevent rapid clicks during the 1-second reveal delay
- After two non-matching cards are revealed, a 1-second timeout fires before refreshing game state from the server

### Frontend Interactions
1. Player enters name on home page -> POST /api/players, then POST /api/games
2. On success, navigates to game page with gameId and playerId
3. Player clicks a face-down card -> POST /api/games/{id}/flip
4. Frontend receives FlipResponse, updates query cache with new game state
5. If two cards are flipped (non-matched), a 1-second delay shows them before re-fetching state
6. Game over -> GameOver modal overlay appears with score and moves
7. Play again -> creates new game with same playerId, invalidates query cache
8. Leaderboard accessible from home page via button

### Card Color Mapping
| Value | Color |
|-------|-------|
| 1 | Red |
| 2 | Blue |
| 3 | Green |
| 4 | Yellow |
| 5 | Purple |
| 6 | Pink |
| 7 | Orange |
| 8 | Teal |

## Database Schema

### players table
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| name | TEXT | NOT NULL UNIQUE |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

### games table
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| player_id | INTEGER | NOT NULL, FOREIGN KEY -> players(id) |
| board | TEXT | NOT NULL (JSON serialized array of Card objects) |
| moves | INTEGER | DEFAULT 0 |
| matches_found | INTEGER | DEFAULT 0 |
| total_pairs | INTEGER | DEFAULT 8 |
| status | TEXT | DEFAULT 'in_progress' |
| score | INTEGER | DEFAULT 0 |
| started_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |
| completed_at | TIMESTAMP | NULLABLE |

### flips table
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| game_id | INTEGER | NOT NULL, FOREIGN KEY -> games(id) |
| position | INTEGER | NOT NULL |
| flipped_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

### Indexes
| Index | Table | Column |
|-------|-------|--------|
| idx_games_player_id | games | player_id |
| idx_games_status | games | status |
| idx_flips_game_id | flips | game_id |

## Applied Fixes

### Transaction Safety
- The flip_card handler wraps all database mutations (game update + flip insert) in a SQLite transaction using `pool.begin()` and `tx.commit()`
- Foreign keys are enabled via `PRAGMA foreign_keys = ON` both at database initialization and within each transaction

### Foreign Key Enforcement
- The db::init_db function runs `PRAGMA foreign_keys = ON` at startup
- The flip_card transaction re-enables the pragma since SQLite pragmas do not persist across connections
- The games.player_id and flips.game_id columns are declared with NOT NULL in the runtime schema (db.rs), ensuring referential integrity

### Error Handling
- All handler functions return `Response` directly, with explicit status codes for each error case
- A shared `internal_error()` helper returns 500 with a JSON error body
- Each database query and serialization step is individually matched for errors with early returns
- Player creation returns 409 on duplicate name
- Game and player lookups return 404 when not found
- Flip validation returns 400 for: completed game, invalid position, already matched card, already flipped card

### Schema Differences: db/schema.sql vs Runtime (db.rs)
- The schema.sql file uses `REFERENCES` shorthand for foreign keys; db.rs uses explicit `FOREIGN KEY` clauses
- The schema.sql file does not have NOT NULL on player_id and game_id foreign key columns; db.rs adds NOT NULL to both
- The schema.sql file includes index definitions; db.rs does not create indexes (they are only in the .sql file)

## Integration Points

- Frontend fetches game state via REST API after each flip
- Backend validates all game logic server-side to prevent cheating
- Card values are hidden in API responses for face-down, unmatched cards
- SQLite database stores all game state persistently
- CORS configured on backend to allow frontend dev server at localhost:5173
- Frontend uses TanStack React Query cache for optimistic-style updates (setQueryData from flip response)
- Connection pool limited to 5 connections via SqlitePoolOptions
