# Memory Game - Feature Documentation

## 1. Game Features

### Card Matching Mechanics

The game uses a 4x4 grid containing 16 cards that form 8 matching pairs. Each card has a hidden numeric value from 1 to 8, with exactly two cards sharing each value. The board is randomly shuffled at the start of every game using the Fisher-Yates algorithm via Rust's `rand::seq::SliceRandom`.

Players flip cards one at a time. When two cards are flipped:

- If the values match, both cards are marked as matched and remain revealed permanently.
- If the values do not match, both cards stay visible for 1 second before automatically flipping back face-down.
- A maximum of 2 cards can be flipped at any time. Flipping a third card automatically resets previously unmatched flipped cards.

Server-side validation prevents:

- Flipping a card that is already flipped.
- Flipping a card that is already matched.
- Flipping a card in a completed game.
- Flipping a card at an invalid board position (outside 0-15).

### Scoring System

The score is calculated using the formula: `max(1000 - (moves * 10), 100)`.

| Scenario | Moves | Score |
|----------|-------|-------|
| Perfect game (minimum possible moves) | 8 | 920 |
| Average game | 20 | 800 |
| Struggling game | 50 | 500 |
| Floor (any game with 90+ moves) | 90+ | 100 |

A "move" is counted each time a player flips the second card in a pair attempt. Flipping the first card of a new pair does not increment the move counter. The minimum score is always 100 regardless of how many moves are taken.

### Game Flow

1. **Player Registration** - The player enters their name on the start screen. The name is sent to `POST /api/players`. Names must be unique; duplicate names return a conflict error.
2. **Game Creation** - After registration, a new game session is created via `POST /api/games` with the player's ID. The backend generates a shuffled 16-card board and stores it in the database.
3. **Gameplay** - The player clicks cards to flip them. Each flip sends `POST /api/games/:id/flip` with the card position. The backend validates the move, updates the board state, and returns the updated game state.
4. **Completion** - When all 8 pairs are matched, the game status changes to `completed`, the final score is calculated, and the completion timestamp is recorded. A modal overlay displays the final score and move count.
5. **Replay** - The player can start a new game from the completion screen via the "Play Again" button, which returns to the player registration screen.

## 2. API Features

### POST /api/players

Registers a new player.

**Request:**
```json
{
  "name": "string"
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "name": "Alice"
}
```

**Response (409 Conflict):**
```json
{
  "error": "Player name already exists"
}
```

### POST /api/games

Creates a new game session with a shuffled board.

**Request:**
```json
{
  "player_id": 1
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "player_id": 1,
  "board": [
    { "position": 0, "value": null, "flipped": false, "matched": false },
    { "position": 1, "value": null, "flipped": false, "matched": false }
  ],
  "moves": 0,
  "matches_found": 0,
  "total_pairs": 8,
  "status": "in_progress",
  "score": 0
}
```

All card values are hidden (`null`) in the response. The actual values are stored server-side only.

### GET /api/games/:id

Returns the current state of a game. Card values are only revealed for cards that are currently flipped or matched.

**Response (200 OK):**
```json
{
  "id": 1,
  "player_id": 1,
  "board": [
    { "position": 0, "value": 3, "flipped": true, "matched": false },
    { "position": 1, "value": null, "flipped": false, "matched": false }
  ],
  "moves": 5,
  "matches_found": 2,
  "total_pairs": 8,
  "status": "in_progress",
  "score": 0
}
```

**Response (404 Not Found):**
```json
{
  "error": "Game not found"
}
```

### POST /api/games/:id/flip

Flips a card at the specified position.

**Request:**
```json
{
  "position": 5
}
```

**Response (200 OK):**
```json
{
  "game": {
    "id": 1,
    "player_id": 1,
    "board": [ ... ],
    "moves": 6,
    "matches_found": 3,
    "total_pairs": 8,
    "status": "in_progress",
    "score": 0
  },
  "matched": true
}
```

The `matched` field is:
- `true` when the second card of a pair matches the first.
- `false` when the second card does not match.
- `null` when only one card is flipped (first of a new pair).

**Error Responses (400 Bad Request):**
- `{"error": "Game already completed"}` - Game is already finished.
- `{"error": "Invalid position"}` - Position is outside the board range.
- `{"error": "Card already matched"}` - Card at that position was already matched.
- `{"error": "Card already flipped"}` - Card at that position is already face-up.

### GET /api/leaderboard

Returns the top 10 scores across all completed games, sorted by score descending.

**Response (200 OK):**
```json
[
  { "player_name": "Alice", "score": 920, "moves": 8 },
  { "player_name": "Bob", "score": 800, "moves": 20 }
]
```

### GET /api/players/:id/stats

Returns statistics for a specific player.

**Response (200 OK):**
```json
{
  "id": 1,
  "name": "Alice",
  "games_played": 10,
  "games_won": 7,
  "best_score": 920,
  "average_moves": 15.3
}
```

**Response (404 Not Found):**
```json
{
  "error": "Player not found"
}
```

## 3. Frontend Features

### Components

**PlayerForm** - The entry point of the application. Displays a centered card with a text input for the player's name and a "Start Game" button. The button is disabled when the input is empty or while the registration request is in flight. Error messages from the API (such as duplicate names) are displayed below the button in red text.

**GameBoard** - The main gameplay screen. Renders a 4x4 CSS grid of Card components and includes the ScoreBoard above the grid. Card clicks are disabled while a flip animation is in progress or while waiting for the server response. When the game status is `completed`, the GameOver modal is shown as an overlay. The component uses TanStack Query to fetch and cache the game state.

**Card** - An individual card in the grid. Each card has two faces: a back face showing a "?" symbol with an indigo gradient background, and a front face showing the card's numeric value with a color-coded background. The 8 different card values map to 8 distinct colors: red, blue, green, yellow, purple, pink, orange, and teal. Matched cards display with reduced opacity (70%) and a green ring indicator.

**ScoreBoard** - A horizontal stats bar displayed above the game board showing three metrics: the current move count, the match progress (e.g., "3/8"), and a live elapsed timer in minutes:seconds format. The timer starts when the component mounts and stops when the game is completed.

**GameOver** - A full-screen modal overlay with a dark backdrop that appears when the game is completed. Shows a "You Won!" heading in green, the final score, the total move count, and a "Play Again" button that restarts the flow.

**Leaderboard** - A full-page table view of the top scores. Displays rank, player name, score, and moves for each entry. Shows a "No scores yet" message when the leaderboard is empty. Includes a "Back" button to return to the previous screen.

### Animations and UI Interactions

- **Card Flip Animation** - Cards use a CSS 3D transform (`rotate-y-180`) with a 500ms transition to create a realistic flip effect. The `perspective`, `preserve-3d`, and `backface-hidden` CSS utilities enable the 3D card rotation.
- **Hover Effects** - Face-down cards scale up slightly (`hover:scale-105`) and increase their shadow on hover to provide visual feedback.
- **Mismatch Delay** - When two non-matching cards are flipped, they remain visible for 1 second (1000ms setTimeout) before the board refreshes and flips them back, giving the player time to memorize the values.
- **Disabled State** - During flip animations and server requests, all card clicks are blocked to prevent rapid clicking from creating inconsistent state.
- **Loading States** - Both the PlayerForm and GameBoard show loading indicators ("Starting..." and "Loading..." respectively) while waiting for API responses.

## 4. Database Features

### Data Persistence

The application uses SQLite via SQLx for persistent storage. All game logic runs server-side with the database as the source of truth.

**players table** - Stores registered players with unique names and a creation timestamp.

| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| name | TEXT | NOT NULL UNIQUE |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

**games table** - Stores each game session including the serialized board state (as JSON), move count, match progress, game status, score, and timestamps.

| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| player_id | INTEGER | FOREIGN KEY -> players(id) |
| board | TEXT | NOT NULL (JSON serialized array of cards) |
| moves | INTEGER | DEFAULT 0 |
| matches_found | INTEGER | DEFAULT 0 |
| total_pairs | INTEGER | DEFAULT 8 |
| status | TEXT | DEFAULT 'in_progress' |
| score | INTEGER | DEFAULT 0 |
| started_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |
| completed_at | TIMESTAMP | NULLABLE |

**flips table** - Records every individual card flip as an audit trail with the game ID, card position, and timestamp.

| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| game_id | INTEGER | FOREIGN KEY -> games(id) |
| position | INTEGER | NOT NULL |
| flipped_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

### Leaderboard Tracking

The leaderboard is computed dynamically by querying the top 10 completed games ordered by score descending. It joins the `games` and `players` tables to display player names alongside their scores and move counts. Player statistics (games played, games won, best score, average moves) are also computed via aggregate queries at request time.

### Transaction Safety

The flip_card operation uses a database transaction to ensure atomicity. The board state update, move counter increment, and flip audit log entry are all committed together or rolled back on failure.

## 5. How to Play

1. Open the application in your browser.
2. Enter your name in the text field and click "Start Game".
3. You will see a 4x4 grid of face-down cards, each showing a "?" symbol.
4. Click any card to flip it and reveal its hidden number and color.
5. Click a second card to flip it. This counts as one move.
   - If both cards show the same number, they are a match. They stay revealed with a green border and reduced opacity.
   - If the numbers differ, both cards flip back face-down after 1 second.
6. Keep track of where you saw each number to find matches faster.
7. Continue flipping pairs until all 8 matches are found.
8. When the game is complete, a results screen shows your final score and total moves.
9. Your score starts at 1000 and decreases by 10 for each move, with a minimum score of 100. Fewer moves means a higher score.
10. Click "Play Again" to start a new game.
