# Memory Game

A full-stack card-matching memory game built with Rust (Axum), React (TypeScript), and SQLite.

## Quick Start

### Database
```bash
cd db
bash create-schema.sh
```

### Backend
```bash
cd backend
bash run.sh
```
Server starts on http://localhost:3000

### Frontend
```bash
cd frontend
bash run.sh
```
App starts on http://localhost:5173

## How to Play

1. Enter your name and start a game
2. Click cards to flip them - find matching pairs
3. Match all 8 pairs to complete the game
4. Score is based on number of moves: fewer moves = higher score
5. Check the leaderboard to see top scores

## Tech Stack

- **Backend**: Rust 1.93+, Axum 0.8, SQLx, Tokio
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, TanStack Query
- **Database**: SQLite
- **Testing**: Rust test, Vitest, Playwright, K6

## Documentation

- [Design Document](design-doc.md)
- [Feature Documentation](review/2026-02-28/features.md)
- [Code Review](review/2026-02-28/code-review.md)
- [Security Review](review/2026-02-28/sec-review.md)
- [Changes Summary](review/2026-02-28/summary.md)
- [Changelog](changelog.md)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/players | Register player |
| GET | /api/players/{id}/stats | Player statistics |
| POST | /api/games | Create game |
| GET | /api/games/{id} | Get game state |
| POST | /api/games/{id}/flip | Flip a card |
| GET | /api/leaderboard | Top 10 scores |

## Running Tests

### Backend
```bash
cd backend
cargo test
```

### Frontend
```bash
cd frontend
bun run test
```

### E2E (requires backend + frontend running)
```bash
cd frontend
bunx playwright test
```

### Stress Test (requires backend running)
```bash
cd k6
bash run-stress-test.sh
```
