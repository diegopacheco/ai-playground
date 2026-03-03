# Ghost

https://github.com/adamveld12/ghost

Commit intentions, not code.

Ghost is a CLI that flips the git workflow: instead of committing code, you commit prompts. An AI coding agent generates the artifacts; the commit captures both the intent and the output. Your git history becomes a chain of prompts + their results.

Supports claude, gemini, codex, and opencode — swap agents per-commit or set a default.

## Experience Notes

* Installion was easy
* Did not like Opus was not the default model for claude
* 

## Result

```
ghost commit -m "create a memory game with frontend written in React, Tanstack, vite, bun and the backend in rust with sqllite. make sure there is a leaderboard ui, there is a timmer for 120s to finish the game and the game tracks how many moves."
```

```
  ▸ ghost
  agent     claude
  model     claude-sonnet-4-6
  intent    create a memory game with frontend written in React, Tanstack, vite, bun and the backend in rust with sqllite. make sure there is a leaderboard ui, there is a timmer for 120s to finish the game and the game tracks how many moves.

⠙ running claude…
```

```
⠴ running claude…Everything is built and tested. Here's the full project structure:

```
app/
├── start.sh          # starts backend + frontend
├── stop.sh           # stops both
├── backend/          # Rust + actix-web + rusqlite
│   ├── Cargo.toml
│   └── src/main.rs
└── frontend/         # React + Vite + Bun + TanStack Query
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── App.css
        ├── index.css
        ├── api.ts
        └── components/
            ├── Board.tsx   # game logic + timer + move counter
            ├── Card.tsx    # flip animation
            ├── Timer.tsx   # countdown with danger pulse
            └── Leaderboard.tsx  # TanStack Query, auto-refreshes
```

**How to run:**
```bash
cd app
./start.sh
# Backend: http://localhost:8080
# Frontend: http://localhost:5173
```

**Features implemented:**
- **16 cards** (8 emoji pairs), shuffled each game
- **120s countdown timer** — turns red and pulses at ≤20s
- **Move counter** — increments each time you flip a second card
- **Win modal** — shows moves + time used, lets you enter name and save score
- **Time's up modal** — shown if timer hits 0
- **Leaderboard** — sorted by fewest moves then fastest time, auto-refreshes every 5s
- **Backend** — Rust/actix-web with SQLite via rusqlite (bundled, no system dep)
[main bceb7426] added Ghost ghost-meta ghost-prompt: create a memory game with frontend written in React, Tanstack, vite, bun and the backend in rust with sqllite. make sure there is a leaderboard ui, there is a timmer for 120s to finish the game and the game tracks how many moves. ghost-agent: claude ghost-model: claude-sonnet-4-6 ghost-session: f1ad9cbd-1392-4652-885b-94077c4a31e7 ghost-files: pocs/Ghost-Fun/app/frontend/src/App.css,pocs/Ghost-Fun/app/frontend/src/App.tsx,pocs/Ghost-Fun/app/frontend/src/components/Board.tsx,pocs/Ghost-Fun/app/frontend/src/components/Card.tsx,pocs/Ghost-Fun/app/frontend/src/components/Leaderboard.tsx,pocs/Ghost-Fun/app/frontend/src/components/Timer.tsx
 Date: Mon Mar 2 22:17:51 2026 -0800
 6 files changed, 481 insertions(+), 54 deletions(-)
 create mode 100644 pocs/Ghost-Fun/app/frontend/src/components/Board.tsx
 create mode 100644 pocs/Ghost-Fun/app/frontend/src/components/Card.tsx
 create mode 100644 pocs/Ghost-Fun/app/frontend/src/components/Leaderboard.tsx
 create mode 100644 pocs/Ghost-Fun/app/frontend/src/components/Timer.tsx

  ✓ ghost: tagged agent commit with ghost-meta 'create a memory game with frontend written in React, Tanstack, vite, bun and the backend in rust with sqllite. make sure there is a leaderboard ui, there is a timmer for 120s to finish the game and the game tracks how many moves.'
[main 2d0a074a] create a memory game with frontend written in React, Tanstack, vite, bun and the backend in rust with sqllite. make sure there is a leaderboard ui, there is a timmer for 120s to finish the game and the game tracks how many moves. (follow-up)
 3 files changed, 25 insertions(+), 61 deletions(-)
 create mode 100755 pocs/Ghost-Fun/app/start.sh
 create mode 100755 pocs/Ghost-Fun/app/stop.sh
you  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ai
     0%                                  100%
     100% AI code accepted

  ✓ ghost: committed follow-up changes 'create a memory game with frontend written in React, Tanstack, vite, bun and the backend in rust with sqllite. make sure there is a leaderboard ui, there is a timmer for 120s to finish the game and the game tracks how many moves.'
```