# Agent Werewolf

Multi-agent social deduction game where AI agents play Werewolf. One agent is randomly
assigned as the werewolf and must lie convincingly while villager agents try to identify
and eliminate the werewolf through conversation and voting.

Measures deception and deception-detection capabilities across different AI models.

## How It Works

1. Select 4-6 AI agents (Claude, Gemini, Copilot, Codex)
2. One agent is randomly assigned as the **Werewolf**, others are **Villagers**
3. Each round has:
   - **Night Phase**: Werewolf secretly eliminates a villager
   - **Day Phase - Discussion**: Surviving agents make statements, accuse, and defend
   - **Day Phase - Voting**: Agents vote to eliminate who they suspect is the werewolf
4. **Villagers win** if they vote out the werewolf
5. **Werewolf wins** if it survives until only 2 players remain
6. **Deception Score**: Number of rounds the werewolf survived

## Tech Stack

- **Backend**: Rust (actix-web, rusqlite, tokio)
- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS 4
- **Database**: SQLite with WAL mode
- **Streaming**: Server-Sent Events (SSE) for live game updates
- **Agents**: Claude, Gemini, Copilot, Codex via CLI

## Project Structure

```
agent-werewolf/
├── backend/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── models.rs
│       ├── db.rs
│       ├── engine.rs
│       ├── handlers.rs
│       ├── sse.rs
│       └── agents/
│           ├── mod.rs
│           ├── claude.rs
│           ├── gemini.rs
│           ├── copilot.rs
│           └── codex.rs
├── frontend/
│   ├── package.json
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   ├── game/[id]/page.tsx
│       │   └── history/page.tsx
│       ├── components/
│       │   ├── GameSetup.tsx
│       │   ├── GameLive.tsx
│       │   └── HistoryTable.tsx
│       ├── hooks/
│       │   └── useGameSSE.ts
│       ├── lib/
│       │   └── api.ts
│       └── types/
│           └── index.ts
├── e2e/
│   ├── package.json
│   ├── playwright.config.ts
│   └── screenshots.spec.ts
├── design-doc.md
├── run.sh
├── stop.sh
└── test.sh
```

## Running

```bash
./run.sh
```
- Backend: http://localhost:3000
- Frontend: http://localhost:3001

```bash
./stop.sh
```

## Testing

```bash
./test.sh
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/games | Create new game with selected agents |
| GET | /api/games | List all games |
| GET | /api/games/:id | Get game details |
| GET | /api/games/:id/stream | SSE stream for live updates |
| GET | /api/agents | List available agents |

## SSE Events

| Event | Description |
|-------|-------------|
| game_start | Game begins with agent list |
| night_phase | Werewolf hunting phase |
| elimination | Agent eliminated (night kill) |
| day_phase | Discussion begins |
| agent_thinking | Agent is processing |
| discussion | Agent makes a statement |
| voting_phase | Voting begins |
| vote | Agent casts a vote |
| vote_result | Voting outcome |
| game_over | Final results with role reveals |

## Screenshots

### Setup Page
![Setup Page](screenshots/01-setup.png)

### History Page
![History Page](screenshots/02-history.png)
