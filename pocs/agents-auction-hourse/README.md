# Agent Auction House

A multi-agent auction system where AI agents (Claude, Gemini, Copilot, Codex) compete by bidding sequentially on fun items. Each agent sees previous bids before placing theirs, creating strategic depth. The agent that wins the most items while spending the least is declared the Savviest Bidder.

## How It Works

1. Pick exactly 3 agents from: Claude, Gemini, Copilot, Codex
2. Configure each agent's model and budget (default $100)
3. 3 rounds of sequential bidding on random fun items
4. Each agent sees previous bids before placing theirs
5. Highest bidder wins the item
6. Winner = most items won, tiebreaker = least total spent

## Fun Auction Items

Agents bid on items like: Ancient Dragon Egg, Time-Travel Microwave, Haunted Rubber Duck, Quantum Burrito, Singing Cactus, Telepathic Toaster, Portal Gun (Slightly Used), Infinite Coffee Mug, Mass Destruction Cat, and more.

## Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | React 19, TypeScript, Vite, TanStack (Router, Query, Table), Tailwind CSS |
| Backend | Go 1.24, Gin Gonic, SQLite |
| Agents | CLI subprocesses (claude, gemini, copilot, codex) |
| Streaming | Server-Sent Events (SSE) |
| Screenshots | Playwright |

## Project Structure

```
agents-auction-hourse/
├── backend/
│   ├── main.go
│   ├── router/router.go
│   ├── handlers/
│   │   ├── auction.go
│   │   ├── history.go
│   │   └── stream.go
│   ├── engine/
│   │   ├── auction.go
│   │   └── bidding.go
│   ├── agents/
│   │   ├── registry.go
│   │   ├── runner.go
│   │   ├── claude.go
│   │   ├── gemini.go
│   │   ├── copilot.go
│   │   └── codex.go
│   ├── models/models.go
│   ├── persistence/db.go
│   └── sse/broadcaster.go
├── frontend/
│   └── src/
│       ├── routes/
│       ├── components/
│       ├── hooks/
│       ├── api/
│       └── types/
├── e2e/
│   └── screenshots.spec.ts
├── screenshots/
├── run.sh
├── stop.sh
├── test.sh
└── design-doc.md
```

## Running

```bash
./run.sh
```

Backend: http://localhost:3000
Frontend: http://localhost:5173

```bash
./stop.sh
```

## Testing

```bash
./test.sh
```

## Screenshots

### Setup Page
Pick exactly 3 agents from Claude, Gemini, Copilot, and Codex. Each agent card shows model selection and budget configuration.

![Setup Page](screenshots/setup.png)

### Agents Selected
Once 3 agents are selected, their cards expand to show model dropdowns and budget inputs. The Start Auction button becomes active.

![Agents Selected](screenshots/setup-selected.png)

### Live Auction
Agents bid sequentially on fun items. Each bid card reveals the agent's strategy and reasoning. The leaderboard sidebar tracks wins and remaining budgets in real-time.

![Live Auction](screenshots/auction.png)

### Auction Results
Final standings show the Savviest Bidder (most items won, least spent). Full round history displays every bid with reasoning. Fallback bids are marked with a yellow FALLBACK badge.

![Auction Results](screenshots/results.png)

### Auction History
The history page shows all past auctions in a sortable table. Expand any row to see the complete round-by-round breakdown with all bids, reasoning, response times, and fallback indicators.

![History Page](screenshots/history.png)

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auctions` | Start a new auction (3 agents required) |
| GET | `/api/auctions` | List all past auctions |
| GET | `/api/auctions/:id` | Get auction details with bids |
| GET | `/api/auctions/:id/stream` | SSE stream for live updates |
| GET | `/api/agents` | List available agents and models |

## Bid Fallback

If an agent's CLI output cannot be parsed as valid JSON:
- First bidder: fallback bid = $5
- Subsequent bidders: fallback bid = current highest + $1
- A yellow warning badge appears on fallback bids in the UI

## Sequential Bidding

Each round, agents bid one at a time in order. Agent 2 sees Agent 1's bid, Agent 3 sees both Agent 1 and Agent 2's bids. This creates interesting strategic dynamics where later bidders can adjust their strategy based on what they've seen.
