# Agent Auction House - Design Document

## Overview

A multi-agent auction system where AI agents compete by bidding on fun items. Users pick exactly 3 agents from (claude, gemini, copilot, codex), configure budgets (default $100 each, tweakable per agent), and watch them bid sequentially across 3 rounds. Bidding is sequential so each agent can see previous bids before placing theirs. The agent that wins the most items with the least total spend is declared the winner.

Agents are invoked via CLI (claude, gemini, copilot, codex) following the same pattern as agent-debate-club.

## Architecture

```
┌─────────────────────┐       ┌──────────────────────┐
│   React Frontend    │◄─SSE──│    Go Backend (Gin)   │
│  TanStack Router    │──HTTP─►│    Port 3000          │
│  TanStack Query     │       │                        │
│  TanStack Table     │       │  ┌──────────────────┐  │
│  Port 5173          │       │  │  Auction Engine   │  │
└─────────────────────┘       │  │  (3 rounds seq)   │  │
                              │  └────────┬─────────┘  │
                              │           │             │
                              │  ┌────────▼─────────┐  │
                              │  │  Agent Runner     │  │
                              │  │  (CLI subprocess) │  │
                              │  └────────┬─────────┘  │
                              │           │             │
                              │  ┌────────▼─────────┐  │
                              │  │  SQLite Storage   │  │
                              │  └──────────────────┘  │
                              └──────────────────────┘
                                         │
                          ┌──────────────┼──────────────┐
                          │              │              │
                     ┌────▼───┐    ┌────▼───┐    ┌────▼───┐
                     │ claude │    │ gemini │    │ copilot│
                     │  CLI   │    │  CLI   │    │  CLI   │
                     └────────┘    └────────┘    └────────┘
```

## Key Design Decisions

- **Sequential Bidding**: Agents bid one at a time. Each agent sees previous bids in the current round before placing theirs. This creates strategic depth.
- **3 Agents**: User picks exactly 3 from claude, copilot, gemini, codex.
- **Budget**: Default $100 per agent, user can tweak each individually via UI.
- **Bid Parse Fallback**: If an agent's response cannot be parsed: $5 if they are the first bidder in the round, otherwise current highest bid + $1. This is logged and the UI shows a warning indicator.
- **Fast Timeouts**: 30 second timeout per agent CLI call.

## Tech Stack

### Frontend
- React 19 + TypeScript
- Vite (bundler)
- TanStack Router (routing)
- TanStack Query (data fetching)
- TanStack Table (history/results tables)
- Tailwind CSS (styling)
- Bun (runtime/package manager)

### Backend
- Go 1.24 + Gin Gonic
- SQLite (mattn/go-sqlite3)
- SSE (Server-Sent Events for live updates)
- Modular file structure

## Go Backend Structure

```
backend/
├── main.go
├── go.mod
├── go.sum
├── router/
│   └── router.go
├── handlers/
│   ├── auction.go
│   ├── history.go
│   └── stream.go
├── engine/
│   ├── auction.go
│   └── bidding.go
├── agents/
│   ├── registry.go
│   ├── runner.go
│   ├── claude.go
│   ├── gemini.go
│   ├── copilot.go
│   └── codex.go
├── models/
│   └── models.go
├── persistence/
│   └── db.go
└── sse/
    └── broadcaster.go
```

## Frontend Structure

```
frontend/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── routeTree.gen.ts
│   ├── routes/
│   │   ├── __root.tsx
│   │   ├── index.tsx
│   │   ├── auction.$id.tsx
│   │   └── history.tsx
│   ├── components/
│   │   ├── AuctionSetup.tsx
│   │   ├── AuctionLive.tsx
│   │   ├── BidCard.tsx
│   │   ├── RoundSummary.tsx
│   │   ├── Leaderboard.tsx
│   │   └── HistoryTable.tsx
│   ├── hooks/
│   │   ├── useAuctionSSE.ts
│   │   └── useAuctions.ts
│   ├── api/
│   │   └── auctions.ts
│   └── types/
│       └── index.ts
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## Auction Flow

### Setup Phase
1. User selects exactly 3 agents from: claude, gemini, copilot, codex
2. User picks the model for each agent
3. User sets budget per agent (default $100, tweakable)
4. System randomly picks 3 fun items from the pool for the 3 rounds

### Fun Auction Items (Pool)

Items are randomly selected from a curated pool:
- Ancient Dragon Egg 🥚🐉
- Time-Travel Microwave 🕐🔥
- Haunted Rubber Duck 🦆👻
- Quantum Burrito 🌯⚛️
- Invisible Bicycle 🚲👻
- Singing Cactus 🌵🎤
- Telepathic Toaster 🍞🧠
- Unicorn Roller Skates ⛸️🦄
- Gravity-Defying Hammock 🪢🚀
- Exploding Pinata of Wisdom 🪅💥
- Cursed Disco Ball 🪩😈
- Levitating Pizza Slice 🍕🛸
- Mind-Reading Sunglasses 🕶️🧠
- Portal Gun (Slightly Used) 🔫🌀
- Shrink Ray Flashlight 🔦🔬
- Enchanted USB Cable (Always Right Side Up) 🔌✨
- Self-Driving Shopping Cart 🛒🤖
- Infinite Coffee Mug ☕♾️
- Philosophers Stone (Cracked) 💎🔨
- Mass Destruction Cat 🐱💣

### Sequential Bidding Phase (Per Round)

Each round, agents bid one at a time in order:
1. Item is revealed
2. Agent 1 bids (sees item + budgets, no previous bids)
3. Agent 2 bids (sees item + budgets + Agent 1's bid)
4. Agent 3 bids (sees item + budgets + Agent 1 & 2's bids)
5. Highest bidder wins, their budget is reduced
6. Ties broken by lowest bid timestamp

### Bid Parsing Fallback

If agent output cannot be parsed as valid JSON with a bid field:
- If first bidder in round: fallback bid = $5
- If not first: fallback bid = current highest bid + $1
- A `fallback: true` flag is set on the bid record
- UI shows a warning icon next to fallback bids
- Full raw output is logged for debugging

### Agent Prompt Template

```
You are bidding in a fun auction. You have ${remaining_budget} auction dollars left.
There are ${rounds_left} rounds remaining after this one.

ITEM UP FOR BID: ${item_name} ${item_emoji}

Your remaining budget: $${remaining_budget}
Other agents budgets: ${agent_budgets}

${previous_bids_in_round}

Previous round results:
${round_history}

RULES:
- Bid an integer amount you can afford
- Goal: win the most items while spending the least
- Winner = most items won, tiebreaker = least total spent
- You can see what others bid before you (sequential auction)

Respond with ONLY a JSON object, nothing else:
{"bid": <integer>, "reasoning": "<one sentence strategy>"}
```

### Scoring

After 3 rounds:
- Primary: Most items won
- Tiebreaker: Least total money spent
- Display: Items won with emoji, total spent, remaining budget, reasoning per bid

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auctions` | Create and start a new auction |
| GET | `/api/auctions` | List all past auctions |
| GET | `/api/auctions/:id` | Get auction details with all bids |
| GET | `/api/auctions/:id/stream` | SSE stream for live auction updates |
| GET | `/api/agents` | List available agents |

### POST /api/auctions Request Body

```json
{
  "agents": [
    {"name": "claude", "model": "opus", "budget": 100},
    {"name": "gemini", "model": "gemini-2.5-pro", "budget": 100},
    {"name": "copilot", "model": "claude-sonnet-4", "budget": 120}
  ]
}
```

### SSE Events

| Event | Data | When |
|-------|------|------|
| `round_start` | `{round, item, item_emoji}` | New round begins |
| `agent_thinking` | `{agent, round}` | Agent is computing bid |
| `agent_bid` | `{agent, bid, reasoning, round, fallback}` | Agent submits bid |
| `round_result` | `{winner, item, winning_bid, round}` | Round ends |
| `auction_over` | `{final_standings, winner}` | Auction complete |
| `error` | `{message}` | Error occurred |

## Database Schema

```sql
CREATE TABLE auctions (
    id TEXT PRIMARY KEY,
    winner TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME
);

CREATE TABLE auction_agents (
    id TEXT PRIMARY KEY,
    auction_id TEXT,
    agent_name TEXT,
    model TEXT,
    initial_budget INTEGER,
    remaining_budget INTEGER,
    items_won INTEGER DEFAULT 0,
    FOREIGN KEY (auction_id) REFERENCES auctions(id)
);

CREATE TABLE rounds (
    id TEXT PRIMARY KEY,
    auction_id TEXT,
    round_number INTEGER,
    item_name TEXT,
    item_emoji TEXT,
    winner_agent TEXT,
    winning_bid INTEGER,
    FOREIGN KEY (auction_id) REFERENCES auctions(id)
);

CREATE TABLE bids (
    id TEXT PRIMARY KEY,
    round_id TEXT,
    auction_id TEXT,
    agent_name TEXT,
    amount INTEGER,
    reasoning TEXT,
    fallback INTEGER DEFAULT 0,
    raw_output TEXT,
    response_time_ms INTEGER,
    FOREIGN KEY (round_id) REFERENCES rounds(id),
    FOREIGN KEY (auction_id) REFERENCES auctions(id)
);
```

## Agent CLI Commands

| Agent | Command | Args | Timeout |
|-------|---------|------|---------|
| Claude | `claude` | `-p "<prompt>" --model <model> --dangerously-skip-permissions` | 30s |
| Gemini | `gemini` | `-y -p "<prompt>"` | 30s |
| Copilot | `copilot` | `--allow-all --model <model> -p "<prompt>"` | 30s |
| Codex | `codex` | `exec --full-auto --model <model> "<prompt>"` | 30s |

## UI Pages

### 1. Setup Page (`/`)
- Pick exactly 3 agents from claude, copilot, gemini, codex
- Model dropdown per agent
- Budget input per agent (default $100)
- Start Auction button

### 2. Live Auction Page (`/auction/:id`)
- Current round indicator (Round 1/3, 2/3, 3/3)
- Item card with emoji and name (large, centered)
- Sequential bid reveal: each agent's bid card appears one at a time
- Bid cards show: agent name, bid amount, reasoning, fallback warning if applicable
- Round winner announcement with confetti
- Running leaderboard sidebar with budget bars
- Savviest bidder highlight (most wins, least spent)

### 3. History Page (`/history`)
- TanStack Table with sortable columns
- Columns: Date, Agents, Winner, Items, Total Spent
- Expandable rows to see full round-by-round details
- Each bid shows reasoning and fallback indicators

## Scripts

```
run.sh      # Build and start backend + frontend
stop.sh     # Kill both processes
test.sh     # Curl test against API
```

## Playwright Screenshots

After implementation, Playwright tests will capture:
1. Setup page with agents configured
2. Live auction - round in progress with sequential bids
3. Live auction - bid reveal with reasoning
4. Live auction - final results and leaderboard
5. History page with past auctions

All screenshots linked in README.md.
