# Flight Agent MCP - Design Document

## Overview

Flight Agent MCP is a flight search application that uses AI agents (Claude or Codex) as intermediaries to search for flights using the travel-hacking-toolkit skills (Seats.aero and SerpAPI). Users select origin/destination airports with autocomplete, choose an agent, and receive a grid of flight results. Clicking a result opens the booking website.

## Technology Stack

### Backend
- Rust 1.94+ (edition 2024)
- Axum (web framework)
- Tokio (async runtime)

### Frontend
- React 19
- Bun (runtime/package manager)
- Vite (build tool)
- TanStack Query (data fetching)
- Tailwind CSS (styling)
- TypeScript

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Frontend (React 19)                       │
├─────────────────────────────────────────────────────────────────┤
│  SearchForm (autocomplete)  │  ResultsGrid (clickable rows)    │
└─────────────────────────────┴───────────────────────────────────┘
                              │
                              │ REST API (POST /api/search)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (Rust/Axum)                        │
├─────────────────────────────────────────────────────────────────┤
│  Routes  │  AgentRunner  │  Prompt Builder  │  JSON Parser     │
└──────────┴───────────────┴──────────────────┴──────────────────┘
                              │
                              │ CLI Subprocess
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Runners                              │
├─────────────────────────────────────────────────────────────────┤
│  Claude (claude -p)  │  Codex (codex exec)                     │
└──────────────────────┴──────────────────────────────────────────┘
                              │
                              │ curl (inside agent)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Travel Hacking Toolkit APIs                   │
├─────────────────────────────────────────────────────────────────┤
│  Seats.aero (award flights)  │  SerpAPI (cash prices)          │
└──────────────────────────────┴──────────────────────────────────┘
```

## UI Screens

### Search Screen

**Components**:
- Airport autocomplete input (From): searches by code, city, country, or airport name
- Airport autocomplete input (To): same behavior
- Date picker: defaults to 14 days from today
- Agent selector: dropdown with Claude and Codex options
- Search button: triggers the agent search

**Validation**:
- Both origin and destination must be selected from autocomplete
- Date must be provided
- Button disabled while searching

### Results Grid

**Components**:
- Header showing result count and agent used
- Table with columns: Date, Route, Airline, Price, Cabin, Stops, Duration, Source
- Color-coded cabin classes (first=gold, business=blue, premium=teal, economy=default)
- Source badges (purple for seats.aero, green for google_flights)
- Clickable rows that open booking URLs in new tab

## Backend Structure

```
backend/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── routes/
│   │   ├── mod.rs
│   │   └── search.rs
│   └── agents/
│       ├── mod.rs
│       ├── runner.rs
│       ├── claude.rs
│       └── codex.rs
```

## Frontend Structure

```
frontend/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css
│   ├── components/
│   │   ├── SearchForm.tsx
│   │   └── ResultsGrid.tsx
│   ├── api/
│   │   └── flights.ts
│   ├── data/
│   │   └── airports.ts
│   └── types/
│       └── index.ts
```

## API Endpoints

### POST /api/search

Search for flights using an AI agent.

**Request**:
```json
{
  "origin": "SFO",
  "origin_city": "San Francisco, United States",
  "destination": "NRT",
  "destination_city": "Tokyo, Japan",
  "date": "2026-04-18",
  "agent": "claude"
}
```

**Response**:
```json
{
  "results": [
    {
      "id": "flight-0",
      "date": "2026-04-18",
      "origin": "SFO",
      "destination": "NRT",
      "airline": "United Airlines",
      "price": "45000 miles",
      "cabin": "economy",
      "stops": "0",
      "duration": "11h 15m",
      "source": "seats.aero",
      "booking_url": "https://..."
    }
  ],
  "agent_used": "claude",
  "raw_output": "...",
  "error": null
}
```

### GET /api/agents

List available agents.

**Response**:
```json
[
  { "id": "claude", "name": "Claude" },
  { "id": "codex", "name": "Codex" }
]
```

## Agent Prompt Design

The backend constructs a prompt that includes:
1. The search parameters (origin, destination, date)
2. Pre-built curl commands for both Seats.aero and SerpAPI with API keys
3. Strict instructions to return only a JSON array with specified fields
4. No markdown, no explanations, just the JSON

The agent runs the curl commands, processes the responses, and returns structured JSON.

## Agent Runner Pattern

Follows the same subprocess pattern from agent-debate-club:
- Spawn CLI process (claude or codex) with the prompt
- Capture stdout
- Timeout after 180 seconds
- Parse JSON array from output
- Extract individual flight results

### Claude
```
claude -p "<prompt>" --model sonnet --dangerously-skip-permissions
```

### Codex
```
codex exec --full-auto "<prompt>"
```

## Data Flow

1. User selects airports via autocomplete and chooses agent
2. Frontend sends POST /api/search to backend
3. Backend builds prompt with API keys from environment
4. Backend spawns agent CLI subprocess with the prompt
5. Agent runs curl commands against Seats.aero and SerpAPI
6. Agent returns JSON array of flight results
7. Backend parses JSON and returns structured response
8. Frontend renders results in a sortable grid
9. User clicks a row to open booking URL in new tab

## Environment Variables

```
SEATS_AERO_API_KEY=your-seats-aero-key
SERPAPI_API_KEY=your-serpapi-key
```

## Error Handling

- Agent timeout: Returns error message after 180s
- Agent spawn failure: Returns error with agent type and reason
- Empty agent response: Returns error indicating empty response
- Invalid JSON from agent: Returns empty results array
- Missing API keys: Agent receives empty keys, APIs will return errors, agent handles gracefully
- Network errors: Frontend catches and displays connection error
