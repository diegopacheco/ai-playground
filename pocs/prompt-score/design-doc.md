# Prompt Score - Design Document

## Overview

Prompt Score is a web application that analyzes the quality of AI prompts. Users input prompts, receive quality scores across multiple dimensions, and get recommendations from multiple AI models with live progress updates via Server-Sent Events (SSE).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│  React 19 + TanStack Router + TailwindCSS + Vite + Bun + TS     │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │  Prompt Page │───▶│           Score Page                  │   │
│  │  - TextArea  │    │  - Progress Bar (SSE live updates)   │   │
│  │  - Char/Word │    │  - Quality Scores (6 dimensions)     │   │
│  │  - Analyze   │    │  - Model Analysis Results            │   │
│  └──────────────┘    └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Backend                                  │
│           Rust 2024 Edition + Tokio + Axum                      │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │  /api/analyze  │  │ Score Engine   │  │ Agent Executor  │   │
│  │  POST (SSE)    │─▶│ (dimensions)   │─▶│ (subprocess)    │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
│                                                  │               │
│                              ┌───────────────────┴───────┐      │
│                              ▼                           ▼      │
│                    ┌──────────────┐            ┌──────────────┐ │
│                    │ Claude CLI   │            │ Codex CLI    │ │
│                    │ opus-4.5     │            │ o3           │ │
│                    └──────────────┘            └──────────────┘ │
│                              ▼                           ▼      │
│                    ┌──────────────┐            ┌──────────────┐ │
│                    │ Copilot CLI  │            │ Gemini CLI   │ │
│                    │ sonnet4      │            │ gemini-3.0   │ │
│                    └──────────────┘            └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Frontend
- React 19
- TanStack Router
- TailwindCSS
- Vite
- Bun
- TypeScript

### Backend
- Rust 2024 edition (1.85+)
- Tokio async runtime
- Axum web framework
- tokio-stream for SSE
- futures for stream handling

## Pages

### Page 1: Prompt Input

- Large textarea for prompt input
- Real-time character count display
- Real-time word count display
- "Analyze" button (navigates to Score page with prompt data)

### Page 2: Score Dashboard

#### Progress Bar (SSE Live Updates)
- Visual progress bar with percentage
- Step counter (e.g., "Step 2 of 6")
- Status message showing current operation
- Spinning indicator during processing

#### Quality Dimensions (1-5 scale each)
| Dimension | Description |
|-----------|-------------|
| Quality | Overall prompt clarity and structure |
| Stack Definitions | Technical stack specifications |
| Clear Goals | Objectives and expected outcomes |
| Non-obvious Decisions | Edge cases and architectural choices |
| Security & Operations | Security considerations and operational aspects |
| Overall Effectiveness | Combined effectiveness score |

#### AI Model Analysis
Each model provides:
- Individual score (1-5)
- Text recommendations
- Improvement suggestions

Supported Models:
- claude/opus-4.5
- codex/o3
- copilot/sonnet4
- gemini/gemini-3.0

## API Endpoints

### POST /api/analyze (SSE Stream)
Request:
```json
{
  "prompt": "string"
}
```

Response (Server-Sent Events stream):
```
data: {"type":"start","total_steps":6,"message":"Starting analysis..."}

data: {"type":"scores","scores":{...},"step":1,"message":"Dimension scores calculated"}

data: {"type":"agent_start","agent":"claude/opus-4.5","step":2,"message":"Querying claude/opus-4.5..."}

data: {"type":"agent_done","agent":"claude/opus-4.5","result":{...},"step":2,"message":"claude/opus-4.5 completed"}

data: {"type":"complete","scores":{...},"model_results":[...],"message":"Analysis complete!"}
```

## SSE Progress Events

| Event Type | Fields | Description |
|------------|--------|-------------|
| start | total_steps, message | Initial event with total step count |
| scores | scores, step, message | Dimension scores calculated |
| agent_start | agent, step, message | Agent query started |
| agent_done | agent, result, step, message | Agent completed with result |
| complete | scores, model_results, message | All processing complete |

## Project Structure

```
prompt-score/
├── design-doc.md
├── run.sh
├── frontend/
│   ├── package.json
│   ├── bun.lock
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       ├── routes/
│       │   ├── prompt.tsx
│       │   └── score.tsx
│       ├── components/
│       │   ├── PromptInput.tsx
│       │   ├── ProgressBar.tsx
│       │   ├── ScoreCard.tsx
│       │   └── ModelResult.tsx
│       └── types/
│           └── index.ts
└── backend/
    ├── Cargo.toml
    └── src/
        ├── main.rs
        ├── handlers.rs
        ├── models.rs
        ├── score_engine.rs
        └── agents/
            ├── mod.rs
            ├── claude.rs
            ├── codex.rs
            ├── copilot.rs
            └── gemini.rs
```

## Agent Execution

Each AI agent runs as a subprocess CLI call with a structured prompt asking for:
1. Score (1-5) for the user's prompt
2. Specific recommendations for improvement
3. Analysis of missing elements

The prompt template sent to each agent:
```
Analyze the following prompt and provide:
1. A score from 1-5 based on effectiveness
2. Specific recommendations for improvement
3. What is missing or could be clearer

Prompt to analyze:
---
{user_prompt}
---

Respond in JSON format:
{"score": N, "recommendations": "..."}
```

## Frontend Routes

| Route | Component | Description |
|-------|-----------|-------------|
| / | PromptPage | Redirect to /prompt |
| /prompt | PromptPage | Prompt input with textarea |
| /score | ScorePage | Dashboard with progress bar, scores and results |

## Data Flow

1. User enters prompt in textarea on /prompt
2. Frontend tracks char/word count in real-time
3. User clicks "Analyze" button
4. Frontend sends POST /api/analyze with prompt
5. Backend starts SSE stream, sends "start" event
6. Backend calculates dimension scores, sends "scores" event
7. For each agent sequentially:
   - Send "agent_start" event
   - Execute agent subprocess
   - Send "agent_done" event with result
8. Send "complete" event with all results
9. Frontend updates UI progressively as events arrive

## Score Calculation

Backend uses heuristic analysis for initial dimension scores:
- Quality: Sentence structure, grammar indicators
- Stack Definitions: Presence of tech keywords
- Clear Goals: Action verbs, measurable outcomes
- Non-obvious Decisions: Edge case mentions, constraints
- Security & Operations: Security keywords, deployment mentions
- Overall: Weighted average of above dimensions

AI models provide independent scoring with qualitative feedback.

## Error Handling

- Agent timeout: 60 seconds per agent
- Failed agent: Returns null score with error message
- All agents fail: Returns dimension scores only with error flag
- Network error: Frontend displays retry option

## Dependencies

### Frontend (package.json)
- react: ^19.0.0
- react-dom: ^19.0.0
- @tanstack/react-router: latest
- tailwindcss: ^3.4.0
- typescript: ^5.7.0
- vite: ^6.0.0

### Backend (Cargo.toml)
- axum: 0.8
- tokio: 1.43 (full features)
- tokio-stream: 0.1
- futures: 0.3
- serde: 1.0 (derive)
- serde_json: 1.0
- tower-http: 0.6 (cors)
