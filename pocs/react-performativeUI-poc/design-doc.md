# Design Doc — Synthetica (React 19 + performative-ui)

## 1. Goal

Build a single-page React 19 web app that showcases ten components from the
[`performative-ui`](https://vorpus.github.io/performativeUI/) library, wired to a
real OpenAI backend. The page reads as a parody AI-startup landing page; the
prompt inputs actually call the OpenAI API and stream the answer back into the UI.

## 2. Constraints

- React 19, built with Vite.
- Minimum dependencies: only `react`, `react-dom`, and `performative-ui` at runtime.
- The OpenAI key is provided by the operator as an environment variable
  (`OPENAI_API_KEY`) in the shell, never committed and never shipped to the browser.
- `start.sh` / `stop.sh` to run and stop the app.

## 3. Components used

The brief required these ten components. Each one has a real job on the page:

| Library export        | Doc route             | Where it appears on the page                          |
| --------------------- | --------------------- | ----------------------------------------------------- |
| `Sparkle`             | `/sparkle`            | Brand mark, section titles, eyebrow                   |
| `StatusDot`           | `/status-dot`         | Nav "all systems operational", hero eyebrow           |
| `Prompt`              | `/prompt`             | Playground input (textarea + model dropdown + send)   |
| `PromptHero`          | `/prompt-hero`        | Hero call-to-action input                             |
| `AsciiHero`           | `/ascii-hero`         | Cursor-reactive ASCII field behind the hero (`bare`)  |
| `NodeGraphBackground` | `/node-graph`         | Drifting node graph behind the "platform" section     |
| `MockIDE`             | `/mock-ide`           | "Watch it think" panel typing out `agent.ts`          |
| `ChatBubble`          | `/chat-bubble`        | User and AI turns in the playground conversation      |
| `TokenStream`         | `/token-stream`       | Reveals the AI answer token by token inside the bubble|
| `LogoMarquee`         | `/logo-marquee`       | Infinite "trusted by" logo wall                       |

## 4. Architecture

```
Browser (React 19) ─▶ Prompt/PromptHero ─▶ POST /api/chat ─▶ Vite middleware ─▶ OpenAI API
                                                              (holds OPENAI_API_KEY)
        ▲                                                                          │
        └──────────────── ChatBubble + TokenStream ◀── streamed answer ◀───────────┘
```

The diagram is rendered hand-drawn in `printscreens/architecture.png` and embedded
in the README.

### 4.1 Why a server-side proxy

Calling OpenAI directly from the browser would expose the API key to anyone who
opens the network tab. Instead the app posts to its own `/api/chat` endpoint. That
endpoint is a small Vite dev-server middleware (`openaiProxy` in `vite.config.js`)
that runs in Node, reads `process.env.OPENAI_API_KEY`, forwards the request to
`https://api.openai.com/v1/chat/completions`, and returns only the completion text.
The key stays on the server. This keeps the runtime dependency count at zero extra
libraries — the proxy uses Node's built-in `fetch`.

### 4.2 Request flow

1. User types into `PromptHero` (hero) or `Prompt` (playground) and submits.
2. `App.submit()` appends a user `ChatBubble`, then calls `ask()` from `src/api.js`.
3. `ask()` POSTs `{ prompt, model }` to `/api/chat`.
4. The middleware calls OpenAI with a small system prompt and returns `{ text }`.
5. `App` appends an AI `ChatBubble` whose child is a `TokenStream`, which reveals
   the answer token by token.

### 4.3 Model selection

The `Prompt` toolbar dropdown lists real OpenAI model ids
(`gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`). The chosen id is passed straight through
to the API. The hero `PromptHero` uses the default (`gpt-4o-mini`).

## 5. Error handling

- No key set: the middleware returns HTTP 400 with a clear message, surfaced in the
  UI as a red error line. The rest of the page still renders fully.
- OpenAI error: the upstream status and message are passed back to the client.
- Network/parse failure: caught in `ask()` and shown as an error line.

## 6. File layout

```
react-performativeUI-poc/
├── index.html            entry, fonts, sparkle favicon
├── vite.config.js        react plugin + openaiProxy middleware
├── package.json          react 19, performative-ui, vite
├── src/
│   ├── main.jsx          mounts <App>, imports performative-ui/styles.css
│   ├── App.jsx           the page; uses all ten components
│   ├── api.js            fetch wrapper for /api/chat
│   └── styles.css        layout, reuses performative-ui CSS variables
├── start.sh / stop.sh    run / stop the dev server
├── test.sh               smoke-check page + proxy
└── printscreens/         architecture diagram + UI screenshots
```

## 7. Trade-offs and decisions

- **Dev-server middleware instead of a standalone backend.** A separate Express/Fastify
  server would add a dependency and a second process. The Vite middleware reuses the
  one process already running and ships no extra packages. The cost: the proxy lives in
  the dev server, so production hosting would need an equivalent serverless function.
  For a POC driven by `start.sh`, this is the simplest correct choice.
- **Non-streaming upstream, streamed UI.** The middleware waits for the full OpenAI
  response, then the client fake-streams it with `TokenStream`. This keeps the proxy
  trivial while still using the `TokenStream` component for its intended effect. True
  SSE streaming would be a later refinement.
- **Dark theme.** `performative-ui` ships a dark default; the layout reuses its
  `--pui-*` variables so the custom chrome matches the components exactly.

## 8. Out of scope

- Multi-turn memory (each prompt is independent).
- Real server-sent-events token streaming from OpenAI.
- Production deployment / serverless function for `/api/chat`.
- Authentication and rate limiting.
