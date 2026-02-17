# Claude Code Teams

https://code.claude.com/docs/en/agent-teams#orchestrate-teams-of-claude-code-sessions

## Prompt

```
Create a team with 4 teammates to build a twitter like app in parallel.
Backend must be rust (edition 2024 and version 1.93+)
Frontend should be react + tanstack
code is on the read (at least the initial projet)
create a run.sh and a stop.sh
```

### Progreess

```
 Let me first check what's already in the project directory.

⏺ Read 11 files (ctrl+o to expand)

⏺ Good, I have the full picture. Scaffold Rust backend (hello world + tokio) and default Vite React frontend. Let me set up the
  team now.

  Activating Team mode - 4 teammates building a Twitter-like app in parallel.

⏺ Now let me create the tasks for the team.

⏺ Now let me spawn all 4 teammates in parallel.

⏺ 4 agents launched (ctrl+o to expand)
   ├─ @rust-backend (oh-my-claudecode:executor)
   │  ⎿  Build Rust backend API (shift+↑ to manage)
   ├─ @react-ui (oh-my-claudecode:executor)
   │  ⎿  Build React UI components (shift+↑ to manage)
   ├─ @tanstack-data (oh-my-claudecode:executor)
   │  ⎿  Set up TanStack Query layer (shift+↑ to manage)
   └─ @scripts (oh-my-claudecode:executor)
      ⎿  Create scripts and vite config (shift+↑ to manage)

⏺ All 4 teammates are running in parallel:

  ┌───────────────┬──────────────────────────────────┬─────────┐
  │   Teammate    │               Task               │ Status  │
  ├───────────────┼──────────────────────────────────┼─────────┤
  │ rust-backend  │ Rust API with axum (port 3001)   │ Running │
  ├───────────────┼──────────────────────────────────┼─────────┤
  │ react-ui      │ React components (dark theme)    │ Running │
  ├───────────────┼──────────────────────────────────┼─────────┤
  │ tanstack-data │ TanStack Query hooks + API layer │ Running │
  ├───────────────┼──────────────────────────────────┼─────────┤
  │ scripts       │ run.sh, stop.sh, vite proxy      │ Running │
  └───────────────┴──────────────────────────────────┴─────────┘

  Waiting for teammates to complete their work...

⏺ Good, tasks #1 and #2 are in progress. #3 and #4 should pick up shortly. Let me wait a bit and check progress.

⏺ Bash(sleep 30 && echo "check complete")
  ⎿  check complete
  ⎿  (timeout 1m)

  4 tasks (0 done, 4 in progress, 0 open)
  ◼ Build Rust REST API backend with axum (@rust-backend)
    Editing src/main.rs…
  ◼ Build React UI components and pages (@react-ui)
    Writing frontend/src/components/TweetCompose.tsx…
  ◼ Set up TanStack Query and Router with API integration (@tanstack-data)
    Writing frontend/src/components/TweetFeed.tsx…
  ◼ Create run.sh, stop.sh, and vite proxy config (@scripts)
    Running ls -la /Users/diegopacheco/git/diegopacheco/ai-pl……
```

### Build

```bash
cargo build
```

### Run

```bash
cargo run
```

### Result

