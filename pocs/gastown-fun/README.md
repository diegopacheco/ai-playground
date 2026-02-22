# GasTown

https://github.com/steveyegge/gastown

## Experience Notes

* npm instalation did not work
* I had to install via homebrew
* Instalation was a bit manual and had todo several steps.
* POC Repo: https://github.com/diegopacheco/gastown-poc
* I had issues with convoy to create a feature, doc mention flags that dont exist anymore.
* GasTown Mayor/tmux crashed doring claude code run in a middle of a healthchecker.
* GastTown is crashing and looping over:
```
⏺ Bash(cd ~/gt && gt hook status 2>&1)
  ⎿  Error: Exit code 1
     Error: listing hooked beads: bd list --json --status=hooked --assignee=mayor/ --limit=0: Warning: redirect target does not
     exist or is not a directory: /Users/.beads
     Error initializing --no-db mode: no .beads directory found (hint: run 'bd init' first or set BEADS_DIR)
     Usage:
       gt hook status [target] [flags]

     Flags:
       -h, --help   help for status
           --json   Output as JSON
```
* GT uses a lot of tokens indeed.
* Spent all my subscription tokens + 14 USD Dollars in less than 30min.
* All got stuck with conf issue and could not produce anything yet but sucked money.
* Had to re-install gastown even with the recommended is brew using go like this:
First update go to: `go version go1.26.0 darwin/arm64` them:
```
brew install icu4c
CGO_ENABLED=1 go install github.com/steveyegge/gastown/cmd/gt@latest

mkdir /tmp/beads
cd cd /tmp/beads/
git clone https://github.com/steveyegge/beads.git
CGO_CFLAGS="-I$(brew --prefix icu4c)/include" CGO_LDFLAGS="-L$(brew --prefix icu4c)/lib" CGO_ENABLED=1 go build -o "$HOME/.local/bin/bd" ./cmd/bd
```
* Claude Code with Opus 4.6 Comment on Gas Town:
```
3. Give up on gastown for now - it's v0.7.0, clearly still rough. The tool is eating tokens and time without producing results,
as you already noted.

My honest take: gastown is early-stage software with significant installation/configuration friction. You've already spent considerable time debugging infrastructure that should "just work." Unless you specifically need to evaluate gastown further, your time is better spent elsewhere.

⏺ Same problems even after a fresh install. The root issue is now clear:

Version mismatch: The Go-built gt (from source) calls bd init --backend but the Go-built bd (v0.55.4 from source) doesn't have a
--backend flag anymore. The two binaries are incompatible.

Also: No formulas found - the formula provisioning that was supposed to happen during gt install failed because beads init failed
  first. Without formulas, gt sling can't work.

This is a circular dependency:
- gt needs bd with --backend flag (old version)
- Old bd (Homebrew) doesn't have CGO/dolt support
- New bd (source) has CGO but removed --backend flag

Bottom line: Gastown v0.7.0 is not in a usable state for your setup. The Homebrew build lacks CGO, and building from source
creates version incompatibilities. This matches everything in your README notes - it eats tokens/money and can't produce results.

I'd recommend updating your README with these findings and moving on.
```

## Results

Gas Town Dashboard <br/>
<img src="gastown-dashboard.png" width="600" />

## Install

```
brew install gastown 
```

## Create HQ

```
❯ gt install ~/gt --git

🏭 Creating Gas Town HQ at /Users/diegopacheco/gt

   ✓ Created mayor/
   ✓ Created mayor/town.json
   ✓ Created mayor/rigs.json
   ✓ Created CLAUDE.md (town root identity anchor)
   ✓ Created mayor/.claude/settings.json
   ✓ Created deacon/.claude/settings.json
   ✓ Created plugins/
   ✓ Created mayor/daemon.json

   ✓ Created .gitignore
Initialized empty Git repository in /Users/diegopacheco/gt/.git/
   ✓ Initialized git repository
   ✓ Installed branch protection (auto-reverts non-main checkouts)
   ⚠ dolt not found in PATH — Dolt backend may not fully initialize
   ✓ Initialized .beads/ (town-level beads with hq- prefix)
   ✓ Provisioned 32 formulas
   ⚠ Could not create town-level agent beads: creating hq-mayor: bd create --json --id=hq-mayor --title=Mayor - global coordinator, handles cross-rig communication and escalations. --description=Mayor - global coordinator, handles cross-rig communication and escalations.

role_type: mayor
rig: null
agent_state: idle
hook_bead: null
cleanup_status: null
active_mr: null
notification_level: null --type=agent --labels=gt:agent: warning: beads.role not configured. Run 'bd init' to set.
Error: validation failed: invalid issue type: agent
   ✓ Detected overseer: diegopacheco <diego.pacheco.it@gmail.com> (via git-config)
   ✓ Created settings/escalation.json
   ✓ Created .claude/commands/ (slash commands for all agents)
   ✓ Synced 2 hook target(s)

✓ HQ created successfully!

Next steps:
  1. Add a rig: gt rig add <name> <git-url>
  2. (Optional) Configure agents: gt config agent list
  3. Enter the Mayor's office: gt mayor attach

Note: Dolt server is running (stop with gt dolt stop)
```

## Create a Rig and Crew

```
 gt rig add twitter_like https://github.com/diegopacheco/gastown-poc.git
Creating rig twitter_like...
  Repository: https://github.com/diegopacheco/gastown-poc.git
  Cloning repository (this may take a moment)...
   ✓ Created shared bare repo
  Creating mayor clone...
   ✓ Created mayor clone
  Initializing beads database...
   ✓ Initialized beads (prefix: tl)
  Creating refinery worktree...
   ✓ Created refinery worktree
  Warning: Could not create agent beads: creating tl-twitter_like-witness: bd create --json --id=tl-twitter_like-witness --title=Witness for twitter_like - monitors polecat health and progress. --description=Witness for twitter_like - monitors polecat health and progress.

role_type: witness                                                                                                                 rig: twitter_like                                                                                                                  agent_state: idle
hook_bead: null
cleanup_status: null
active_mr: null
notification_level: null --type=agent --labels=gt:agent --force: Error: failed to open rig "twitter_like" database: dolt backend requires CGO (not available on this build); use sqlite backend or install from pre-built binaries
  ! Could not create rig identity bead: bd create --json --id=tl-rig-twitter_like --title=twitter_like --description=Rig identity bead for twitter_like.

repo: https://github.com/diegopacheco/gastown-poc.git
prefix: tl
state: active --labels=gt:rig --force: Error: failed to open rig "twitter_like" database: dolt backend requires CGO (not available on this build); use sqlite backend or install from pre-built binaries
  Synced hooks for 4 target(s)

✓ Rig created in 7.6s

Structure:
  twitter_like/
  ├── config.json
  ├── .repo.git/        (shared bare repo for refinery+polecats)
  ├── .beads/           (prefix: tl)
  ├── plugins/          (rig-level plugins)
  ├── mayor/rig/        (clone: main)
  ├── refinery/rig/     (worktree: main, sees polecat branches)
  ├── crew/             (empty - add crew with 'gt crew add')
  ├── witness/
  └── polecats/         (.claude/ scaffolded for polecat sessions)

Next steps:
  gt crew add <name> --rig twitter_like   # Create your personal workspace
  cd /Users/diegopacheco/gt/twitter_like/crew/<name>              # Start working
❯ gt crew add diegopacheco --rig twitter_like
Creating crew workspace diegopacheco in twitter_like...
✓ Created crew workspace: twitter_like/diegopacheco
  Path: /Users/diegopacheco/gt/twitter_like/crew/diegopacheco
  Branch: main
⚠ Warning: could not create agent bead for diegopacheco: bd create --json --id=tl-twitter_like-crew-diegopacheco --title=Crew worker diegopacheco in twitter_like - human-managed persistent workspace. --description=Crew worker diegopacheco in twitter_like - human-managed persistent workspace.

role_type: crew
rig: twitter_like
agent_state: idle
hook_bead: null
cleanup_status: null
active_mr: null
notification_level: null --type=agent --labels=gt:agent --force: Error: failed to open rig "twitter_like" database: dolt backend requires CGO (not available on this build); use sqlite backend or install from pre-built binaries

✓ Created 1 crew workspace(s): [diegopacheco]

Start working with: cd /Users/diegopacheco/gt/twitter_like/crew/diegopacheco
```

## Attach to Mayor's office

```
cd twitter_like/crew/diegopacheco/
gt mayor attach
```

## Build a Feature

```
bd init
gt convoy create "build a twitter like application with registration, login, timeline, profile, search, follow, like, post images and limit chars 140. make sure there is a default admin user admin/admin and a run.sh to run the app. frontend must be react and backend must be rust, make sure react is with bun and vite and typescript use as much astanstack as possible. for backend use tokio and actixes, makes ure the frontend and backend are not monolithic, use sqllite for the db in rust." gt-abc12 gt-def34 --notify
```

```
WARNING: This binary was built with 'go build' directly.
         Use 'make build' to create a properly signed binary.
⚠ Warning: couldn't track gt-abc12: Error: resolving dependency ID gt-abc12: no issue found matching "gt-abc12"
⚠ Warning: couldn't track gt-def34: Error: resolving dependency ID gt-def34: no issue found matching "gt-def34"
✓ Created convoy 🚚 hq-cv-gezjo

  Name:     build a twitter like application with registration, login, timeline, profile, search, follow, like, post images and limit chars 140. make sure there is a default admin user admin/admin and a run.sh to run the app. frontend must be react and backend must be rust, make sure react is with bun and vite and typescript use as much astanstack as possible. for backend use tokio and actixes, makes ure the frontend and backend are not monolithic, use sqllite for the db in rust.
  Tracking: 0 issues
  Issues:   gt-abc12, gt-def34
  Owner:    overseer
  Notify:   mayor/

  Convoy auto-closes when all tracked issues complete
```

## Track Progress

```
gt convoy list
```

```
❯ gt convoy list
WARNING: This binary was built with 'go build' directly.
         Use 'make build' to create a properly signed binary.
Convoys

  1. 🚚 hq-cv-gezjo: build a twitter like application with registration, login, timeline, profile, search, follow, like, post images and limit chars 140. make sure there is a default admin user admin/admin and a run.sh to run the app. frontend must be react and backend must be rust, make sure react is with bun and vite and typescript use as much astanstack as possible. for backend use tokio and actixes, makes ure the frontend and backend are not monolithic, use sqllite for the db in rust. ●
  2. 🚚 hq-cv-7ueam: build a twitter like application with registration, login, timeline, profile, search, follow, like, post images and limit chars 140. make sure there is a default admin user admin/admin and a run.sh to run the app. frontend must be react and backend must be rust, make sure react is with bun and vite and typescript use as much astanstack as possible. for backend use tokio and actixes, makes ure the frontend and backend are not monolithic, use sqllite for the db in rust. ●

Use 'gt convoy status <id>' or 'gt convoy status <n>' for detailed view.
```

## Give work to agent

```
gt sling hq-cv-gezjo twitter_like
gt sling hq-cv-7ueam twitter_like
```



## Monitor Agents

```
gt agent list
```

```
❯ gt agent list

  🎩 Mayor
── twitter_like ──
  🏭 refinery
  🦉 witness
```

## Shutdown

```
❯ gt shutdown --all --yes --cleanup-orphans
Sessions to stop:
  → hq-boot
  → hq-deacon
  → hq-mayor
  → tl-refinery
  → tl-witness

Shutting down Gas Town...
  ✓ tl-refinery stopped
  ✓ tl-witness stopped
  ✓ hq-mayor stopped
  ✓ hq-boot stopped
  ✓ hq-deacon stopped

Cleaning up orphaned Claude processes...
  ○ No orphaned processes found

Cleaning up polecats...
  ○ No polecats to clean up

Stopping daemon...
  ○ Daemon not tracked by PID file
  ⚠ Found 1 orphaned daemon process(es): [26174]
  ✓ Killed 1 orphaned daemon(s)

Verifying shutdown...
  ✓ No orphaned Claude processes detected

✓ Gas Town shutdown complete (5 sessions stopped)
❯ gt status
Town: gt
/Users/diegopacheco/gt

👤 Overseer: diegopacheco <diego.pacheco.it@gmail.com>

🎩 mayor        ○
🐺 deacon       ○

─── twitter_like/ ───────────────────────────────────────────

🦉 witness      ○
🏭 refinery     ○
👷 Crew (1)
   diegopacheco ○
```