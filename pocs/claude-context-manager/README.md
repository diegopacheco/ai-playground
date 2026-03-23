# Claude Context Manager

A Rust TUI application that gives you full visibility and control over all Claude Code configuration.
Scan, inspect, remove, backup, restore, and install Claude artifacts from a single terminal dashboard.

## Features

- **7 tabs**: Context/Memory, MCPs, Hooks, Commands, Agents/Skills, Catalog, Backup/Restore
- **Health checks**: color-coded status (green = active, yellow = warning, red = broken)
- **Remove**: delete any MCP, hook, command, agent, or skill with confirmation
- **Backup**: create `.tar.gz` snapshots of all Claude config to `~/.claude-backups/`
- **Restore**: full or selective restore from any backup (auto-backup before overwrite)
- **Catalog**: browse and install skills, commands, agents, and MCPs from a remote GitHub repo
- **Search**: filter any list with `/` search
- **Vim-style navigation**: `j/k` or arrow keys

## Scanned Locations

| Scope   | Path                          |
|---------|-------------------------------|
| Global  | `~/.claude/settings.json`     |
| Global  | `~/.claude/CLAUDE.md`         |
| Global  | `~/.claude/commands/`         |
| Global  | `~/.claude/agents/`           |
| Global  | `~/.claude/projects/` (memory)|
| Project | `.claude/settings.json`       |
| Project | `.claude/settings.local.json` |
| Project | `.claude/commands/`           |
| Project | `.claude/agents/`             |
| Project | `CLAUDE.md`                   |
| Project | `skills/`                     |

## Key Bindings

| Key         | Action                              |
|-------------|-------------------------------------|
| `Tab`       | Next tab                            |
| `Shift+Tab` | Previous tab                        |
| `j` / `k`  | Navigate down / up                  |
| `d`         | Delete selected item                |
| `b`         | Create backup                       |
| `r`         | Full restore (Backup tab)           |
| `s`         | Selective restore (Backup tab)      |
| `l`         | Load catalog from GitHub            |
| `i`         | Install from catalog                |
| `/`         | Search / filter current list        |
| `?`         | Toggle help overlay                 |
| `q`         | Quit                                |

## Catalog

The Catalog tab connects to `github.com/diegopacheco/ai-playground` and scans for:
- Skills (`SKILL.md` directories)
- Commands (`.md` files in `commands/` folders)
- Agents (`.md` files in `agents/` folders)
- MCP configs (`mcpServers` in `settings.json` files)

Press `[l]` to load, then `[i]` to install. Choose global (`~/.claude/`) or project (`.claude/`) scope.

## Build

```
Rust 1.94+
Edition 2024
```

## Run

```
./run.sh
```

## Release (self-contained binary)

```
./release.sh
./claude-context-manager
```

## Tech Stack

| Component    | Choice                |
|--------------|-----------------------|
| Language     | Rust 1.94+            |
| TUI          | ratatui + crossterm   |
| Serialization| serde + serde_json    |
| Archive      | tar + flate2          |
| File walking | walkdir               |
| Date/time    | chrono                |

## Project Structure

```
src/
  main.rs       - entry point, terminal setup
  app.rs        - application state, event loop
  ui.rs         - ratatui rendering, tabs, dialogs
  scanner.rs    - config discovery and parsing
  backup.rs     - create/list backups
  restore.rs    - full and selective restore
  remover.rs    - delete MCPs, hooks, commands
  catalog.rs    - git clone, scan remote repo, install
  model.rs      - data types
  health.rs     - status checks
```
