# Claude Context Manager

A Rust TUI application that gives you full visibility and control over all Claude Code configuration.
Scan, inspect, preview, remove, backup, restore, and install Claude artifacts from a single terminal dashboard.

## Features

- **7 tabs**: Context/Memory (with dashboard), MCPs, Hooks, Commands, Agents/Skills, Catalog, Backup/Restore
- **Dashboard**: bar charts showing counts per artifact type, health summary (active/warning/broken), backup count
- **Preview**: press `Space` to view file contents of any item in a full-screen overlay
- **Health checks**: color-coded status (green = active, yellow = warning, red = broken)
- **Plugin/MCP scanning**: discovers plugins from `~/.claude/plugins/marketplaces/`, reads `.claude-plugin/plugin.json` and `.mcp.json`, checks enabled status via `enabledPlugins` in settings.json
- **Hook display**: shows event name, matcher pattern, and command name (e.g., `PostToolUse [Edit]: eslint-hook.sh`)
- **Skills scanning**: discovers skills from `~/.claude/skills/` (global), `.claude/skills/` (project), and `skills/` (project root)
- **Remove**: delete any MCP, hook, command, agent, or skill with confirmation
- **Backup**: create `.tar.gz` snapshots of all Claude config to `~/.claude-backups/`
- **Restore**: full or selective restore from any backup (auto-backup before overwrite)
- **Catalog**: browse and install skills, commands, agents, and MCPs from a remote GitHub repo
- **Background catalog loading**: git clone runs in a background thread via mpsc channels, UI stays responsive
- **Search**: filter any list with `/` search
- **Vim-style navigation**: `j/k` or arrow keys

## Scanned Locations

| Scope   | Path                                     |
|---------|------------------------------------------|
| Global  | `~/.claude/settings.json`                |
| Global  | `~/.claude/CLAUDE.md`                    |
| Global  | `~/.claude/commands/`                    |
| Global  | `~/.claude/agents/`                      |
| Global  | `~/.claude/skills/`                      |
| Global  | `~/.claude/projects/` (memory)           |
| Global  | `~/.claude/plugins/marketplaces/` (MCPs) |
| Project | `.claude/settings.json`                  |
| Project | `.claude/settings.local.json`            |
| Project | `.claude/commands/`                      |
| Project | `.claude/agents/`                        |
| Project | `CLAUDE.md`                              |
| Project | `skills/`                                |

## Key Bindings

| Key         | Action                              |
|-------------|-------------------------------------|
| `Tab`       | Next tab                            |
| `Shift+Tab` | Previous tab                        |
| `j` / `k`  | Navigate down / up                  |
| `Space`     | Preview selected item content       |
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

Press `[l]` to load (runs git clone in background thread), then `[i]` to install. Choose `[g]` for global (`~/.claude/`) or `[p]` for project (`.claude/`) scope. Already-installed items are marked with a checkmark.

## Plugin/MCP Discovery

Plugins are discovered from `~/.claude/plugins/marketplaces/`. For each plugin:
- Reads `.claude-plugin/plugin.json` for description and version
- Checks for `.mcp.json` presence
- Checks `enabledPlugins` in `~/.claude/settings.json` using key format `pluginName@marketplaceName`
- Disabled plugins show a yellow warning status

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
  app.rs        - application state, event loop, preview toggle
  ui.rs         - ratatui rendering, dashboard, tabs, preview overlay, dialogs
  scanner.rs    - config discovery, plugin scanning, hook parsing
  backup.rs     - create/list backups
  restore.rs    - full and selective restore
  remover.rs    - delete MCPs, hooks, commands
  catalog.rs    - background git clone, scan remote repo, install
  model.rs      - data types
  health.rs     - status checks
```
