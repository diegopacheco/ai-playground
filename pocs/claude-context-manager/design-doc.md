# Claude Context Manager — Design Document

## 1. Overview

Claude Context Manager is a Rust TUI application that provides full visibility and control over all Claude Code configuration artifacts. It scans both global (`~/.claude/`) and project-level (`.claude/`) directories, presenting a unified view of context files, MCP servers, hooks, commands, subagents, and skills. Users can inspect, remove, backup, and restore any configuration element through a visually rich terminal interface. It also connects to a remote catalog repository to browse and install shared skills, commands, subagents, and MCP configurations.

## 2. Problem Statement

Claude Code stores configuration across multiple files and directories at both global and project scope. There is no single tool to see everything at a glance, audit what is configured, safely remove entries, or create/restore backups. Managing these manually is error-prone and tedious.

## 3. Goals

- Provide a single TUI dashboard for all Claude Code configuration
- Support full and selective backup/restore via `.tar.gz` snapshots
- Allow safe removal of any MCP, hook, command, subagent, or skill
- Visual, color-coded health status for all entries
- Self-contained binary with zero runtime dependencies
- Browse and install skills, commands, subagents, and MCPs from a remote catalog repo

## 4. Non-Goals

- Editing configuration values inline (out of scope for v1)
- Remote/cloud backup sync
- Auto-discovery of Claude Code installations outside `~/.claude`
- Any web or GUI interface

## 5. Tech Stack

| Component       | Choice                          |
|-----------------|---------------------------------|
| Language        | Rust 1.94+, edition 2024        |
| TUI framework   | `ratatui` + `crossterm` backend |
| Serialization   | `serde` + `serde_json`          |
| Archive         | `tar` + `flate2` crates         |
| File walking    | `walkdir`                       |
| Date/time       | `chrono`                        |

## 6. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TUI Layer (ratatui)                   │
│  ┌─────────┬──────┬───────┬──────────┬────────┬───────┐ │
│  │ Context │ MCPs │ Hooks │ Commands │ Agents │Backup │ │
│  │ /Memory │      │       │          │/Skills │Restore│ │
│  └─────────┴──────┴───────┴──────────┴────────┴───────┘ │
├─────────────────────────────────────────────────────────┤
│                   Core Logic Layer                       │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐ │
│  │ Scanner  │ │ Remover  │ │  Backup   │ │  Restore  │ │
│  │ Module   │ │ Module   │ │  Module   │ │  Module   │ │
│  └──────────┘ └──────────┘ └───────────┘ └───────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Catalog Module (git clone)              │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  File System Layer                       │
│         ~/.claude/  +  .claude/ (project)               │
└─────────────────────────────────────────────────────────┘
```

## 7. Configuration Sources

### 7.1 Global Scope (`~/.claude/`)

| Path                          | Contains                        |
|-------------------------------|---------------------------------|
| `~/.claude/settings.json`     | MCPs, hooks, permissions        |
| `~/.claude/CLAUDE.md`         | Global instructions             |
| `~/.claude/commands/`         | User-defined commands           |
| `~/.claude/agents/`           | Custom subagents                |
| `~/.claude/projects/`         | Project-specific memory/config  |

### 7.2 Project Scope (`.claude/`)

| Path                          | Contains                        |
|-------------------------------|---------------------------------|
| `.claude/settings.json`       | Project MCPs, hooks             |
| `.claude/settings.local.json` | Local overrides                 |
| `.claude/commands/`           | Project commands                |
| `.claude/agents/`             | Project subagents               |
| `CLAUDE.md`                   | Project instructions            |
| `skills/`                     | Skill definitions               |

## 8. TUI Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Claude Context Manager v0.1.0                    [Q]uit     │
├──────────────────────────────────────────────────────────────┤
│ [Context] [MCPs] [Hooks] [Commands] [Agents/Skills] [Catalog] [Backup]│
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ MCPs ──────────────────────────────────────────────────┐ │
│  │                                                         │ │
│  │  ● context-mode     ~/.claude/settings.json    [global] │ │
│  │  ● playwright       ~/.claude/settings.json    [global] │ │
│  │  ○ broken-mcp       .claude/settings.json    [project]  │ │
│  │                                                         │ │
│  │  ● = active (green)   ○ = broken (red)                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [↑↓] Navigate  [Tab] Switch tab  [D]elete  [B]ackup  [?]Help│
└──────────────────────────────────────────────────────────────┘
```

### 8.1 Color Scheme

| Color   | Meaning                                            |
|---------|----------------------------------------------------|
| Green   | Active, healthy, reachable                         |
| Yellow  | Warning — config present but binary/path not found |
| Red     | Broken — parse error, missing file, invalid JSON   |
| Cyan    | Selected / highlighted item                        |
| White   | Normal text                                        |
| Dim     | Disabled or informational                          |

### 8.2 Tab Descriptions

**Context/Memory** — Lists `CLAUDE.md` files (global + project), memory files from `~/.claude/projects/`, and their sizes. Shows a preview of the selected file content in a side panel.

**MCPs** — Lists all MCP server entries from `settings.json` (global and project). Shows name, command, scope (global/project), and health status by checking if the binary exists.

**Hooks** — Lists all hook definitions. Shows event name, command, scope, and whether the command is executable.

**Commands** — Lists all custom commands from `commands/` directories. Shows name, source file, scope.

**Agents/Skills** — Lists subagents from `agents/` directories and skills from `skills/`. Shows name, description (parsed from frontmatter or SKILL.md), scope.

**Catalog** — Connects to the remote catalog repository (`github.com/diegopacheco/ai-playground`). On first access, clones the repo into a temp folder. Lists all available skills, commands, subagents, and MCP configurations found in the repo. User can browse, preview, and install any item. Already-installed items are marked with a checkmark.

**Backup/Restore** — Lists existing backups in `~/.claude-backups/`. Provides create backup, full restore, and selective restore actions.

## 9. Features

### 9.1 Scanner

- Walks both global and project config directories
- Parses `settings.json` files for MCPs, hooks, permissions
- Reads command/agent/skill directories for definitions
- Validates health: checks if referenced binaries/paths exist
- Returns a unified model of all discovered artifacts

### 9.2 Remove

- User selects an item and presses `[D]`
- Confirmation dialog appears: "Remove MCP 'context-mode' from global settings? [y/N]"
- On confirm: modifies the appropriate `settings.json` or deletes the file
- For MCPs/hooks: removes the entry from JSON and writes back
- For commands/agents/skills: deletes the file from disk

### 9.3 Backup

- Creates a `.tar.gz` archive of all Claude config
- Default destination: `~/.claude-backups/backup-YYYY-MM-DD-HHMMSS.tar.gz`
- Archive structure preserves relative paths from `~/.claude/` and `.claude/`
- Stores a manifest JSON inside the archive listing all included items

### 9.4 Restore

- **Full restore**: extracts entire archive, overwrites current config
- **Selective restore**: shows archive contents in a checklist, user picks what to restore
- Confirmation dialog before applying
- Creates an automatic backup of current state before restoring

## 10. Project Structure

```
claude-context-manager/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── app.rs              (application state, event loop)
│   ├── ui.rs               (ratatui rendering, tabs, layout)
│   ├── scanner.rs          (config discovery and parsing)
│   ├── backup.rs           (create/list backups)
│   ├── restore.rs          (full and selective restore)
│   ├── remover.rs          (delete MCPs, hooks, commands)
│   ├── model.rs            (data types for all artifacts)
│   └── health.rs           (status checks, color assignment)
├── run.sh
├── release.sh
└── design-doc.md
```

## 11. Data Model

```
Artifact
├── name: String
├── kind: ArtifactKind (Mcp | Hook | Command | Agent | Skill | ContextFile | MemoryFile)
├── scope: Scope (Global | Project)
├── source_path: PathBuf
├── health: Health (Active | Warning(reason) | Broken(reason))
└── metadata: HashMap<String, String>

Backup
├── path: PathBuf
├── created_at: DateTime
├── size_bytes: u64
└── manifest: Vec<Artifact>
```

## 12. Key Bindings

| Key         | Action                         |
|-------------|--------------------------------|
| `Tab`       | Next tab                       |
| `Shift+Tab` | Previous tab                   |
| `↑` / `k`  | Move selection up              |
| `↓` / `j`  | Move selection down            |
| `Enter`     | View details / expand          |
| `d`         | Delete selected item           |
| `b`         | Create backup                  |
| `r`         | Restore backup                 |
| `s`         | Selective restore              |
| `/`         | Search / filter current list   |
| `q`         | Quit                           |
| `?`         | Show help overlay              |

## 13. Scripts

### 13.1 run.sh

Runs `cargo run --release` from the project directory.

### 13.2 release.sh

Builds a statically linked release binary via `cargo build --release` and copies it to `./claude-context-manager` (self-contained, no runtime dependencies).

## 14. Dependencies (Cargo.toml)

| Crate       | Purpose                    |
|-------------|----------------------------|
| `ratatui`   | TUI rendering              |
| `crossterm` | Terminal backend           |
| `serde`     | Serialization traits       |
| `serde_json`| JSON parsing/writing       |
| `tar`       | Archive creation/extraction|
| `flate2`    | Gzip compression           |
| `walkdir`   | Directory traversal        |
| `chrono`    | Timestamps for backups     |
| `dirs`      | Home directory resolution  |

## 15. Risks and Mitigations

| Risk                                      | Mitigation                                        |
|-------------------------------------------|---------------------------------------------------|
| Corrupting settings.json on write         | Auto-backup before any mutation                   |
| Removing active MCP breaks Claude session | Confirmation dialog with clear warning            |
| Backup archive grows large                | Only config files (no caches/logs), show size      |
| Project-level paths vary                  | Auto-detect `.claude/` relative to CWD            |
| Rust edition 2024 ecosystem maturity      | Pin dependency versions, test on CI               |

## 16. Future Considerations (Out of Scope for v1)

- Inline editing of settings values
- Diff view between backup and current state
- Multi-project dashboard (scan multiple repos)
- Export inventory as markdown report
- Watch mode for live config changes
