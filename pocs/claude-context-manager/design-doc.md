# Claude Context Manager -- Design Document

## 1. Overview

Claude Context Manager is a Rust TUI application that provides full visibility and control over all Claude Code configuration artifacts. It scans global (`~/.claude/`), project-level (`.claude/`), and plugin directories, presenting a unified view of context files, MCP servers (including plugins), hooks, commands, subagents, and skills. Users can inspect, preview, remove, backup, and restore any configuration element through a visually rich terminal interface with a dashboard and bar charts. It also connects to a remote catalog repository to browse and install shared skills, commands, subagents, and MCP configurations. Catalog loading runs in a background thread so the UI stays responsive.

## 2. Problem Statement

Claude Code stores configuration across multiple files and directories at both global and project scope. There is no single tool to see everything at a glance, audit what is configured, safely remove entries, or create/restore backups. Managing these manually is error-prone and tedious.

## 3. Goals

- Provide a single TUI dashboard for all Claude Code configuration
- Visual dashboard with bar charts showing artifact counts and health summary
- Support full and selective backup/restore via `.tar.gz` snapshots
- Allow safe removal of any MCP, hook, command, subagent, or skill
- Preview any artifact content with Space key overlay
- Visual, color-coded health status for all entries
- Discover plugins/MCPs from `~/.claude/plugins/marketplaces/`
- Self-contained binary with zero runtime dependencies
- Browse and install skills, commands, subagents, and MCPs from a remote catalog repo
- Non-blocking catalog loading via background threads and mpsc channels

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
+-----------------------------------------------------------+
|                    TUI Layer (ratatui)                      |
|  +----------+------+-------+----------+---------+---------+|
|  | Context  | MCPs | Hooks | Commands | Agents  | Catalog ||
|  | Dashboard|      |       |          | /Skills | Backup  ||
|  +----------+------+-------+----------+---------+---------+|
|  |               Preview Overlay (Space)                   ||
|  +---------------------------------------------------------+|
+-----------------------------------------------------------+
|                   Core Logic Layer                          |
|  +----------+ +----------+ +-----------+ +-----------+     |
|  | Scanner  | | Remover  | |  Backup   | |  Restore  |     |
|  | Module   | | Module   | |  Module   | |  Module   |     |
|  +----------+ +----------+ +-----------+ +-----------+     |
|  +---------------------------------------------------------+|
|  |  Catalog Module (background thread + mpsc channels)     ||
|  +---------------------------------------------------------+|
+-----------------------------------------------------------+
|                  File System Layer                          |
|    ~/.claude/  +  .claude/ (project)  +  plugins/          |
+-----------------------------------------------------------+
```

## 7. Configuration Sources

### 7.1 Global Scope (`~/.claude/`)

| Path                                     | Contains                        |
|------------------------------------------|---------------------------------|
| `~/.claude/settings.json`                | MCPs, hooks, permissions, enabledPlugins |
| `~/.claude/CLAUDE.md`                    | Global instructions             |
| `~/.claude/commands/`                    | User-defined commands           |
| `~/.claude/agents/`                      | Custom subagents                |
| `~/.claude/skills/`                      | Global skill definitions        |
| `~/.claude/projects/`                    | Project-specific memory/config  |
| `~/.claude/plugins/marketplaces/`        | Installed plugins/MCPs          |

### 7.2 Project Scope (`.claude/` and project root)

| Path                          | Contains                        |
|-------------------------------|---------------------------------|
| `.claude/settings.json`       | Project MCPs, hooks             |
| `.claude/settings.local.json` | Local overrides                 |
| `.claude/commands/`           | Project commands                |
| `.claude/agents/`             | Project subagents               |
| `.claude/skills/`             | Project skills                  |
| `CLAUDE.md`                   | Project instructions            |
| `skills/`                     | Project skill definitions       |

## 8. TUI Layout

```
+--------------------------------------------------------------+
|  Claude Context Manager v0.1.0                               |
+--------------------------------------------------------------+
| [Context] [MCPs] [Hooks] [Commands] [Agents/Skills] [Catalog] [Backup] |
+--------------------------------------------------------------+
|                                                              |
|  +-- Dashboard -----------------------------------------+   |
|  |  Claude Context Manager   12 total artifacts          |   |
|  |                                                       |   |
|  |  MCPs           ████████████░░░░░░░░░░░░░░░░░░ 4     |   |
|  |  Hooks          ██████░░░░░░░░░░░░░░░░░░░░░░░░ 2     |   |
|  |  Commands       ████████████████░░░░░░░░░░░░░░ 5     |   |
|  |  Agents         ███░░░░░░░░░░░░░░░░░░░░░░░░░░░ 1     |   |
|  |  Skills         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0     |   |
|  |  Context Files  ██████░░░░░░░░░░░░░░░░░░░░░░░░ 2     |   |
|  |  Memory Files   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0     |   |
|  |                                                       |   |
|  |  Health  * 10 active  * 1 warning  o 1 broken  2 bkps |   |
|  +-------------------------------------------------------+   |
|                                                              |
|  +-- Context & Memory (4) -------------------------------+   |
|  |  > * CLAUDE.md (global)             [global]          |   |
|  |    * CLAUDE.md                      [project]         |   |
|  +-------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
| [Tab] switch [j/k] navigate [d] delete [b] backup [Space] preview [q] quit |
+--------------------------------------------------------------+
```

### 8.1 Color Scheme

| Color   | Meaning                                            |
|---------|----------------------------------------------------|
| Green   | Active, healthy, reachable                         |
| Yellow  | Warning -- config present but binary/path not found or plugin disabled |
| Red     | Broken -- parse error, missing file, invalid JSON  |
| Cyan    | Selected / highlighted item                        |
| White   | Normal text                                        |
| Dim     | Disabled or informational                          |
| Magenta | MCP bar chart color                                |
| Blue    | Hook bar chart color                               |

### 8.2 Tab Descriptions

**Context/Memory (Dashboard)** -- Shows a visual dashboard with horizontal bar charts counting all artifact types (MCPs, Hooks, Commands, Agents, Skills, Context Files, Memory Files). Below the charts is a health summary line showing active/warning/broken counts and backup count. Under the dashboard, lists `CLAUDE.md` files (global + project) and memory files from `~/.claude/projects/`.

**MCPs** -- Lists all MCP server entries from `settings.json` (global and project) plus plugins discovered from `~/.claude/plugins/marketplaces/`. Plugin enabled status is checked against the `enabledPlugins` field in `~/.claude/settings.json`. Shows name, command, scope (global/project), and health status.

**Hooks** -- Lists all hook definitions. Each hook displays as `EventName [MatcherPattern]: command-name.sh` (e.g., `PostToolUse [Edit]: eslint-hook.sh`). When no matcher is present, displays as `EventName: command-name.sh`. Shows scope and health status by checking if the command is executable.

**Commands** -- Lists all custom commands from `commands/` directories (global and project). Shows name, source file, scope. Recursively scans subdirectories.

**Agents/Skills** -- Lists subagents from `agents/` directories and skills from `skills/` directories. Skills are scanned from three locations: `~/.claude/skills/` (global), `.claude/skills/` (project), and `skills/` (project root). Only directories containing a `SKILL.md` file are recognized as skills.

**Catalog** -- Connects to the remote catalog repository (`github.com/diegopacheco/ai-playground`). On first access via `[l]` or `[i]`, clones the repo into a temp folder using a background thread with mpsc channels so the UI remains responsive. The `check_loaded()` method polls the channel via `try_recv()`. Lists all available skills, commands, subagents, and MCP configurations found in the repo. Already-installed items are marked with a green checkmark. Press `[i]` to install, then choose global or project scope.

**Backup/Restore** -- Lists existing backups in `~/.claude-backups/`. Shows creation timestamp and human-readable file size. Provides create backup, full restore, and selective restore actions.

## 9. Features

### 9.1 Scanner

- Walks global (`~/.claude/`), project (`.claude/`), and project root directories
- Parses `settings.json` and `settings.local.json` files for MCPs and hooks
- Scans `~/.claude/plugins/marketplaces/` for installed plugins by reading `.claude-plugin/plugin.json` (description, version) and checking for `.mcp.json`
- Checks plugin enabled status against `enabledPlugins` in `~/.claude/settings.json`
- Reads command and agent directories recursively for definitions
- Scans skills from `~/.claude/skills/`, `.claude/skills/`, and `skills/` -- only directories with `SKILL.md`
- Stores a 200-character preview in metadata for commands, agents, skills, and memory files
- Validates health: checks if referenced binaries/paths exist
- Returns a unified model of all discovered artifacts

### 9.2 Dashboard

- Context tab renders a visual dashboard as the top section
- Horizontal bar charts (30-char wide, filled/empty blocks) for each artifact type
- Bar colors: Magenta (MCPs), Blue (Hooks), Green (Commands), Yellow (Agents), Cyan (Skills), White (Context Files), DarkGray (Memory Files)
- Health summary line: green active count, yellow warning count, red broken count, backup count
- Below the dashboard, the Context and Memory file list is rendered

### 9.3 Preview

- Press `Space` to open a full-screen overlay showing the content of the selected item
- Works on all artifact tabs (Context, MCPs, Hooks, Commands, Agents/Skills) and Catalog
- For files: reads and displays file content (up to 100 lines)
- For directories (e.g., skills): lists directory contents
- For catalog items: reads the file from the cloned repo
- Press `Space` again to close the overlay

### 9.4 Remove

- User selects an item and presses `[d]`
- Confirmation dialog appears: "Remove 'name'? [y/N]"
- On confirm: modifies the appropriate `settings.json` or deletes the file
- For MCPs/hooks: removes the entry from JSON and writes back
- For commands/agents/skills: deletes the file from disk
- Selection adjusts after deletion

### 9.5 Backup

- Creates a `.tar.gz` archive of all Claude config
- Default destination: `~/.claude-backups/backup-YYYY-MM-DD-HHMMSS.tar.gz`
- Archive structure preserves relative paths from `~/.claude/` and `.claude/`
- Stores a manifest JSON inside the archive listing all included items

### 9.6 Catalog (Remote Install)

- Source repository: `https://github.com/diegopacheco/ai-playground`
- Loading runs in a background thread using `std::sync::mpsc` channels
- `start_load()` spawns a thread that runs `git clone --depth 1` into a temp directory
- The thread sends a `CatalogResult` (items + temp_dir or error) through the channel
- `check_loaded()` polls via `try_recv()` -- non-blocking, called from the event loop tick
- Scanner walks the cloned repo (max depth 6) looking for:
  - `skills/` directories containing `SKILL.md` files
  - `.md` files in `commands/` directories
  - `.md` files in `agents/` directories
  - `settings.json` files containing `mcpServers` entries
- Results are sorted by type then name, deduplicated
- Items already installed locally are marked with a green checkmark
- User selects an item and presses `[i]` to install
- Install scope dialog: `[g]` for global `[p]` for project
- Install actions per type:
  - **Skills**: copies the skill directory recursively into `skills/` (project) or `~/.claude/commands/` (global)
  - **Commands**: copies the `.md` file into `~/.claude/commands/` or `.claude/commands/`
  - **Agents**: copies the agent file into `~/.claude/agents/` or `.claude/agents/`
  - **MCPs**: extracts the MCP entry from source `settings.json` and merges it into target `settings.json`
- Temp directory is cleaned up on application exit (held by `TempDir`)

### 9.7 Restore

- **Full restore**: extracts entire archive, overwrites current config
- **Selective restore**: shows archive contents in a checklist with `[Space]` to toggle, `[Enter]` to restore
- Confirmation dialog before applying
- Creates an automatic backup of current state before restoring

### 9.8 Hook Format

Hooks are parsed from the `hooks` object in `settings.json`. Each hook entry contains an event name, optional matcher pattern, and one or more commands. The scanner produces a display name in the format:

- With matcher: `PostToolUse [Edit]: eslint-hook.sh`
- Without matcher: `PreToolUse: my-hook.sh`

The command path is shortened to just the filename for display. Full command path, event name, and matcher are stored in metadata.

### 9.9 Plugin/MCP Scanning

Plugins are discovered from `~/.claude/plugins/marketplaces/`. For each plugin directory:

1. Reads `.claude-plugin/plugin.json` for description and version
2. Checks if `.mcp.json` exists (warning if missing)
3. Checks `enabledPlugins` in `~/.claude/settings.json` using the key format `pluginName@marketplaceName`
4. Disabled plugins get a yellow "disabled" warning status
5. Plugins are listed as MCP artifacts with global scope

## 10. Project Structure

```
claude-context-manager/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── app.rs              (application state, event loop, preview toggle)
│   ├── ui.rs               (ratatui rendering, dashboard, tabs, preview overlay, dialogs)
│   ├── scanner.rs          (config discovery, plugin scanning, hook parsing)
│   ├── backup.rs           (create/list backups)
│   ├── restore.rs          (full and selective restore)
│   ├── remover.rs          (delete MCPs, hooks, commands)
│   ├── model.rs            (data types for all artifacts)
│   ├── health.rs           (status checks, color assignment)
│   └── catalog.rs          (background git clone, scan remote repo, install)
├── run.sh
├── release.sh
└── design-doc.md
```

## 11. Data Model

```
ArtifactKind = Mcp | Hook | Command | Agent | Skill | ContextFile | MemoryFile

Scope = Global | Project

Health = Active | Warning(reason) | Broken(reason)

Artifact
├── name: String
├── kind: ArtifactKind
├── scope: Scope
├── source_path: PathBuf
├── health: Health
└── metadata: HashMap<String, String>
    (keys: "command", "args", "event", "matcher", "preview", "size",
     "version", "marketplace")

CatalogItem
├── name: String
├── kind: ArtifactKind
├── description: String
├── repo_path: PathBuf
└── installed: bool

CatalogResult
├── items: Vec<CatalogItem>
├── temp_dir: Option<TempDir>
└── error: Option<String>

CatalogStatus = NotLoaded | Loading | Loaded | Error(String)

Catalog
├── items: Vec<CatalogItem>
├── temp_dir: Option<TempDir>
├── status: CatalogStatus
└── receiver: Option<mpsc::Receiver<CatalogResult>>

BackupEntry
├── path: PathBuf
├── created_at: String
└── size_bytes: u64

Tab = Context | Mcps | Hooks | Commands | Agents | Catalog | Backup

Dialog = ConfirmDelete(idx) | ConfirmBackup | ConfirmFullRestore(idx)
       | SelectiveRestore(idx) | InstallScope(idx)
```

## 12. Key Bindings

| Key         | Action                         |
|-------------|--------------------------------|
| `Tab`       | Next tab                       |
| `Shift+Tab` | Previous tab                   |
| `Up` / `k`  | Move selection up              |
| `Down` / `j` | Move selection down            |
| `Space`     | Preview selected item content  |
| `d`         | Delete selected item           |
| `b`         | Create backup                  |
| `r`         | Restore backup (Backup tab)    |
| `s`         | Selective restore (Backup tab) |
| `l`         | Load catalog from GitHub       |
| `i`         | Install from catalog           |
| `/`         | Search / filter current list   |
| `?`         | Show help overlay              |
| `q`         | Quit                           |

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
| `tempfile`  | Temp directory for catalog |

## 15. Risks and Mitigations

| Risk                                      | Mitigation                                        |
|-------------------------------------------|---------------------------------------------------|
| Corrupting settings.json on write         | Auto-backup before any mutation                   |
| Removing active MCP breaks Claude session | Confirmation dialog with clear warning            |
| Backup archive grows large                | Only config files (no caches/logs), show size      |
| Project-level paths vary                  | Auto-detect `.claude/` relative to CWD            |
| Rust edition 2024 ecosystem maturity      | Pin dependency versions, test on CI               |
| Catalog clone fails (no network/git)     | Show error in TUI, allow retry, background thread  |
| Catalog repo structure changes            | Scanner uses flexible pattern matching, not hardcoded paths |
| Plugin directory structure changes        | Graceful fallback if plugin.json or .mcp.json missing |

## 16. Future Considerations (Out of Scope for v1)

- Inline editing of settings values
- Diff view between backup and current state
- Multi-project dashboard (scan multiple repos)
- Export inventory as markdown report
- Watch mode for live config changes
