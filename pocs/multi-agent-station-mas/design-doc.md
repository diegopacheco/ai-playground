# MAS (Multi-Agent Station) Design Document

## Overview

MAS (Multi-Agent Station) is a Rust TUI application that manages multiple AI agent sessions (Claude, Copilot, Gemini, Codex) in a single terminal interface. Users can spawn, switch between, and manage multiple agent terminals simultaneously.

## Technology Stack

| Component | Choice |
|-----------|--------|
| Language | Rust 2024 Edition (v1.93) |
| TUI Framework | ratatui + crossterm |
| Async Runtime | tokio |
| Session Persistence | JSON files in ~/multi-claude/sessions/ |
| Process Management | std::process / tokio::process |

## UI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAS    14:32:05                                │  <- Header
├───────────────┬─────────────────────────────────────────────────────────────┤
│  [+] New      │                                                             │
│               │                                                             │
│  ┌─────────┐  │                                                             │
│  │ Claude  │◀ │                    Agent Terminal Output                    │  <- Center
│  │ PID:123 │  │                    (selected session)                       │
│  └─────────┘  │                                                             │
│  ┌─────────┐  │                                                             │
│  │ Copilot │  │                                                             │
│  │ PID:456 │  │                                                             │
│  └─────────┘  │                                                             │
│  ┌─────────┐  │                                                             │
│  │ Gemini  │  │                                                             │
│  │ PID:789 │  │                                                             │
│  └─────────┘  │                                                             │
│               │                                                             │
│   Left Panel  │                                                             │
├───────────────┴─────────────────────────────────────────────────────────────┤
│  Sessions: 3                                                                │  <- Footer
└─────────────────────────────────────────────────────────────────────────────┘
```

### UI Areas

1. **Header (Top)**: Application title "MAS" centered with a live clock (HH:MM:SS)
2. **Left Panel**: Session list with search bar and [+] button to spawn new sessions, displays session name and PID
3. **Center Panel**: PTY terminal output of the currently selected agent session
4. **Footer**: Total number of active sessions

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+W | Quit application (kill all sessions) |
| Ctrl+T | Open new session dialog |
| Ctrl+E | Toggle fullscreen for the focused panel |
| Tab | Cycle focus between panels (left ↔ center) |
| Arrow Up/Down | Navigate session list (when left panel focused) |
| Enter | Select session from list |
| q | Kill selected session (when left panel focused) |
| r | Rename selected session (when left panel focused) |
| Esc | Exit input mode / close dialogs |
| Mouse Click | Focus clicked panel |

## New Session Dialog

When user presses [+] or Cmd+T:

```
┌─────────────────────────────────────┐
│         New Agent Session           │
├─────────────────────────────────────┤
│  Agent Type:                        │
│  ┌─────────────────────────────┐    │
│  │ ▼ Claude                    │    │
│  │   Copilot                   │    │
│  │   Gemini                    │    │
│  │   Codex                     │    │
│  └─────────────────────────────┘    │
│                                     │
│  Working Directory:                 │
│  ┌─────────────────────────────┐    │
│  │ ~/projects/my-app           │    │
│  │ [Browse...]                 │    │
│  └─────────────────────────────┘    │
│                                     │
│  [Cancel]              [Create]     │
└─────────────────────────────────────┘
```

### File Explorer

Simple tree-based directory browser:
- Arrow keys to navigate
- Enter to expand/collapse directories or select
- Backspace to go up one level
- Tab to confirm selection

## Agent Spawning

Each agent is spawned as a subprocess with PTY (pseudo-terminal) for full terminal emulation.

### Agent Commands (No YOLO Mode)

| Agent | Command |
|-------|---------|
| Claude | `claude --model opus` |
| Copilot | `copilot --model claude-sonnet-4` |
| Gemini | `gemini` |
| Codex | `codex --model gpt-5.2` |

Note: No auto-accept flags (--dangerously-skip-permissions, --full-auto, -y, --allow-all) since user will interact directly with each agent.

### Agent Spawning Logic

```
spawn_agent(agent_type, working_dir):
    1. Create PTY master/slave pair
    2. Fork process
    3. In child: set working directory, exec agent command
    4. In parent: store PID, PTY fd, create Session object
    5. Add to session manager
    6. Update UI
```

## Session Management

### Session Structure

```rust
struct Session {
    id: Uuid,
    agent_type: AgentType,      // Claude, Copilot, Gemini, Codex
    pid: u32,
    pty_fd: RawFd,
    working_dir: PathBuf,
    created_at: DateTime<Utc>,
    buffer: TerminalBuffer,     // scrollback buffer
}

enum AgentType {
    Claude,
    Copilot,
    Gemini,
    Codex,
}
```

### Session Persistence

Sessions saved to `~/mas/sessions/layout.json`:

```json
{
    "version": 1,
    "last_saved": "2026-02-03T02:15:00Z",
    "sessions": [
        {
            "id": "uuid-1",
            "agent_type": "claude",
            "working_dir": "/home/user/projects/app1"
        },
        {
            "id": "uuid-2", 
            "agent_type": "copilot",
            "working_dir": "/home/user/projects/app2"
        }
    ],
    "active_session_index": 0
}
```

### Startup Flow

```
1. Check if ~/mas/sessions/layout.json exists
2. If exists:
   - Show dialog: "Continue previous session? [Y/n]"
   - If Y: restore sessions from layout
   - If N: start fresh (empty state)
3. If not exists: start fresh
```

## Module Structure

```
src/
├── main.rs                 # Entry point, tokio runtime
├── app.rs                  # Application state machine
├── ui/
│   ├── mod.rs
│   ├── header.rs           # Title + clock widget
│   ├── session_list.rs     # Left panel widget
│   ├── terminal_view.rs    # Center panel (PTY output)
│   ├── footer.rs           # Session count widget
│   ├── dialog.rs           # New session dialog
│   └── file_browser.rs     # Directory picker
├── session/
│   ├── mod.rs
│   ├── manager.rs          # Session lifecycle
│   ├── session.rs          # Session struct
│   └── persistence.rs      # Save/load layout
├── agent/
│   ├── mod.rs
│   ├── spawner.rs          # PTY + process spawn
│   ├── claude.rs           # Claude command builder
│   ├── copilot.rs          # Copilot command builder
│   ├── gemini.rs           # Gemini command builder
│   └── codex.rs            # Codex command builder
├── pty/
│   ├── mod.rs
│   └── handler.rs          # PTY I/O handling
└── input/
    ├── mod.rs
    └── handler.rs          # Keyboard input routing
```

## Dependencies (Cargo.toml)

```toml
[package]
name = "multi-claude"
version = "0.1.0"
edition = "2024"
rust-version = "1.93"

[dependencies]
ratatui = "0.29"
crossterm = "0.28"
tokio = { version = "1.43", features = ["full"] }
portable-pty = "0.8"
uuid = { version = "1.11", features = ["v4"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
dirs = "6.0"
```

## Event Loop

```
loop {
    1. Poll for keyboard events (non-blocking)
    2. Poll for PTY output from all sessions (non-blocking)
    3. Update clock (every second)
    4. Handle any pending events:
       - Keyboard: route to appropriate handler
       - PTY output: append to session buffer
       - Session exit: remove from manager, update UI
    5. Render UI
    6. Sleep briefly to prevent CPU spinning
}
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Agent binary not found | Show error in terminal view, keep session in error state |
| Agent crashes | Mark session as "Exited (code X)", keep in list with option to restart |
| PTY read error | Log error, attempt reconnect once, then mark as disconnected |
| Session file corrupt | Log warning, start fresh |
| Permission denied on working dir | Show error in dialog, don't create session |

## Future Enhancements (Out of Scope for V1)

- Session renaming
- Session reordering (drag & drop)
- Split view (multiple terminals visible)
- Search within terminal buffer
- Copy/paste between sessions
- Session groups/tabs
- Custom agent configurations
- Themes/color schemes
