# MAS (Multi-Agent Station)

A Rust TUI application for managing multiple AI coding agents in a single terminal interface. Run Claude, Copilot, Gemini, and Codex side by side, switch between sessions instantly, and keep all your AI assistants organized in one place.

![MAS Screenshot](mas.png)

## Features

- **Multi-Agent Support**: Claude, GitHub Copilot, Gemini CLI, and OpenAI Codex
- **Session Management**: Create, switch, and kill agent sessions on the fly
- **PTY Integration**: Full terminal emulation for each agent
- **Session Persistence**: Automatically saves and restores your workspace layout
- **Keyboard-Driven**: Fast navigation with intuitive shortcuts

## Requirements

- Rust 1.85+ (2024 edition)
- One or more AI CLI agents installed:
  - `claude` - Anthropic Claude CLI
  - `copilot` - GitHub Copilot CLI
  - `gemini` - Google Gemini CLI
  - `codex` - OpenAI Codex CLI

## Build

```bash
cargo build --release
```

## Run

```bash
cargo run
```

Or after building:

```bash
./target/release/mas
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Cmd+T` | New session |
| `Cmd+E` | Focus terminal |
| `Cmd+1-9` | Switch to session 1-9 |
| `Cmd+Q` | Quit application |
| `Tab` | Toggle focus between panels |
| `↑/↓` | Navigate session list |
| `Enter` | Select session |
| `c` | Create new session (in list) |
| `q` | Kill selected session (in list) |
| `Space` | Confirm folder selection (in browser) |
| `Esc` | Cancel / go back |

## Session Storage

Sessions are persisted to `~/mas/sessions/layout.json` and automatically restored on startup.
