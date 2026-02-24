# LLM Judges MCP

MCP server that sends content to 4 LLM judges (Claude, Codex, Copilot, Gemini) via CLI and returns a consolidated verdict.

## Install

```bash
./install.sh
```

This will:
1. Build the Rust binary
2. Register the MCP in Claude Code (user scope)
3. Register the MCP in Codex CLI

## Tools

### judge
Send content to all 4 judges for evaluation.
```
"fact check this: Rust was created by Mozilla in 2010"
```

### judge_pick
Send content to selected judges.
```
"judge this using claude and gemini: the code is thread-safe"
```

### list_judges
List available judges.

## Requirements

- Rust 1.93+
- Claude CLI (`claude`)
- Codex CLI (`codex`)
- Copilot CLI (`copilot`)
- Gemini CLI (`gemini`)
