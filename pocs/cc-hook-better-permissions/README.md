# cc-hook-better-permissions

A Claude Code PreToolUse hook that enforces a smarter permissions policy. Blocks credential access, asks before destructive or network operations, and auto-approves everything else.

## Permission Tiers

| Tier | What | Behavior |
|------|------|----------|
| BLOCK | SSH keys, `.env`, AWS/GCP/K8s credentials, GPG keys, tokens | Always denied |
| ASK | `rm`, `rmdir`, `shred` (file deletion) | Prompts user |
| ASK | `curl`, `wget`, `ssh`, `scp`, `nc` (network) | Prompts user |
| ALLOW | Read, Glob, Grep, Edit, Write (non-credential paths) | Auto-approved |

## Blocked Credential Paths

- `~/.ssh/id_*`, `~/.ssh/config`
- `~/.aws/credentials`, `~/.aws/config`
- `~/.gnupg/*`
- `~/.kube/config`
- `~/.docker/config.json`
- `~/.npmrc`, `~/.pypirc`, `~/.netrc`
- `~/.git-credentials`
- `~/.config/gh/hosts.yml`
- Any file named `.env`, `.env.*`, `credentials.json`, `secrets.yaml`, `token`, `password*`, `*secret*`

## How It Works

The hook registers as a PreToolUse handler matching all tools (`.*`). On every tool call:

1. Reads the tool invocation JSON from stdin
2. Checks tool name and params against the permission rules
3. Outputs a decision JSON: `allow`, `block`, or `ask`

For Bash commands, it parses the command string to detect credential paths, rm operations, and network commands. For file-based tools (Read, Write, Edit), it checks the file path. Paths are normalized and symlinks are resolved before matching.

## Install

```bash
chmod +x install.sh
./install.sh
```

This copies `permissions.py` to `~/.claude/hooks/` and registers the PreToolUse hook in `~/.claude/settings.json`. A backup of your existing settings is created automatically.

## Uninstall

```bash
chmod +x uninstall.sh
./uninstall.sh
```

Removes the hook file and cleans the entry from `settings.json`.

## Requirements

- Python 3.6+
- Claude Code CLI

## Testing

After installing, verify each tier in Claude Code:

```
cat ~/.ssh/id_rsa          → BLOCKED
rm -rf /tmp/test           → ASK
curl https://httpbin.org   → ASK
cat README.md              → ALLOWED (no prompt)
```
