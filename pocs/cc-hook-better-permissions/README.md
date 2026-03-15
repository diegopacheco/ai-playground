# cc-hook-better-permissions

A Claude Code hook-based permissions system that blocks credential access, asks before destructive or network operations (and remembers your answer), and auto-approves everything else.

## Permission Tiers

| Tier | What | Behavior |
|------|------|----------|
| BLOCK | SSH keys, `.env`, AWS/GCP/K8s credentials, GPG keys, tokens | Always denied, no override |
| ASK | `rm`, `rmdir`, `shred` (file deletion) | Prompts user once, then remembers |
| ASK | `curl`, `wget`, `ssh`, `scp`, `nc` (network) | Prompts user once, then remembers |
| ALLOW | Read, Glob, Grep, Edit, Write (non-credential paths) | Auto-approved silently |

## Remember My Answer

When you approve a command category (network or rm), the PostToolUse hook saves your decision to `~/.claude/hooks/permissions_approved.json`. Next time a command in that category runs, it auto-approves without asking again.

To reset approvals, delete the file:
```bash
rm ~/.claude/hooks/permissions_approved.json
```

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

Two hooks work together:

- **PreToolUse** (`permissions.py`) — intercepts every tool call, checks against permission rules, returns `allow`, `deny`, or `ask`. Checks the approved file to skip asking for previously approved categories.
- **PostToolUse** (`permissions_post.py`) — runs after Bash commands succeed. If the command was a network or rm command (meaning the user approved it), saves the category to the approved file.

Paths are normalized and symlinks are resolved before matching.

## Install

```bash
chmod +x install.sh
./install.sh
```

The installer:
- Copies `permissions.py` and `permissions_post.py` to `~/.claude/hooks/`
- Registers PreToolUse and PostToolUse hooks in `~/.claude/settings.json`
- Patches `approve-variants.py` to remove `curl` from its safe list (if present)
- Patches `context-mode` plugin to remove its curl/wget/WebFetch blocking (if present)
- Creates `.bak` backups before modifying any file

## Uninstall

```bash
chmod +x uninstall.sh
./uninstall.sh
```

Removes hook files, cleans entries from `settings.json`, restores `context-mode` and `approve-variants.py` from backups, and deletes the approved file.

## File Tree

```
cc-hook-better-permissions/
├── design-doc.md
├── permissions.py          # PreToolUse hook (allow/deny/ask)
├── permissions_post.py     # PostToolUse hook (remember approvals)
├── install.sh
├── uninstall.sh
├── test.sh
└── README.md
```

## Proof It Works

First session — curl asked once, then remembered:
```
❯ curl google.com

⏺ Bash(curl -s google.com)
  ⎿  <HTML><HEAD><meta http-equiv="content-type" content="text/html;charset=utf-8">
     <TITLE>301 Moved</TITLE></HEAD><BODY>
     <H1>301 Moved</H1>

⏺ Google returns a 301 redirect to http://www.google.com/.

❯ curl google.com

⏺ Bash(curl -s google.com)
  ⎿  <HTML><HEAD><meta http-equiv="content-type" content="text/html;charset=utf-8">
     <TITLE>301 Moved</TITLE></HEAD><BODY>
     <H1>301 Moved</H1>

⏺ Same 301 redirect response. Google redirects to www.google.com.
```

New session — still remembered, no prompt:
```
❯ claude

❯ curl google.com

⏺ Bash(curl -s google.com)
  ⎿  <HTML><HEAD><meta http-equiv="content-type" content="text/html;charset=utf-8">
     <TITLE>301 Moved</TITLE></HEAD><BODY>
     <H1>301 Moved</H1>

⏺ Google returns a 301 redirect to http://www.google.com/.
```

## Requirements

- Python 3.6+
- Claude Code CLI

## Testing

```bash
./test.sh
```

Or manually in Claude Code after installing:

```
cat ~/.ssh/id_rsa          -> BLOCKED
rm -rf /tmp/test           -> ASK (first time), ALLOWED (after)
curl https://httpbin.org   -> ASK (first time), ALLOWED (after)
cat README.md              -> ALLOWED (no prompt)
```
