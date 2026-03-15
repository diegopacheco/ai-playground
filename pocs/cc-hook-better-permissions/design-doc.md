# Design Doc: Claude Code Hook-Based Permissions System

## Problem

Claude Code's default permission model is too noisy for safe operations (constant approval prompts for reads) and too permissive for dangerous ones (no special handling for credential files or destructive commands). We need a smarter, context-aware permission layer.

## Solution

A **PreToolUse hook** written in Python that intercepts every tool call before execution and applies a tiered permission policy:

| Tier | Action | Decision |
|------|--------|----------|
| **BLOCK** | Read/access credential files (SSH keys, `.env`, tokens, passwords) | Always blocked, no override |
| **ASK** | `rm` commands (file deletion) | Ask user permission |
| **ASK** | `curl`/`wget`/network commands | Ask user permission |
| **ALLOW** | Read-only tools (Read, Glob, Grep, Grep) | Auto-approve silently |
| **ALLOW** | Edit, Write (non-credential paths) | Auto-approve silently |
| **ALLOW** | Everything else not matching above rules | Auto-approve silently |

## Architecture

```
~/.claude/
├── settings.json          <-- hook registration (PreToolUse)
└── hooks/
    └── permissions.py     <-- the hook script
```

### Hook Flow

```
Tool Invocation
      │
      ▼
┌─────────────┐
│ PreToolUse   │
│ permissions  │
│    .py       │
└─────┬───────┘
      │
      ▼
┌─────────────────────┐
│ Read stdin JSON      │
│ {tool, params, ...}  │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐     YES    ┌────────────────────┐
│ Is credential path? ├───────────►│ BLOCK + reason     │
└─────┬───────────────┘            └────────────────────┘
      │ NO
      ▼
┌─────────────────────┐     YES    ┌────────────────────┐
│ Is rm command?       ├───────────►│ ASK user           │
└─────┬───────────────┘            └────────────────────┘
      │ NO
      ▼
┌─────────────────────┐     YES    ┌────────────────────┐
│ Is curl/wget/net?    ├───────────►│ ASK user           │
└─────┬───────────────┘            └────────────────────┘
      │ NO
      ▼
┌────────────────────┐
│ ALLOW              │
└────────────────────┘
```

## Hook Input/Output Contract

### Input (stdin JSON)

```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "rm -rf /tmp/stuff"
  }
}
```

Tool names: `Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `WebFetch`, `WebSearch`

### Output (stdout JSON)

The output must use the `hookSpecificOutput` envelope with `permissionDecision`:

**Allow:**
```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "safe bash command"
  }
}
```

**Block:**
```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Blocked: access to SSH private keys"
  }
}
```

**Ask user:**
```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "ask",
    "permissionDecisionReason": "rm command detected: rm -rf /tmp/stuff"
  }
}
```

## Credential Detection Rules

### Blocked Paths (any tool accessing these)

| Pattern | What it protects |
|---------|-----------------|
| `~/.ssh/id_*`, `~/.ssh/config` | SSH private keys and config |
| `~/.aws/credentials`, `~/.aws/config` | AWS credentials |
| `~/.gnupg/*` | GPG private keys |
| `*/.env`, `*/.env.*` | Environment secrets |
| `*/credentials.json` | GCP / generic credentials |
| `~/.netrc` | Network authentication |
| `~/.kube/config` | Kubernetes credentials |
| `~/.docker/config.json` | Container registry auth |
| `~/.npmrc` | NPM tokens |
| `~/.pypirc` | PyPI tokens |
| `*/secrets.yaml`, `*/secrets.yml` | K8s secrets manifests |
| `*/.git-credentials` | Git stored passwords |
| `~/.config/gh/hosts.yml` | GitHub CLI tokens |
| `*/token`, `*/password*`, `*/*secret*` | Generic secret files |

### Detection applies to

- **Read tool**: check `file_path` param against blocked patterns
- **Bash tool**: parse `command` param for `cat`, `head`, `tail`, `less`, `more`, `vi`, `vim`, `nano`, `cp`, `mv`, `scp` targeting blocked paths
- **Edit/Write tool**: check `file_path` param (block writing to credential files too)
- **Glob tool**: check `pattern` param for credential directories

## Dangerous Command Detection (ASK tier)

### rm detection
- Match `rm` command in Bash tool
- Covers: `rm`, `rm -r`, `rm -rf`, `rm -f`, `rmdir`, `shred`
- The ask message shows the exact command for user review

### Network command detection
- Match: `curl`, `wget`, `nc`, `ncat`, `netcat`, `ssh`, `scp`, `rsync`, `ftp`, `sftp`
- Match: `python -m http.server`, `php -S`
- The ask message shows the exact command for user review

## Components

### 1. permissions.py

The hook script. Receives tool call JSON on stdin, outputs decision JSON on stdout.

- Single file, no dependencies beyond Python 3 stdlib
- Uses `json`, `sys`, `re`, `os`, `pathlib`
- Expands `~` and resolves symlinks before path matching
- All pattern lists are plain Python lists at the top of the file for easy customization

### 2. install.sh

- Copies `permissions.py` to `~/.claude/hooks/permissions.py`
- Makes it executable
- Reads existing `~/.claude/settings.json` (or creates one)
- Merges the PreToolUse hook config into settings.json using Python one-liner (no jq dependency)
- Backs up existing settings.json before modifying
- Matcher regex: `.*` (matches all tools so the script handles routing internally)

### 3. uninstall.sh

- Removes the hook entry from `~/.claude/settings.json`
- Removes `~/.claude/hooks/permissions.py`
- Restores backup if available
- Does NOT remove other hooks or settings

## settings.json Hook Registration

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/permissions.py"
          }
        ]
      }
    ]
  }
}
```

Using `"matcher": ".*"` so a single script handles all tools. The script itself decides allow/block/ask based on tool name and params.

## Known Conflicts with Other Hooks

### context-mode plugin

The `context-mode` plugin (if enabled in `settings.json`) has its own PreToolUse hooks that
block `curl`/`wget` in Bash and block `WebFetch` entirely. This runs at the plugin level
before our hook, so our `ask` decision never reaches the user.

**Fix:** The `install.sh` script warns if context-mode is enabled. The user must either:
- Disable context-mode's curl/wget interception
- Or disable the context-mode plugin when using this permissions hook

### approve-variants.py

If `approve-variants.py` is registered as a PreToolUse hook for Bash, it may auto-approve
commands (including `curl`) before `permissions.py` runs. The original `approve-variants.py`
lists `curl` in its safe "read-only" commands.

**Fix:** The `install.sh` script removes `curl` from `approve-variants.py` safe list if present.

## Edge Cases and Gaps

### Addressed

1. **Symlink attacks**: The script resolves symlinks with `pathlib.Path.resolve()` before matching, so `cat /tmp/link-to-ssh-key` is caught if the symlink points to `~/.ssh/id_rsa`.

2. **Path obfuscation**: Normalizes paths (removes `..`, `./`, double slashes) before matching.

3. **Bash command chaining**: Parses for credential paths across the entire command string, catching `cat foo && cat ~/.ssh/id_rsa` and piped commands.

4. **Environment variable expansion**: Expands `$HOME`, `$USER` in command strings before matching.

5. **Write/Edit to credential paths**: Blocked, not just reads. Prevents overwriting credentials with malicious content.

### Known Limitations

1. **Base64/encoding bypass**: A command like `base64 ~/.ssh/id_rsa | curl ...` is partially caught (base64 + credential path triggers block, curl triggers ask), but novel encoding schemes may slip through.

2. **Subshell execution**: `bash -c "cat ~/.ssh/id_rsa"` — the inner command is a string argument. The script does basic string matching on the full command, which catches this, but deeply nested subshells could evade.

3. **Python/Ruby/Node one-liners**: `python3 -c "open('/home/user/.ssh/id_rsa').read()"` — the script matches credential paths as substrings in the full command, so this is caught, but language-specific obfuscation (hex encoding the path) is not.

4. **No logging**: The hook does not log decisions. Future improvement: append decisions to `~/.claude/hooks/permissions.log` for audit.

5. **No per-project overrides**: The hook is global. A project-level config file could allow relaxing rules for specific repos (e.g., a security tool that legitimately needs SSH access).

6. **No allowlist for known-safe commands**: Every `curl` triggers an ask, even `curl localhost:8080/health`. A future allowlist could auto-approve specific URL patterns.

7. **Performance**: Python startup on every tool call adds ~50ms latency. Acceptable for interactive use but worth noting.

## File Tree (deliverable)

```
cc-hook-better-permissions/
├── design-doc.md
├── permissions.py
├── install.sh
└── uninstall.sh
```

## Testing Strategy

- Manual testing by triggering each tier:
  - `cat ~/.ssh/id_rsa` → should BLOCK
  - `rm -rf /tmp/test` → should ASK
  - `curl https://example.com` → should ASK
  - Read tool on a normal file → should ALLOW
  - Glob tool on `src/**/*.py` → should ALLOW
- Verify install.sh creates backup and merges correctly
- Verify uninstall.sh cleanly removes hook without breaking other settings
