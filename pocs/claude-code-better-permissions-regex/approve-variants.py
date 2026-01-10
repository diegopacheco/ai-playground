#!/usr/bin/env python3
"""
Claude Code PreToolUse Hook: Compositional Bash Command Approval

PROBLEM
-------
Claude Code's static permission system uses prefix matching:
    "Bash(git diff:*)" matches "git diff --staged" but NOT "git -C /path diff"
    "Bash(timeout 30 pytest:*)" matches that exact timeout, not "timeout 20 pytest"

This leads to frequent permission prompts for safe command variations.

SOLUTION
--------
This hook auto-approves Bash commands that are safe combinations of:
    WRAPPERS (timeout, env vars, .venv/bin/) + CORE COMMANDS (git, pytest, etc.)

Example: "timeout 60 RUST_BACKTRACE=1 cargo test" is approved as:
    wrapper(timeout) + wrapper(env vars) + safe_command(cargo)

CHAINED COMMANDS
----------------
Commands with &&, ||, ;, | are split and ALL segments must be safe:
    "ls && pwd"           -> approved (both safe)
    "ls && rm -rf /"      -> rejected (rm not safe)
    "git diff | head"     -> approved (both safe)

Command substitution ($(...) and backticks) is always rejected.

CONFIGURATION
-------------
Registered in ~/.claude/settings.json:

    "hooks": {
      "PreToolUse": [{
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "python3 ~/.claude/hooks/approve-variants.py"}]
      }]
    }

EXTENDING
---------
To add new safe wrappers: Add to WRAPPER_PATTERNS (regex, name)
To add new safe commands: Add to SAFE_COMMANDS (regex, name)

DEBUG
-----
    echo '{"tool_name": "Bash", "tool_input": {"command": "timeout 30 pytest"}}' | python3 ~/.claude/hooks/approve-variants.py
"""
import json
import sys
import re

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool_name = data.get("tool_name")
tool_input = data.get("tool_input", {})

if tool_name != "Bash":
    sys.exit(0)


def approve(reason):
    """Output approval JSON and exit."""
    result = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": reason
        }
    }
    print(json.dumps(result))
    sys.exit(0)

cmd = tool_input.get("command", "")

# --- Reject dangerous constructs that are hard to parse safely ---
if re.search(r"\$\(|`", cmd):
    sys.exit(0)


def split_command_chain(cmd):
    """Split command into segments on &&, ||, ;, |.

    Note: We don't split on newlines if:
    - Quotes are present (multiline strings like python -c "...")
    - Backslash continuations are present (cmd \\\n  --flag)
    """
    # First, collapse backslash-newline continuations
    cmd = re.sub(r"\\\n\s*", " ", cmd)

    # Protect quoted strings from splitting (replace with placeholders)
    quoted_strings = []
    def save_quoted(m):
        quoted_strings.append(m.group(0))
        return f"__QUOTED_{len(quoted_strings)-1}__"
    cmd = re.sub(r'"[^"]*"', save_quoted, cmd)
    cmd = re.sub(r"'[^']*'", save_quoted, cmd)

    # Normalize redirections to prevent splitting on & in 2>&1
    cmd = re.sub(r"(\d*)>&(\d*)", r"__REDIR_\1_\2__", cmd)
    cmd = re.sub(r"&>", "__REDIR_AMPGT__", cmd)

    # Split on command separators: &&, ||, ;, |, & (background)
    if quoted_strings:
        segments = re.split(r"\s*(?:&&|\|\||;|\||&)\s*", cmd)
    else:
        segments = re.split(r"\s*(?:&&|\|\||;|\||&)\s*|\n", cmd)

    # Restore quoted strings and redirections
    def restore(s):
        s = re.sub(r"__REDIR_(\d*)_(\d*)__", r"\1>&\2", s)
        s = s.replace("__REDIR_AMPGT__", "&>")
        for i, qs in enumerate(quoted_strings):
            s = s.replace(f"__QUOTED_{i}__", qs)
        return s
    segments = [restore(s) for s in segments]
    return [s.strip() for s in segments if s.strip()]


# --- Safe wrappers that can prefix any safe command ---
WRAPPER_PATTERNS = [
    (r"^timeout\s+\d+\s+", "timeout"),
    (r"^nice\s+(-n\s*\d+\s+)?", "nice"),
    (r"^env\s+", "env"),
    (r"^([A-Z_][A-Z0-9_]*=[^\s]*\s+)+", "env vars"),
    # Virtual env paths: .venv/bin/, ../.venv/bin/, /abs/path/.venv/bin/, venv/bin/
    (r"^(\.\./)*\.?venv/bin/", ".venv"),
    (r"^/[^\s]+/\.?venv/bin/", ".venv"),
    # do (loop body prefix)
    (r"^do\s+", "do"),
]

# --- Safe core command patterns ---
SAFE_COMMANDS = [
    # git read operations (with optional -C flag)
    (r"^git\s+(-C\s+\S+\s+)?(diff|log|status|show|branch|stash\s+list|bisect|worktree\s+list|fetch)\b",
     "git read op"),
    # git write operations
    (r"^git\s+(-C\s+\S+\s+)?(add|checkout|merge|rebase|stash)\b",
     "git write op"),
    # pytest
    (r"^pytest\b", "pytest"),
    # python
    (r"^python\b", "python"),
    # ruff (python linter/formatter)
    (r"^ruff\b", "ruff"),
    # uv / uvx
    (r"^uv\s+(pip|run|sync|venv|add|remove|lock)\b", "uv"),
    (r"^uvx\b", "uvx"),
    # npm / npx
    (r"^npm\s+(install|run|test|build|ci)\b", "npm"),
    (r"^npx\b", "npx"),
    # cargo
    (r"^cargo\s+(build|test|run|check|clippy|fmt|clean)\b", "cargo"),
    # maturin (rust python bindings)
    (r"^maturin\s+(develop|build)\b", "maturin"),
    # make
    (r"^make\b", "make"),
    # common read-only commands
    (r"^(ls|cat|head|tail|wc|find|grep|rg|file|which|pwd|du|df|curl|sort|uniq|cut|tr|awk|sed|xargs)\b", "read-only"),
    # touch (update timestamps, create empty files)
    (r"^touch\b", "touch"),
    # shell builtins for control flow
    (r"^(true|false|exit(\s+\d+)?)$", "shell builtin"),
    # pkill/kill (process management)
    (r"^(pkill|kill)\b", "process mgmt"),
    # echo (often used for logging/separators in chained commands)
    (r"^echo\b", "echo"),
    # cd (change directory, often first in a chain)
    (r"^cd\s", "cd"),
    # source/. (activate scripts, set env)
    (r"^(source|\.) [^\s]*venv/bin/activate", "venv activate"),
    # sleep (delays, often used in scripts)
    (r"^sleep\s", "sleep"),
    # variable assignment (VAR=value, VAR=$!, etc.)
    (r"^[A-Z_][A-Z0-9_]*=\S*$", "var assignment"),
    # for/while loops and loop constructs
    (r"^for\s+\w+\s+in\s", "for loop"),
    (r"^while\s", "while loop"),
    (r"^done$", "done"),
]


def strip_wrappers(cmd):
    """Strip safe wrapper prefixes, return (core_cmd, list_of_wrappers)."""
    wrappers = []
    changed = True
    while changed:
        changed = False
        for pattern, name in WRAPPER_PATTERNS:
            m = re.match(pattern, cmd)
            if m:
                wrappers.append(name)
                cmd = cmd[m.end():]
                changed = True
                break
    return cmd.strip(), wrappers


def check_safe(cmd):
    """Check if command matches a safe pattern. Returns reason or None."""
    for pattern, reason in SAFE_COMMANDS:
        if re.match(pattern, cmd):
            return reason
    return None


# --- Main Bash logic ---
segments = split_command_chain(cmd)

reasons = []
for segment in segments:
    core_cmd, wrappers = strip_wrappers(segment)
    reason = check_safe(core_cmd)
    if not reason:
        sys.exit(0)  # One unsafe segment = reject entire command
    if wrappers:
        reasons.append(f"{'+'.join(wrappers)} + {reason}")
    else:
        reasons.append(reason)

approve(" | ".join(reasons))