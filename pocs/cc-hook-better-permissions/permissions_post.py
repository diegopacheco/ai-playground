#!/usr/bin/env python3

import json
import sys
import re
import os
from pathlib import Path

HOME = str(Path.home())
APPROVED_FILE = os.path.join(HOME, ".claude", "hooks", "permissions_approved.json")

NETWORK_COMMANDS = [
    "curl", "wget", "nc", "ncat", "netcat",
    "ssh", "scp", "rsync", "ftp", "sftp",
    "http.server", "php -S",
]

RM_COMMANDS = ["rm", "rmdir", "shred", "unlink"]


def load_approved():
    if os.path.exists(APPROVED_FILE):
        try:
            with open(APPROVED_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_approved(approved):
    os.makedirs(os.path.dirname(APPROVED_FILE), exist_ok=True)
    with open(APPROVED_FILE, "w") as f:
        json.dump(approved, f, indent=2)
        f.write("\n")


def is_network_command(command):
    for cmd in NETWORK_COMMANDS:
        pattern = r'(^|[;&|]\s*)' + re.escape(cmd) + r'(\s|$)'
        if re.search(pattern, command):
            return True
    return False


def is_rm_command(command):
    tokens = command.split()
    for token in tokens:
        if token in ("|", "&&", ";", "||"):
            continue
        clean = token.split("/")[-1]
        if clean in RM_COMMANDS:
            return True
    return False


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}

    if tool_name != "Bash":
        sys.exit(0)

    command = tool_input.get("command", "")
    approved = load_approved()
    changed = False

    if is_network_command(command) and not approved.get("network"):
        approved["network"] = True
        changed = True

    if is_rm_command(command) and not approved.get("rm"):
        approved["rm"] = True
        changed = True

    if changed:
        save_approved(approved)

    sys.exit(0)


if __name__ == "__main__":
    main()
