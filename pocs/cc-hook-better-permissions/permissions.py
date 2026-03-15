#!/usr/bin/env python3

import json
import sys
import re
import os
from pathlib import Path
from fnmatch import fnmatch

HOME = str(Path.home())
APPROVED_FILE = os.path.join(HOME, ".claude", "hooks", "permissions_approved.json")

CREDENTIAL_PATTERNS = [
    os.path.join(HOME, ".ssh", "id_*"),
    os.path.join(HOME, ".ssh", "config"),
    os.path.join(HOME, ".ssh", "known_hosts"),
    os.path.join(HOME, ".ssh", "authorized_keys"),
    os.path.join(HOME, ".aws", "credentials"),
    os.path.join(HOME, ".aws", "config"),
    os.path.join(HOME, ".gnupg"),
    os.path.join(HOME, ".netrc"),
    os.path.join(HOME, ".kube", "config"),
    os.path.join(HOME, ".docker", "config.json"),
    os.path.join(HOME, ".npmrc"),
    os.path.join(HOME, ".pypirc"),
    os.path.join(HOME, ".git-credentials"),
    os.path.join(HOME, ".config", "gh", "hosts.yml"),
]

CREDENTIAL_FILENAMES = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    ".env.development",
    "credentials.json",
    "secrets.yaml",
    "secrets.yml",
    "secret.yaml",
    "secret.yml",
    "token",
    "password",
    "passwords",
    "passwords.txt",
    ".git-credentials",
]

CREDENTIAL_SUBSTRINGS = [
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    ".ssh/",
    ".aws/credentials",
    ".gnupg/",
    ".netrc",
    ".kube/config",
    ".docker/config.json",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "hosts.yml",
]

RM_COMMANDS = ["rm", "rmdir", "shred", "unlink"]

NETWORK_COMMANDS = [
    "curl", "wget", "nc", "ncat", "netcat",
    "ssh", "scp", "rsync", "ftp", "sftp",
    "http.server", "php -S",
]


def load_approved():
    if os.path.exists(APPROVED_FILE):
        try:
            with open(APPROVED_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def is_approved(category):
    approved = load_approved()
    return approved.get(category, False)


def make_allow(reason="auto-approved by permissions hook"):
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": reason
        }
    }


def make_block(reason):
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason
        }
    }


def make_ask(message):
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": message
        }
    }


def normalize_path(p):
    p = p.replace("$HOME", HOME).replace("~", HOME)
    p = os.path.normpath(p)
    try:
        p = str(Path(p).resolve())
    except (OSError, ValueError):
        pass
    return p


def is_credential_path(path_str):
    normalized = normalize_path(path_str)
    for pattern in CREDENTIAL_PATTERNS:
        if fnmatch(normalized, pattern):
            return True
    if normalized.startswith(os.path.join(HOME, ".gnupg")):
        return True
    basename = os.path.basename(normalized)
    if basename in CREDENTIAL_FILENAMES:
        return True
    if basename.startswith(".env"):
        return True
    if "secret" in basename.lower() and not basename.endswith((".py", ".js", ".ts", ".go", ".java", ".rs")):
        return True
    if "password" in basename.lower() and not basename.endswith((".py", ".js", ".ts", ".go", ".java", ".rs")):
        return True
    return False


def command_touches_credentials(command):
    for substr in CREDENTIAL_SUBSTRINGS:
        if substr in command:
            return True
    tokens = command.split()
    for token in tokens:
        if "/" in token or token.startswith("."):
            if is_credential_path(token):
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


def is_network_command(command):
    for cmd in NETWORK_COMMANDS:
        pattern = r'(^|[;&|]\s*)' + re.escape(cmd) + r'(\s|$)'
        if re.search(pattern, command):
            return True
    return False


def check_bash(tool_input):
    command = tool_input.get("command", "")
    if command_touches_credentials(command):
        return make_block("Blocked: command accesses credential/secret files")
    if is_rm_command(command):
        if is_approved("rm"):
            return make_allow("rm previously approved by user")
        return make_ask(f"rm command detected: {command}")
    if is_network_command(command):
        if is_approved("network"):
            return make_allow("network command previously approved by user")
        return make_ask(f"Network command detected: {command}")
    return make_allow("safe bash command")


def check_file_path(tool_input):
    file_path = tool_input.get("file_path", "")
    if file_path and is_credential_path(file_path):
        return make_block(f"Blocked: access to credential file {file_path}")
    return make_allow("safe file path")


def check_glob(tool_input):
    pattern = tool_input.get("pattern", "")
    path = tool_input.get("path", "")
    for check in [pattern, path]:
        if check:
            for substr in CREDENTIAL_SUBSTRINGS:
                if substr in check:
                    return make_block("Blocked: glob pattern targets credential directory")
    return make_allow("safe glob")


def check_grep(tool_input):
    path = tool_input.get("path", "")
    if path and is_credential_path(path):
        return make_block(f"Blocked: grep targets credential file {path}")
    return make_allow("safe grep")


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

    if tool_name == "Bash":
        result = check_bash(tool_input)
    elif tool_name in ("Read", "Write", "Edit"):
        result = check_file_path(tool_input)
    elif tool_name == "Glob":
        result = check_glob(tool_input)
    elif tool_name == "Grep":
        result = check_grep(tool_input)
    else:
        result = make_allow("no restriction for this tool")

    print(json.dumps(result))


if __name__ == "__main__":
    main()
